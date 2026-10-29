import base64
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from echo4d.glb_assets import pack_images
from echo4d.model import DataError
from echo4d.video_scene import verify_textured_glb


def external_fixture(path,uri='albedo.png',extra_buffer=False):
    trimesh=pytest.importorskip('trimesh')
    from PIL import Image
    mesh=trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]],faces=[[0,1,2]],process=False)
    mesh.visual=trimesh.visual.texture.TextureVisuals(uv=[[0,0],[1,0],[0,1]],image=Image.new('RGB',(4,4),'red'))
    raw=mesh.export(file_type='glb')
    size=struct.unpack_from('<I',raw,12)[0]
    doc=json.loads(raw[20:20+size]);binary=raw[20+size:]
    image=doc['images'][0];view=doc['bufferViews'][image.pop('bufferView')]
    start=view.get('byteOffset',0)
    (path.parent/'albedo.png').write_bytes(binary[8+start:8+start+view['byteLength']])
    image['uri']=uri
    if extra_buffer:
        doc['buffers'].append({'byteLength':4,'uri':'data:application/octet-stream;base64,'+base64.b64encode(b'abcd').decode()})
        doc['bufferViews'].append({'buffer':1,'byteLength':4})
    payload=json.dumps(doc).encode();payload+=b' '*(-len(payload)%4)
    path.write_bytes(struct.pack('<III',0x46546c67,2,20+len(payload)+len(binary))+struct.pack('<II',len(payload),0x4e4f534a)+payload+binary)
    return trimesh


@pytest.mark.parametrize('extra_buffer',[False,True])
def test_pack_external_png_preserves_geometry_and_texture(tmp_path,extra_buffer):
    path=tmp_path/'scene.glb'
    trimesh=external_fixture(path,extra_buffer=extra_buffer)
    before=trimesh.load(path,force='scene',process=False)
    receipt=pack_images(path)
    assert receipt['existing_buffer_views_unchanged']
    assert verify_textured_glb(path)['triangles']==1
    after=trimesh.load(path,force='scene',process=False)
    a=list(before.geometry.values())[0];b=list(after.geometry.values())[0]
    np.testing.assert_array_equal(a.vertices,b.vertices)
    np.testing.assert_array_equal(a.faces,b.faces)
    np.testing.assert_array_equal(a.visual.uv,b.visual.uv)
    np.testing.assert_array_equal(np.asarray(a.visual.material.baseColorTexture),np.asarray(b.visual.material.baseColorTexture))
    assert pack_images(path)['already_embedded']


def test_reject_texture_path_escape(tmp_path):
    path=tmp_path/'scene.glb'
    external_fixture(path,uri='../secret.png')
    with pytest.raises(DataError):
        pack_images(path)
