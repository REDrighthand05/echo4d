"""Actual inferred-input preservation, using synthetic bytes only."""
import json
import numpy as np
import pytest
from PIL import Image
import tum_rgb_stereo as stereo
from echo4d.model import DataError, file_sha256


def setup(tmp_path):
    root=tmp_path/'source';root.mkdir()
    for name in ('rgb.txt','depth.txt','groundtruth.txt'):
        (root/name).write_bytes(b'# original index\r\n')
    rgb=root/'rgb.png';Image.fromarray(np.zeros((4,4,3),np.uint8)).save(rgb)
    target=tmp_path/'stereo';(target/'depth').mkdir(parents=True);(target/'color').mkdir()
    Image.fromarray(np.zeros((4,4,3),np.uint8)).save(target/'color/00000.png')
    np.save(target/'depth/00000.npy',np.ones((4,4),np.float32))
    manifest=target/'recording.json'
    manifest.write_bytes(b'{ "frames": [{"depth":"depth/00000.npy", "color":"color/00000.png"}] }\r\n')
    frames=[{'index':0,'t_s':0.,'rgb':rgb,'rgb_sha256':file_sha256(rgb),'pose':np.eye(4)}]
    return manifest,frames,root,tmp_path/'archive'


def test_stereo_replay_retains_original_manifest_depth_pose_and_index(tmp_path):
    manifest,frames,root,out=setup(tmp_path)
    receipt=stereo.archive_stereo_inputs(manifest,frames,root,out)
    assert (out/'recording.json').read_bytes()==manifest.read_bytes()
    assert (out/'depth/00000.npy').read_bytes()==(manifest.parent/'depth/00000.npy').read_bytes()
    for name,pin in receipt.items():
        assert file_sha256(out/name)==pin['sha256']
    assert (out/'source-index/rgb.txt').read_bytes()==(root/'rgb.txt').read_bytes()
    data=json.loads((out/'source-frames.json').read_text())
    assert data['frames'][0]['cameraToWorld']==np.eye(4).flatten(order='F').tolist()
    assert json.loads((out/'archive.json').read_text())['depth_kind']=='stereo_inferred_camera_z_not_measured'
    assert not (out/'color').exists()
    with pytest.raises(DataError):
        stereo.archive_stereo_inputs(manifest,frames,root,out)


def test_stereo_changed_input_refuses_publication(tmp_path,monkeypatch):
    manifest,frames,root,out=setup(tmp_path);copy=stereo.shutil.copyfile
    def mutate(source,target):
        result=copy(source,target)
        if source==manifest:source.write_bytes(source.read_bytes()+b' ')
        return result
    monkeypatch.setattr(stereo.shutil,'copyfile',mutate)
    with pytest.raises(DataError,match='changed'):
        stereo.archive_stereo_inputs(manifest,frames,root,out)
    assert not out.exists()


def test_stereo_escaping_depth_path_rejected(tmp_path):
    manifest,frames,root,out=setup(tmp_path)
    manifest.write_text(json.dumps({'frames':[{'depth':'../stereo/depth/00000.npy','color':'color/00000.png'}]}))
    with pytest.raises(DataError,match='Unsafe'):
        stereo.archive_stereo_inputs(manifest,frames,root,out)
    assert not out.exists()


def test_stereo_changed_source_rgb_rejected(tmp_path):
    manifest,frames,root,out=setup(tmp_path);frames[0]['rgb_sha256']='0'*64
    with pytest.raises(DataError,match='Source RGB changed'):
        stereo.archive_stereo_inputs(manifest,frames,root,out)
    assert not out.exists()
