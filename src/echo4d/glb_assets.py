"""Pack OpenMVS 2.3's external texture PNGs into a portable GLB.

Only buffer placement changes: every existing buffer view is byte-compared.
"""
import base64
import hashlib
import json
from pathlib import Path
import shutil
import struct
from urllib.parse import unquote, urlsplit

from .model import DataError, file_sha256, safe_path


def pack_images(path):
    path=Path(path)
    original=path.read_bytes()
    if len(original)<20 or struct.unpack_from('<III',original)!=(0x46546c67,2,len(original)):
        raise DataError('Invalid GLB header while packing textures')
    chunks={}
    offset=12
    while offset<len(original):
        if offset+8>len(original):
            raise DataError('Truncated GLB chunk')
        size,kind=struct.unpack_from('<II',original,offset)
        offset+=8
        if offset+size>len(original) or kind in chunks:
            raise DataError('Invalid or duplicate GLB chunk')
        chunks[kind]=original[offset:offset+size]
        offset+=size
    doc=json.loads(chunks[0x4e4f534a])
    if len(doc.get('buffers',[]))==1 and 'uri' not in doc['buffers'][0] and all('bufferView' in im for im in doc.get('images',[])):
        return {'already_embedded':True,'sha256':file_sha256(path)}

    def read_uri(uri):
        if uri.startswith('data:'):
            head,value=uri.split(',',1)
            if not head.endswith(';base64'):
                raise DataError('Only base64 embedded resources are supported')
            return base64.b64decode(value,validate=True)
        parsed=urlsplit(uri)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
            raise DataError('External network or ambiguous texture URI rejected')
        return safe_path(path.parent,unquote(parsed.path)).read_bytes()

    buffers=[]
    for index,buffer in enumerate(doc.get('buffers',[])):
        data=read_uri(buffer['uri']) if 'uri' in buffer else chunks.get(0x004e4942,b'') if index==0 else b''
        size=buffer['byteLength']
        if not isinstance(size,int) or size<0 or len(data)<size:
            raise DataError('Missing GLB buffer data')
        buffers.append(data[:size])
    merged=bytearray()
    offsets=[]
    for data in buffers:
        merged.extend(b'\0'*(-len(merged)%4))
        offsets.append(len(merged))
        merged.extend(data)
    old_views=[]
    for view in doc.get('bufferViews',[]):
        if 'EXT_meshopt_compression' in view.get('extensions',{}):
            raise DataError('Compressed geometry buffers need a dedicated packer')
        index=view.get('buffer',0)
        start=view.get('byteOffset',0)
        size=view['byteLength']
        if index>=len(buffers) or index<0 or start<0 or size<0 or start+size>len(buffers[index]):
            raise DataError('Invalid GLB buffer view')
        old_views.append(buffers[index][start:start+size])
        view['buffer']=0
        view['byteOffset']=start+offsets[index]
    for image in doc.get('images',[]):
        if 'uri' not in image:
            continue
        data=read_uri(image.pop('uri'))
        if data.startswith(b'\x89PNG\r\n\x1a\n'):
            mime='image/png'
        elif data.startswith(b'\xff\xd8'):
            mime='image/jpeg'
        else:
            raise DataError('Texture is not a PNG/JPEG image')
        merged.extend(b'\0'*(-len(merged)%4))
        image.update(bufferView=len(doc['bufferViews']),mimeType=mime)
        doc['bufferViews'].append({'buffer':0,'byteOffset':len(merged),'byteLength':len(data)})
        merged.extend(data)
    for view,expected in zip(doc['bufferViews'],old_views):
        start=view['byteOffset']
        if bytes(merged[start:start+view['byteLength']])!=expected:
            raise DataError('Packing changed existing geometry buffer data')
    doc['buffers']=[{'byteLength':len(merged)}]
    payload=json.dumps(doc,separators=(',',':')).encode()
    payload+=b' '*(-len(payload)%4)
    merged.extend(b'\0'*(-len(merged)%4))
    result=struct.pack('<III',0x46546c67,2,28+len(payload)+len(merged))+struct.pack('<II',len(payload),0x4e4f534a)+payload+struct.pack('<II',len(merged),0x004e4942)+merged
    backup=path.with_name(path.stem+'.external-textures.glb')
    if backup.exists():
        raise DataError('Raw GLB backup already exists; refusing overwrite')
    shutil.copyfile(path,backup)
    path.write_bytes(result)
    return {'original_glb_sha256':hashlib.sha256(original).hexdigest(),
            'packed_glb_sha256':file_sha256(path),'existing_buffer_views_unchanged':True,
            'embedded_images':len(doc.get('images',[]))}
