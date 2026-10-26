"""Fail-closed observed furniture surface extraction from externally reviewed RGB-D masks.

No AI generated faces, no connected-component semantic guesses, no mesh-scale
changes. Original faces are kept only if every vertex is positively supported
in TWO source RGB/mask/metric-depth views without foreground contradictions.
This script cannot authenticate the truth of a human mask review declaration.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import trimesh

class EvidenceError(ValueError):
    pass

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(1048576),b''):
            h.update(chunk)
    return h.hexdigest()

def asset(root,spec,suffixes):
    if not isinstance(spec,dict) or set(spec)!={'path','sha256'}:
        raise EvidenceError('Every asset needs path and SHA256')
    name=spec['path']
    if not isinstance(name,str) or not name or '\\' in name:
        raise EvidenceError('Invalid asset path')
    relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise EvidenceError('Traversal/absolute path forbidden')
    file=(root/relative).resolve()
    if not file.is_relative_to(root) or not file.is_file() or file.suffix.lower() not in suffixes:
        raise EvidenceError('Missing/escaped/wrong-type asset')
    if not 0<file.stat().st_size<=80*1024**2:
        raise EvidenceError('Empty/oversized input')
    if not isinstance(spec['sha256'],str) or len(spec['sha256'])!=64 or sha256(file)!=spec['sha256']:
        raise EvidenceError('SHA256 mismatch: '+name)
    return file

def load_mesh(file):
    m=trimesh.load_mesh(file,process=False)
    if not isinstance(m,trimesh.Trimesh):
        raise EvidenceError('Source must be a triangular mesh, not a scene')
    v=np.asarray(m.vertices,dtype=np.float64)
    f=np.asarray(m.faces,dtype=np.int64)
    if (v.ndim!=2 or v.shape[1]!=3 or not 3<=len(v)<=2000000 or
            f.ndim!=2 or f.shape[1]!=3 or not 1<=len(f)<=3000000 or
            not np.isfinite(v).all() or f.min()<0 or f.max()>=len(v)):
        raise EvidenceError('Invalid scene mesh')
    return v,f

def read_frame(root,frame):
    names={'name','rgb','mask','depth','camera_to_world_column_major',
           'fx','fy','cx','cy','mask_annotation_source','reviewed_same_object'}
    if not isinstance(frame,dict) or set(frame)!=names:
        raise EvidenceError('Frame proof fields missing or unknown')
    if frame['reviewed_same_object'] is not True or not isinstance(frame['mask_annotation_source'],str) or len(frame['mask_annotation_source'])<12:
        raise EvidenceError('Missing photo-to-instance annotation review')
    rgb=asset(root,frame['rgb'],('.png','.jpg','.jpeg'))
    maskfile=asset(root,frame['mask'],('.png',))
    depthfile=asset(root,frame['depth'],('.npy',))
    with Image.open(rgb) as im:
        size=im.size
        im.verify()
    with Image.open(maskfile) as im:
        if im.mode not in ('L','1'):
            raise EvidenceError('Mask must be grayscale')
        mask=np.asarray(im.convert('L')).copy()
    if mask.shape!=(size[1],size[0]) or not np.isin(mask,[0,255]).all() or np.count_nonzero(mask==255)<10:
        raise EvidenceError('Missing/bad binary instance mask or image dimensions')
    depth=np.load(depthfile,allow_pickle=False)
    if (depth.shape!=mask.shape or depth.dtype.kind!='f' or
            not np.isfinite(depth).all() or np.any(depth<0) or np.max(depth)>30):
        raise EvidenceError('Invalid registered measured camera-Z depth in meters')
    k=np.asarray([frame[x] for x in ('fx','fy','cx','cy')],dtype=float)
    if not np.isfinite(k).all() or min(k[:2])<=0:
        raise EvidenceError('Invalid intrinsics')
    pose=frame['camera_to_world_column_major']
    if not isinstance(pose,list) or len(pose)!=16:
        raise EvidenceError('Missing column-major camera pose')
    m=np.asarray(pose,dtype=float).reshape(4,4,order='F')
    r=m[:3,:3]
    if (not np.isfinite(m).all() or not np.allclose(m[3],[0,0,0,1],atol=1e-8) or
            not np.allclose(r.T@r,np.eye(3),atol=1e-5) or abs(np.linalg.det(r)-1)>1e-5):
        raise EvidenceError('Invalid nonrigid/mirrored camera pose')
    return size,mask,depth,k,m

def vote(vertices,data,tolerance):
    (width,height),mask,depth,k,pose=data
    local=(vertices-pose[:3,3])@pose[:3,:3]
    z=local[:,2]
    ids=np.flatnonzero(np.isfinite(z)&(z>0.05))
    fg=np.zeros(len(vertices),bool)
    bg=np.zeros(len(vertices),bool)
    if not len(ids):
        return fg,bg
    x=np.rint(k[0]*local[ids,0]/z[ids]+k[2]).astype(np.int64)
    y=np.rint(k[1]*local[ids,1]/z[ids]+k[3]).astype(np.int64)
    ok=(x>=0)&(x<width)&(y>=0)&(y<height)
    ids,x,y=ids[ok],x[ok],y[ok]
    matches=(depth[y,x]>0)&(np.abs(depth[y,x]-z[ids])<=tolerance)
    fg[ids[matches&(mask[y,x]==255)]]=True
    bg[ids[matches&(mask[y,x]==0)]]=True
    return fg,bg

def write_compatible_candidate_ply(path, vertices, faces):
    """ECHO-reader-compatible interchange; normals/color are DERIVED, not measured."""
    if len(vertices)>300000 or len(faces)>500000:
        raise EvidenceError('Object exceeds strict downstream ECHO reader bounds')
    v32=vertices.astype(np.float32)
    if not np.isfinite(v32).all():
        raise EvidenceError('Geometry exceeds float32 metric interchange range')
    normals=np.asarray(trimesh.Trimesh(vertices=vertices,faces=faces,process=False).vertex_normals,
                       dtype=np.float32)
    if normals.shape != vertices.shape or not np.isfinite(normals).all():
        raise EvidenceError('Cannot derive finite normals for interchange')
    dtype=[(k,'<f4') for k in ('x','y','z','nx','ny','nz')]
    dtype += [(k,'u1') for k in ('red','green','blue')]
    encoded=np.empty(len(vertices),dtype=dtype)
    for j,k in enumerate(('x','y','z')):
        encoded[k]=v32[:,j]
    for j,k in enumerate(('nx','ny','nz')):
        encoded[k]=normals[:,j]
    for k in ('red','green','blue'):
        encoded[k]=128  # neutral placeholder, NEVER measured RGB
    triangle=np.empty(len(faces),dtype=[('count','u1'),('indices','<i4',(3,))])
    triangle['count']=3
    triangle['indices']=faces
    header=('ply\\nformat binary_little_endian 1.0\\n'
            'comment ECHO candidate interchange; derived normals and synthetic neutral color\\n'
            f'element vertex {len(vertices)}\\n'
            'property float x\\nproperty float y\\nproperty float z\\n'
            'property float nx\\nproperty float ny\\nproperty float nz\\n'
            'property uchar red\\nproperty uchar green\\nproperty uchar blue\\n'
            f'element face {len(faces)}\\n'
            'property list uchar int vertex_indices\\nend_header\\n')
    with path.open('xb') as stream:
        stream.write(header.replace('\\n','\n').encode('ascii'))
        stream.write(encoded.tobytes())
        stream.write(triangle.tobytes())


def extract(manifest,root,output):
    root=Path(root).resolve(strict=True)
    manifest=Path(manifest).resolve(strict=True)
    output=Path(output)
    if output.exists():
        raise EvidenceError('Output already exists; no overwrite')
    if not manifest.is_relative_to(root) or manifest.stat().st_size>1024**2:
        raise EvidenceError('Unsafe/oversized manifest')
    spec=json.loads(manifest.read_text(encoding='utf-8'))
    fields={'version','scene','object_category','instance_id','scale_basis','reviewer','frames','depth_tolerance_m'}
    if not isinstance(spec,dict) or set(spec)!=fields or spec['version']!='echo-furniture-evidence-1':
        raise EvidenceError('Invalid furniture evidence schema')
    if spec['object_category'] not in ('chair','table','sofa','cabinet') or not isinstance(spec['instance_id'],str) or len(spec['instance_id'])<4:
        raise EvidenceError('A single named furniture instance is required')
    if spec['scale_basis'] not in ('externally_measured_sensor_depth_m','verified_metric_reconstruction'):
        raise EvidenceError('No independent meter scale provenance')
    if not isinstance(spec['reviewer'],str) or len(spec['reviewer'].strip())<5:
        raise EvidenceError('Missing human review provenance')
    tolerance=spec['depth_tolerance_m']
    if isinstance(tolerance,bool) or not isinstance(tolerance,(float,int)) or not .001<=tolerance<=.05:
        raise EvidenceError('Depth tolerance must be between 1mm and 5cm')
    frames=spec['frames']
    if not isinstance(frames,list) or not 2<=len(frames)<=30 or any(not isinstance(f,dict) for f in frames) or len({f.get('name') for f in frames})!=len(frames):
        raise EvidenceError('Need 2-30 unique reviewed RGB-D views')
    source=asset(root,spec['scene'],('.ply',))
    vertices,faces=load_mesh(source)
    positive=np.zeros(len(vertices),np.uint16)
    negative=np.zeros(len(vertices),np.uint16)
    receipts=[]
    seen_rgb_paths=set()
    seen_depth_paths=set()
    seen_rgb_hashes=set()
    seen_depth_hashes=set()
    camera_views=[]
    for frame in frames:
        rgb_path=asset(root,frame['rgb'],('.png','.jpg','.jpeg'))
        depth_path=asset(root,frame['depth'],('.npy',))
        rgb_hash=frame['rgb']['sha256']
        depth_hash=frame['depth']['sha256']
        # Different manifest names or byte-for-byte copies never count as new evidence.
        if (str(rgb_path) in seen_rgb_paths or str(depth_path) in seen_depth_paths or
                rgb_hash in seen_rgb_hashes or depth_hash in seen_depth_hashes):
            raise EvidenceError('Duplicate/aliased RGB-D observation; two source captures required')
        data=read_frame(root,frame)
        pose=data[-1]
        if any(np.allclose(pose,previous,atol=1e-6) for previous in camera_views):
            raise EvidenceError('Identical camera viewpoint cannot count as second independent view')
        seen_rgb_paths.add(str(rgb_path))
        seen_depth_paths.add(str(depth_path))
        seen_rgb_hashes.add(rgb_hash)
        seen_depth_hashes.add(depth_hash)
        camera_views.append(pose)
        fg,bg=vote(vertices,data,tolerance)
        positive+=fg
        negative+=bg
        receipts.append({'name':frame['name'],'rgb_sha256':frame['rgb']['sha256'],
                         'mask_sha256':frame['mask']['sha256'],'depth_sha256':frame['depth']['sha256'],
                         'support_vertices':int(fg.sum()),'contradiction_vertices':int(bg.sum())})
    chosen=np.flatnonzero(((positive>=2)&(negative==0))[faces].all(axis=1))
    if len(chosen)==0:
        raise EvidenceError('No original mesh triangle has two consistent instance RGB-D views')
    selected=faces[chosen]
    used,rev=np.unique(selected.reshape(-1),return_inverse=True)
    v,f=vertices[used],rev.reshape(-1,3)
    output.mkdir(parents=True,exist_ok=False)
    try:
        write_compatible_candidate_ply(output/'observed-object-world.ply',v,f)
        np.save(output/'selected-source-face-indices.npy',chosen)
        report={'status':'candidate_requires_independent_instance_review',
            'verified_furniture':False,'hi3d_inference_run':False,'occluded_shape_inferred':False,
            'source_scene_sha256':spec['scene']['sha256'],'source_manifest_sha256':sha256(manifest),
            'output_mesh_sha256':sha256(output/'observed-object-world.ply'),
            'category_declared':spec['object_category'],'instance_id':spec['instance_id'],
            'source_mesh_face_count':len(faces),'kept_original_faces':len(chosen),
            'source_world_bounds_as_declared_m':[vertices.min(axis=0).tolist(),vertices.max(axis=0).tolist()],
            'object_world_bounds_as_declared_m':[v.min(axis=0).tolist(),v.max(axis=0).tolist()],
            'object_extent_as_declared_m':(v.max(axis=0)-v.min(axis=0)).tolist(),
            'world_transform':'identity_no_rescaling_no_repositioning','photo_mask_depth':receipts,
            'original_faces_preserved':True,
            'mesh_interchange':'echo_reader_schema_candidate_only',
            'normals':'derived_from_candidate_faces_not_measured',
            'colors':'neutral_placeholder_not_measured',
            'world_frame_id':'inherited_source_scene_frame_without_external_certification',
            'limitations':['External mask review and measured-scale declarations are not authenticated by software',
                           'No missing faces generated; this is an unclosed observed-only surface',
                           'No real furniture has been independently verified by a synthetic unit test']}
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        return report
    except BaseException:
        import shutil
        shutil.rmtree(output,ignore_errors=True)
        raise

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    try:
        result=extract(args.manifest,args.root,args.output)
        print(json.dumps({'status':result['status'],'faces':result['kept_original_faces']}))
        return 0
    except (EvidenceError,OSError,ValueError,TypeError,KeyError,RuntimeError) as err:
        print(json.dumps({'status':'rejected','error':str(err)},ensure_ascii=False),file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
