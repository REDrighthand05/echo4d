"""Inspect real ECHO meshes and prepare *explicitly selected* single-object Hi3D input.

Not an automatic object recognizer; not a Hi3D model. Never certifies real
measurements, never writes into the source scene. Uses optional trimesh.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import trimesh
from PIL import Image

MAX_BYTES=50*1024*1024
MAX_VERTICES=2_000_000
MAX_FACES=3_000_000

class IntakeError(ValueError):
    pass


def digest(path:Path)->str:
    with path.open('rb') as f:
        h=hashlib.sha256()
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def validated_path(path:Path, extensions:tuple[str,...])->Path:
    p=Path(path).resolve(strict=True)
    if not p.is_file() or p.suffix.lower() not in extensions or not 0<p.stat().st_size<MAX_BYTES:
        raise IntakeError(f'Unsupported/missing/oversized file: {p.name}')
    return p


def mesh_from_file(path:Path)->trimesh.Trimesh:
    p=validated_path(path,('.ply','.glb','.obj'))
    # PLY from ECHO is documented in meters; other formats must be supplied
    # with independently established unit provenance, not guessed here.
    try:
        mesh=trimesh.load_mesh(p,process=False)
    except Exception as exc:
        raise IntakeError('Could not parse triangular mesh') from exc
    if not isinstance(mesh,trimesh.Trimesh):
        raise IntakeError('Multiple scene meshes need an explicit object selection')
    v=np.asarray(mesh.vertices)
    f=np.asarray(mesh.faces)
    if (v.ndim!=2 or v.shape[1]!=3 or len(v)<3 or len(v)>MAX_VERTICES
        or f.ndim!=2 or f.shape[1]!=3 or not 1<=len(f)<=MAX_FACES
        or not np.isfinite(v).all() or f.min()<0 or f.max()>=len(v)):
        raise IntakeError('Invalid mesh points or faces')
    if np.any(np.ptp(v.astype(np.float64),axis=0)>2000):
        raise IntakeError('Source mesh bounds exceed safety limits')
    return mesh


def inventory(path:Path)->dict:
    p=validated_path(path,('.ply','.glb','.obj'))
    m=mesh_from_file(p)
    components=m.split(only_watertight=False)
    faces=sorted((len(c.faces) for c in components),reverse=True)
    bounds=np.asarray(m.bounds,dtype=float)
    return {'version':'echo-hi3d-inventory-1','source_mesh':str(p.name),
            'source_mesh_sha256':digest(p),'vertices':int(len(m.vertices)),
            'triangles':int(len(m.faces)),
            'bounds_as_encoded':bounds.tolist(),
            'bounds_extent':(bounds[1]-bounds[0]).tolist(),
            'face_connected_component_count':len(components),
            'largest_component_face_counts':faces[:12],
            'scene_units_from_input_not_verified':True,
            'is_a_verified_standalone_furniture_object':False,
            'decision':'requires_furniture_mask_and_matching_real_image',
            'reason':'Connected components are geometry, not evidence of object identity; no furniture isolation or reference image was supplied.'}


def validate_selection(record:Path, scene:Path, obj:Path, image:Path)->dict:
    p=validated_path(record,('.json',))
    try:
        s=json.loads(p.read_text(encoding='utf-8'))
    except (UnicodeError,ValueError) as exc:
        raise IntakeError('Invalid selection record') from exc
    expected={'version','source_scene_sha256','object_mesh_sha256','reference_image_sha256',
              'selection_provenance','scale_evidence','reviewed_as_single_object','reference_image_same_object'}
    if not isinstance(s,dict) or set(s)!=expected or s['version']!='echo-hi3d-selection-1':
        raise IntakeError('Missing/extra selection proof fields')
    for key,p in [('source_scene_sha256',scene),('object_mesh_sha256',obj),('reference_image_sha256',image)]:
        if s[key]!=digest(p):
            raise IntakeError('File hash mismatch: '+key)
    if s['reviewed_as_single_object'] is not True or s['reference_image_same_object'] is not True:
        raise IntakeError('Selection must be explicitly reviewed as the same isolated object')
    if not isinstance(s['selection_provenance'],str) or len(s['selection_provenance'].strip())<24:
        raise IntakeError('No auditable selection evidence')
    if s['scale_evidence'] not in ('independent_sensor_depth_m','physically_measured_m'):
        raise IntakeError('No independent physical scale provenance: cannot claim metre placement')
    return s


def prepare(scene:Path, obj:Path, image:Path, selection:Path, out:Path)->dict:
    """Normalize a *reviewed* real-object coarse mesh, preserving inverse gauge.

    Hi3D itself recenters its generated output. This inverse coarse transform
    is only an INITIAL PLACEMENT PROPOSAL: it does not prove corresponding
    surfaces or restore true geometry after generative refinement.
    """
    scene=validated_path(scene,('.ply',))
    obj=validated_path(obj,('.ply','.glb','.obj'))
    image=validated_path(image,('.png','.jpg','.jpeg'))
    if scene==obj:
        raise IntakeError('Whole ECHO scene is not a verified single object')
    evidence=validate_selection(selection,scene,obj,image)
    _=mesh_from_file(scene)
    m=mesh_from_file(obj)
    with Image.open(image) as im:
        im.verify()
    v=np.asarray(m.vertices,dtype=np.float64)
    center=v.mean(axis=0)
    radius=float(np.max(np.linalg.norm(v-center,axis=1)))
    if not np.isfinite(radius) or not 1e-5<radius<100:
        raise IntakeError('Degenerate/implausible coarse object radius')
    vn=(v-center)/radius
    if not np.all(np.isfinite(vn)):
        raise IntakeError('Invalid normalization')
    if out.exists():
        raise IntakeError('Refusing to overwrite existing output')
    if out.resolve()==scene or out.resolve()==obj:
        raise IntakeError('Refusing to overwrite source mesh')
    out.mkdir(parents=True,exist_ok=False)
    try:
        coarse=trimesh.Trimesh(vertices=vn,faces=m.faces.copy(),process=False)
        coarse.export(out/'coarse-normalized.glb')
        shutil.copyfile(image,out/('reference-image'+image.suffix.lower()))
        world=np.eye(4);world[:3,:3]*=radius;world[:3,3]=center
        np.testing.assert_allclose(vn*radius+center,v,atol=1e-9)
        report={
            'version':'echo-hi3d-single-object-candidate-1',
            'kind':'candidate_intake_not_gpu_inference',
            'hi3d_was_run':False,'independent_holdout_was_evaluated':False,
            'source_scene_sha256':digest(scene),'selected_object_mesh_sha256':digest(obj),
            'reference_image_sha256':digest(image),
            'selection_record_sha256':digest(selection),
            'coarse_normalized_glb_sha256':digest(out/'coarse-normalized.glb'),
            'coarse_unit_to_echo_world_column_major':world.flatten(order='F').tolist(),
            'coarse_world_mean_m':center.tolist(),'coarse_max_radius_m':radius,
            'initial_placement_source':'selected coarse mesh only, NOT heldout distances',
            'selection_provenance_claim_not_independently_authenticated':evidence['selection_provenance'],
            'physical_scale_evidence_claim_not_independently_authenticated':evidence['scale_evidence'],
            'status':'prepared_for_optional_gpu_refinement_not_verified',
            'limitations':['Hi3D may change object silhouette, centroid and dimensions after output normalization.',
                           'This transform is a coarse-world placement proposal only; do not label generated geometry as measured truth.',
                           'Candidate requires a GPU inference run and separate heldout depth/shape scoring.',
                           'Source image/object correspondence is a reviewed declaration, not cryptographically certified.']}
        (out/'candidate.json').write_text(json.dumps(report,indent=2,allow_nan=False,ensure_ascii=False)+'\n',encoding='utf-8')
        return report
    except BaseException:
        shutil.rmtree(out,ignore_errors=True)
        raise


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest='command',required=True)
    ip=sp.add_parser('inventory',help='Inspect full ECHO mesh; never claim a furniture object was found')
    ip.add_argument('--mesh',type=Path,required=True)
    ip.add_argument('--output',type=Path,required=True)
    pp=sp.add_parser('prepare',help='Prepare an explicitly reviewed single furniture object')
    for key in ('scene','object','image','selection','out'):
        pp.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args(argv)
    try:
        if a.command=='inventory':
            if a.output.exists():
                raise IntakeError('Refusing overwrite')
            result=inventory(a.mesh)
            a.output.parent.mkdir(parents=True,exist_ok=True)
            a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
            print(json.dumps({'status':result['decision'],'output':str(a.output)}))
        else:
            result=prepare(a.scene,a.object,a.image,a.selection,a.out)
            print(json.dumps({'status':result['status'],'output':str(a.out)}))
        return 0
    except (IntakeError,ValueError,OSError,RuntimeError,AssertionError,KeyError,TypeError) as e:
        print(json.dumps({'status':'rejected','error':str(e)},ensure_ascii=False),file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
