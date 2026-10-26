"""CPU real-triangle-file checks, not Hi3D inference or independent scene GT."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest
# Optional mesh IO is installed in the dedicated intake CI, not ECHO core.
# The base platform matrix must still collect all other ECHO tests.
trimesh = pytest.importorskip("trimesh", reason="Dedicated Hi3D intake workflow installs optional trimesh")
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from hi3d_candidate_intake import inventory, prepare, IntakeError, digest


def make_inputs(tmp_path):
    scene=tmp_path/'scene.ply'
    obj=tmp_path/'chair.ply'
    image=tmp_path/'chair.png'
    scene_mesh=trimesh.util.concatenate([
        trimesh.creation.box(extents=[2,1,.08]),
        trimesh.creation.box(extents=[.45,.4,.65])])
    scene_mesh.export(scene)
    m=trimesh.creation.box(extents=[.45,.4,.65]);m.apply_translation([.2,1.3,.5]);m.export(obj)
    Image.new('RGB',(64,64),(244,240,239)).save(image)
    record=tmp_path/'selection.json'
    evidence={'version':'echo-hi3d-selection-1','source_scene_sha256':digest(scene),
              'object_mesh_sha256':digest(obj),'reference_image_sha256':digest(image),
              'selection_provenance':'Manual object isolation in this synthetic test only',
              'scale_evidence':'physically_measured_m',
              'reviewed_as_single_object':True,'reference_image_same_object':True}
    record.write_text(json.dumps(evidence))
    return scene,obj,image,record


def test_inventory_reports_unknown_objects(tmp_path):
    scene,*_=make_inputs(tmp_path)
    r=inventory(scene)
    assert r['vertices']>=8 and r['triangles']>=12
    assert not r['is_a_verified_standalone_furniture_object']
    assert r['decision']=='requires_furniture_mask_and_matching_real_image'


def test_prepare_roundtrip_coarse_gauge(tmp_path):
    scene,obj,img,record=make_inputs(tmp_path)
    r=prepare(scene,obj,img,record,tmp_path/'out')
    normal=trimesh.load_mesh(tmp_path/'out'/'coarse-normalized.glb',force='mesh',process=False)
    original=trimesh.load_mesh(obj,force='mesh',process=False)
    world=np.asarray(r['coarse_unit_to_echo_world_column_major']).reshape((4,4),order='F')
    got=np.asarray(normal.vertices)@world[:3,:3].T+world[:3,3]
    # GLB uses float32 positions, so reloads with an ~1e-7 m error.
    assert np.max(np.linalg.norm(got-original.vertices,axis=1))<1e-6
    assert r['hi3d_was_run'] is False
    assert r['independent_holdout_was_evaluated'] is False
    assert r['status']=='prepared_for_optional_gpu_refinement_not_verified'


def test_same_source_scene_forbidden(tmp_path):
    scene,obj,img,record=make_inputs(tmp_path)
    with pytest.raises(IntakeError,match='Whole ECHO scene'):
        prepare(scene,scene,img,record,tmp_path/'out')


def test_unreviewed_object_refused(tmp_path):
    scene,obj,img,record=make_inputs(tmp_path)
    d=json.loads(record.read_text());d['reviewed_as_single_object']=False;record.write_text(json.dumps(d))
    with pytest.raises(IntakeError,match='explicitly reviewed'):
        prepare(scene,obj,img,record,tmp_path/'out')


def test_unmeasured_scale_refused(tmp_path):
    scene,obj,img,record=make_inputs(tmp_path)
    d=json.loads(record.read_text());d['scale_evidence']='inferred_from_rgb_only';record.write_text(json.dumps(d))
    with pytest.raises(IntakeError,match='independent physical scale'):
        prepare(scene,obj,img,record,tmp_path/'out')


def test_hash_mismatch_refused(tmp_path):
    scene,obj,img,record=make_inputs(tmp_path)
    Image.new('RGB',(64,64),(1,2,3)).save(img)
    with pytest.raises(IntakeError,match='hash mismatch'):
        prepare(scene,obj,img,record,tmp_path/'out')


def test_wrong_object_to_image_declared_refused(tmp_path):
    scene,obj,img,record=make_inputs(tmp_path)
    d=json.loads(record.read_text());d['reference_image_same_object']=False;record.write_text(json.dumps(d))
    with pytest.raises(IntakeError,match='same isolated object'):
        prepare(scene,obj,img,record,tmp_path/'out')


def test_extra_provenance_field_rejected(tmp_path):
    scene,obj,img,record=make_inputs(tmp_path)
    d=json.loads(record.read_text());d['unknown']=True;record.write_text(json.dumps(d))
    with pytest.raises(IntakeError,match='proof fields'):
        prepare(scene,obj,img,record,tmp_path/'out')


def test_reject_empty_selection_evidence(tmp_path):
    scene,obj,img,record=make_inputs(tmp_path)
    d=json.loads(record.read_text());d['selection_provenance']='manual';record.write_text(json.dumps(d))
    with pytest.raises(IntakeError,match='auditable selection'):
        prepare(scene,obj,img,record,tmp_path/'out')


def test_existing_output_not_overwritten(tmp_path):
    scene,obj,img,record=make_inputs(tmp_path)
    out=tmp_path/'out';out.mkdir();(out/'keep').write_text('KEEP')
    with pytest.raises(IntakeError,match='overwrite'):
        prepare(scene,obj,img,record,out)
    assert (out/'keep').read_text()=='KEEP'


def test_invalid_mesh_rejected(tmp_path):
    scene,obj,img,record=make_inputs(tmp_path)
    scene.write_bytes(b'NOT A PLY')
    with pytest.raises(IntakeError):
        inventory(scene)


def test_nonexistent_mesh_rejected(tmp_path):
    with pytest.raises((IntakeError,OSError)):
        inventory(tmp_path/'missing.ply')


def test_empty_glb_not_accepted(tmp_path):
    scene,obj,img,record=make_inputs(tmp_path)
    obj.write_bytes(b'')
    with pytest.raises(IntakeError):
        prepare(scene,obj,img,record,tmp_path/'out')
