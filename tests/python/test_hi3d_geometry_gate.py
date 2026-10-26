"""Known-answer tests for the CPU-only Hi3D asset gate; no GPU inference."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools"))
from hi3d_geometry_gate import audit, similarity, AuditError

def fixture(root):
    # One known 1-metre wall (coarse and refined are the same physical plane).
    x,y=np.meshgrid(np.linspace(-.6,.6,4),np.linspace(-.6,.6,4))
    vertices=np.column_stack([x.ravel(),y.ravel(),np.ones(16)])
    faces=[]
    for r in range(3):
        for c in range(3):
            i=r*4+c
            faces.extend([[i,i+1,i+4],[i+1,i+5,i+4]])
    faces=np.array(faces)
    m=np.eye(4);m[:3,:3]*=2;m[:3,3]=[1,2,3]
    local=(vertices-m[:3,3])/2
    u,v=np.meshgrid(np.linspace(-.59,.59,32),np.linspace(-.59,.59,32))
    heldout=np.column_stack([u.ravel(),v.ravel(),np.ones(1024)])
    np.savez(root/"coarse.npz",vertices=vertices,faces=faces)
    np.savez(root/"refined.npz",vertices=local,faces=faces)
    np.savez(root/"heldout.npz",points_m=heldout)
    (root/"photo.png").write_bytes(b"synthetic photo placeholder, not a real photo")
    def asset(name):
        return {"path":name,"sha256":hashlib.sha256((root/name).read_bytes()).hexdigest()}
    spec={"version":"echo-hi3d-audit-1","units":"meters",
          "model_kind":"Hi3D-refined-geometry-only","hi3d_revision":"705fc0901c5d",
          "alignment_origin":"coarse_preprocessing_only_not_holdout",
          "alignment_provenance":"synthetic known transform frozen before holdout",
          "world_from_refined_local":m.flatten(order="F").tolist(),
          "coarse_world_mesh":asset("coarse.npz"),
          "refined_local_mesh":asset("refined.npz"),
          "reference_image":asset("photo.png"),
          "heldout_reference_points":asset("heldout.npz")}
    (root/"manifest.json").write_text(json.dumps(spec))
    return spec

def save(root,spec):
    (root/"manifest.json").write_text(json.dumps(spec))

def test_known_transform_keeps_geometry(tmp_path):
    fixture(tmp_path)
    result=audit("manifest.json",tmp_path,1000)
    assert result["candidate_passes_approximate_holdout_check"]
    assert result["scale_m_per_local_unit"]==pytest.approx(2.)
    assert result["coarse"]["measurement_coverage"]==result["candidate"]["measurement_coverage"]
    assert result["coarse"]["measurements_to_surface_rmse_m"]==pytest.approx(
        result["candidate"]["measurements_to_surface_rmse_m"],abs=1e-12)
    assert not result["hi3d_gpu_was_run"]
    assert not result["physical_truth_certified"]

def test_large_refinement_error_rejected(tmp_path):
    spec=fixture(tmp_path)
    with np.load(tmp_path/"refined.npz") as f:
        vertices=f["vertices"].copy();faces=f["faces"].copy()
    vertices[:,2]+=.15
    np.savez(tmp_path/"refined.npz",vertices=vertices,faces=faces)
    spec["refined_local_mesh"]["sha256"]=hashlib.sha256((tmp_path/"refined.npz").read_bytes()).hexdigest()
    save(tmp_path,spec)
    result=audit("manifest.json",tmp_path,1000)
    assert not result["candidate_passes_approximate_holdout_check"]

def test_hash_tampering_fails(tmp_path):
    fixture(tmp_path)
    (tmp_path/"refined.npz").write_bytes(b"tampered")
    with pytest.raises(AuditError,match="sha256 mismatch"):
        audit("manifest.json",tmp_path)

def test_provenance_fitted_to_holdout_rejected(tmp_path):
    spec=fixture(tmp_path);spec["alignment_origin"]="optimised_on_heldout"
    save(tmp_path,spec)
    with pytest.raises(AuditError,match="Alignment"):
        audit("manifest.json",tmp_path)

def test_missing_metric_transform_rejected(tmp_path):
    spec=fixture(tmp_path);del spec["world_from_refined_local"]
    save(tmp_path,spec)
    with pytest.raises(AuditError,match="world transform"):
        audit("manifest.json",tmp_path)

@pytest.mark.parametrize("bad",[
    np.diag([-1.,1.,1.,1.]),
    np.diag([1.,2.,1.,1.]),
    np.diag([0.,1.,1.,1.]),
    np.array([[1.,.2,0,0],[0,1.,0,0],[0,0,1,0],[0,0,0,1.]])
])
def test_mirror_nonuniform_shear_and_zero_scale_rejected(bad):
    with pytest.raises(AuditError):
        similarity(bad.flatten(order="F").tolist())

def test_nan_transform_rejected(tmp_path):
    spec=fixture(tmp_path)
    spec["world_from_refined_local"][0]=float("nan")
    save(tmp_path,spec)
    with pytest.raises(AuditError):
        audit("manifest.json",tmp_path)

def test_asset_path_escape_rejected(tmp_path):
    spec=fixture(tmp_path)
    spec["coarse_world_mesh"]["path"]="../escape.npz"
    save(tmp_path,spec)
    with pytest.raises(AuditError):
        audit("manifest.json",tmp_path)

def test_wrong_asset_key_rejected(tmp_path):
    spec=fixture(tmp_path)
    with np.load(tmp_path/"refined.npz") as f:
        v=f["vertices"];faces=f["faces"]
    np.savez(tmp_path/"refined.npz",vertices=v,faces=faces,unexpected=np.arange(3))
    spec["refined_local_mesh"]["sha256"]=hashlib.sha256((tmp_path/"refined.npz").read_bytes()).hexdigest()
    save(tmp_path,spec)
    with pytest.raises(AuditError,match="Mesh must contain"):
        audit("manifest.json",tmp_path)

def test_too_few_reference_measurements_rejected(tmp_path):
    spec=fixture(tmp_path)
    np.savez(tmp_path/"heldout.npz",points_m=np.ones((3,3)))
    spec["heldout_reference_points"]["sha256"]=hashlib.sha256((tmp_path/"heldout.npz").read_bytes()).hexdigest()
    save(tmp_path,spec)
    with pytest.raises(AuditError,match="Invalid independent measured"):
        audit("manifest.json",tmp_path)

def test_units_not_claimed_without_meters(tmp_path):
    spec=fixture(tmp_path);spec["units"]="unknown";save(tmp_path,spec)
    with pytest.raises(AuditError,match="meter format"):
        audit("manifest.json",tmp_path)


def test_manifest_path_escape_rejected(tmp_path):
    fixture(tmp_path)
    with pytest.raises(AuditError,match="Unsafe manifest"):
        audit("../manifest.json",tmp_path)
