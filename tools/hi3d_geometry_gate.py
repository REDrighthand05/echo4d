"""CPU-only Hi3D geometry gate for ECHO. Never executes model inference."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

class AuditError(ValueError):
    pass

def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def checked(root, obj):
    if not isinstance(obj, dict) or set(obj) != {"path","sha256"}:
        raise AuditError("Missing path and sha256")
    name = obj["path"]
    if not isinstance(name, str) or not name or "\\" in name:
        raise AuditError("Unsafe path")
    p = Path(name)
    if p.is_absolute() or ".." in p.parts:
        raise AuditError("Absolute or traversal path rejected")
    path = (root/p).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size > 160*1024**2:
        raise AuditError("Asset missing, escaped or oversized")
    if file_hash(path) != obj["sha256"]:
        raise AuditError("Asset sha256 mismatch")
    return path

def mesh(path):
    if path.suffix != ".npz":
        raise AuditError("Expected numeric mesh .npz")
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"vertices","faces"}:
            raise AuditError("Mesh must contain only vertices and faces")
        v, f = np.asarray(z["vertices"]), np.asarray(z["faces"])
    if v.ndim!=2 or v.shape[1]!=3 or v.dtype.kind not in "fiu" or not 3<=len(v)<=2000000 or not np.isfinite(v).all():
        raise AuditError("Invalid mesh vertices")
    if f.ndim!=2 or f.shape[1]!=3 or f.dtype.kind not in "iu" or not 1<=len(f)<=3000000 or f.min()<0 or f.max()>=len(v):
        raise AuditError("Invalid mesh faces")
    return v.astype(np.float64), f.astype(np.int64)

def measured(path):
    if path.suffix != ".npz":
        raise AuditError("Measured point source must be numeric .npz")
    with np.load(path, allow_pickle=False) as z:
        if set(z.files)!={"points_m"}:
            raise AuditError("Invalid measured point field")
        x=np.asarray(z["points_m"])
    if x.ndim!=2 or x.shape[1]!=3 or x.dtype.kind not in "fiu" or not 32<=len(x)<=500000 or not np.isfinite(x).all():
        raise AuditError("Invalid independent measured points")
    return x.astype(np.float64)

def similarity(flat):
    if not isinstance(flat, list) or len(flat)!=16:
        raise AuditError("Missing column-major 4x4 world transform")
    m=np.asarray(flat,dtype=np.float64).reshape(4,4,order="F")
    if not np.isfinite(m).all() or not np.allclose(m[3],[0,0,0,1],atol=1e-9):
        raise AuditError("Invalid world transform")
    a=m[:3,:3]
    scale=float(np.cbrt(np.linalg.det(a)))
    if not (1e-5<scale<1000) or not np.allclose(a.T@a,np.eye(3)*scale*scale,rtol=1e-6,atol=1e-8):
        raise AuditError("Mirroring, shear, nonuniform or missing physical scale")
    if not np.isclose(np.linalg.det(a/scale),1.,atol=1e-6):
        raise AuditError("Not a right-handed rotation")
    return m,scale

def surface_sample(vertices,faces,count,seed=0):
    tris=vertices[faces]
    weights=np.linalg.norm(np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]),axis=1)/2
    if not np.isfinite(weights).all() or weights.sum()<=1e-12:
        raise AuditError("Degenerate surface")
    rng=np.random.default_rng(seed)
    i=rng.choice(len(faces),size=count,p=weights/weights.sum())
    u=rng.random(count);v=rng.random(count);r=np.sqrt(u)
    return tris[i,0]*(1-r)[:,None]+tris[i,1]*(r*(1-v))[:,None]+tris[i,2]*(r*v)[:,None]

def score(shape, reference, count=3000, seed=0, tolerance=.03):
    x=surface_sample(*shape,count,seed)
    a=cKDTree(reference).query(x,workers=1)[0]
    b=cKDTree(x).query(reference,workers=1)[0]
    return {"sampled_surface_points":count,
            "surface_to_measurements_rmse_m":float(np.sqrt(np.mean(a*a))),
            "measurements_to_surface_rmse_m":float(np.sqrt(np.mean(b*b))),
            "measurement_coverage":float(np.mean(b<=tolerance)),
            "surface_precision":float(np.mean(a<=tolerance)),
            "measurements_to_surface_p95_m":float(np.percentile(b,95))}

def audit(manifest,root,count=3000):
    root=Path(root).resolve()
    mf=checked(root,{"path":manifest,"sha256":file_hash((root/manifest).resolve())} )
    spec=json.loads(mf.read_text(encoding="utf-8"))
    if spec.get("version")!="echo-hi3d-audit-1" or spec.get("units")!="meters":
        raise AuditError("Expected explicit ECHO meter format")
    if spec.get("model_kind")!="Hi3D-refined-geometry-only" or not isinstance(spec.get("hi3d_revision"),str) or len(spec["hi3d_revision"])<7:
        raise AuditError("Missing explicit Hi3D provenance")
    if spec.get("alignment_origin")!="coarse_preprocessing_only_not_holdout" or not spec.get("alignment_provenance"):
        raise AuditError("Alignment must be declared as derived before held-out evaluation")
    if not 100<=count<=20000:
        raise AuditError("Unsafe sample count")
    assets={k:checked(root,spec[k]) for k in ("coarse_world_mesh","refined_local_mesh","reference_image","heldout_reference_points")}
    coarse=mesh(assets["coarse_world_mesh"])
    refined=mesh(assets["refined_local_mesh"])
    reference=measured(assets["heldout_reference_points"])
    m,scale=similarity(spec.get("world_from_refined_local"))
    local,faces=refined
    # The caller-supplied transform MUST be frozen before holdout data is read.
    # This routine NEVER fits it to evaluation points or modifies the source mesh.
    transformed=(local@m[:3,:3].T +m[:3,3],faces)
    before=score(coarse,reference,count)
    after=score(transformed,reference,count)
    passed=(after["measurements_to_surface_rmse_m"]<=before["measurements_to_surface_rmse_m"]+.005 and
            after["measurement_coverage"]>=before["measurement_coverage"]-.02)
    return {"version":"echo-hi3d-audit-report-1","status":"cpu_contract_evaluated",
            "hi3d_gpu_was_run":False,"physical_truth_certified":False,
            "candidate_passes_approximate_holdout_check":bool(passed),
            "input_manifest_sha256":file_hash(mf),
            "source_sha256":{k:file_hash(v) for k,v in assets.items()},
            "alignment_provenance_claim_not_cryptographically_verified":spec["alignment_provenance"],
            "claimed_hi3d_revision":spec["hi3d_revision"],
            "scale_m_per_local_unit":scale,"coarse":before,"candidate":after,
            "heldout_samples":len(reference),
            "limitations":["This is a CPU contract test, not Hi3D inference or GLB conversion",
                           "The caller must independently prove alignment and holdout separation",
                           "Mesh surface sampling approximates distances; success is not history certification",
                           "AI-generated unseen geometry must remain a separately labelled optional layer"]}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--manifest",required=True)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    try:
        if a.output.exists():
            raise AuditError("Output already exists; refusing overwrite")
        result=audit(a.manifest,a.root)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
        print(json.dumps({"report":str(a.output),"candidate_pass":result["candidate_passes_approximate_holdout_check"]}))
        return 0
    except (AuditError, OSError, ValueError, KeyError, TypeError) as exc:
        import sys
        print(json.dumps({"status":"rejected","error":str(exc)}),file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
