"""Diagnostic-only V2 desk residual decomposition: never alter failed Mesh or frames."""
from __future__ import annotations
import argparse, hashlib, io, json, tarfile
from pathlib import Path
import numpy as np
from PIL import Image
from echo4d.colmap import _cameras
from echo4d.holdout import read_echo_ply, rasterize_mesh_depth
from echo4d.model import DataError, rigid_matrix

ARCHIVE_SHA="e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c"
ARCHIVE_BYTES=344011403
ROOT="rgbd_dataset_freiburg1_desk"
HOLDOUT=tuple(f"frame_{i:05d}.png" for i in (4,9,14,19,24,29,34,39,44,49,54,59))

def digest(data):
    return hashlib.sha256(data).hexdigest()

def discontinuity_proxy(z,valid):
    out=np.zeros(z.shape,bool)
    for ys,yn,xs,xn in (
        (slice(1,None),slice(None,-1),slice(None),slice(None)),
        (slice(None,-1),slice(1,None),slice(None),slice(None)),
        (slice(None),slice(None),slice(1,None),slice(None,-1)),
        (slice(None),slice(None),slice(None,-1),slice(1,None))):
        out[ys,xs]|=valid[ys,xs]&valid[yn,xn]&(np.abs(z[ys,xs]-z[yn,xn])>=.10)
    return out & valid

def bucketize(pred,z):
    if pred.shape!=z.shape or pred.ndim!=2 or not np.isfinite(pred).all() or not np.isfinite(z).all() or (pred<0).any() or (z<0).any():
        raise DataError("Non-finite, negative, or mismatched depth")
    valid=(z>=.15)&(z<=4)
    overlap=valid&(pred>=.15)&(pred<=4)
    residual=pred.astype(np.float64)-z.astype(np.float64)
    errors=residual[overlap]
    h,w=z.shape
    yy,xx=np.indices(z.shape)
    center=((xx+.5)/w>=.2)&((xx+.5)/w<.8)&((yy+.5)/h>=.2)&((yy+.5)/h<.8)
    discontinuity=discontinuity_proxy(z,valid)
    groups={
        "measured_depth_m":[("0.15-1", (z>=.15)&(z<1)),
            ("1-2",(z>=1)&(z<2)),("2-3",(z>=2)&(z<3)),
            ("3-4",(z>=3)&(z<=4))],
        "image_location":[("center60pct",center),("outer40pct",~center)],
        "depth_discontinuity_proxy":[("valid_4neighbor_delta_ge_10cm",discontinuity),
            ("other",~discontinuity)],
    }
    parts={}
    sq=float(np.square(errors).sum())
    for name,items in groups.items():
        values=[]
        for label,mask in items:
            count=int((valid&mask).sum())
            v=residual[overlap&mask]
            values.append({
                "label":label,"valid_pixels":count,
                "overlap_pixels":int(v.size),
                "missing_pixels":count-int(v.size),
                "abs_error_sum_m":float(np.abs(v).sum()),
                "squared_error_sum_m2":float(np.square(v).sum())})
        if (sum(r["valid_pixels"] for r in values)!=int(valid.sum()) or
            sum(r["overlap_pixels"] for r in values)!=len(errors) or
            not np.isclose(sum(r["squared_error_sum_m2"] for r in values),
                           sq,atol=1e-8,rtol=1e-10)):
            raise DataError("Partition counts/squared loss do not conserve original pixels")
        parts[name]=values
    edges=(0,.03,.05,.1,.3,float("inf"))
    abs_errors=np.abs(errors)
    histogram=[]
    for a,b in zip(edges[:-1],edges[1:]):
        mask=(abs_errors>=a)&(abs_errors<b)
        histogram.append({"range_m":[a,b if np.isfinite(b) else None],
                          "overlap_pixels":int(mask.sum()),
                          "squared_error_sum_m2":float(np.square(errors[mask]).sum())})
    if sum(r["overlap_pixels"] for r in histogram)!=len(errors):
        raise DataError("Residual histogram loses pixels")
    return {
        "valid_pixels":int(valid.sum()),"overlap_pixels":int(overlap.sum()),
        "abs_error_sum_m":float(np.abs(errors).sum()),
        "squared_error_sum_m2":sq,
        "signed_residual_positive":int((errors>0).sum()),
        "signed_residual_negative":int((errors<0).sum()),
        "signed_residual_exact_zero":int((errors==0).sum()),
        "partitions":parts,"abs_error_histogram":histogram,
    }

def from_artifact(d,receipt):
    required={
        "mesh/mesh.ply":receipt["provenance"]["mesh_sha256"],
        "source-cohort-plan.json":receipt["provenance"]["source_plan_sha256"],
        "frozen-protocol.json":receipt["provenance"]["frozen_protocol_sha256"],
        "desk-heldout-metrics.json":receipt["provenance"]["heldout_report_sha256"],
        "report.json":receipt["provenance"]["full_report_sha256"],
    }
    for k,v in required.items():
        if not (d/k).is_file() or digest((d/k).read_bytes())!=v:
            raise DataError("Historic real machine artifact changed: "+k)
    keys=("source-cohort-plan.json","frozen-protocol.json",
          "desk-heldout-metrics.json","metric-trajectory-from-sensor.json")
    j={k:json.loads((d/k).read_text()) for k in keys}
    protocol=j["frozen-protocol.json"]
    score=j["desk-heldout-metrics.json"]
    if (tuple(protocol["holdout_names"])!=HOLDOUT or protocol["downsample"]!=4 or
        protocol["version"]!="desk-metric-mesh-v2-source-paired" or
        protocol["screening_gates"]!={"min_coverage":.8,"max_overlap_rmse_m":.15} or
        score["screening_gate_passed"] is not False or score["rmse_m"]<=.15 or
        len(score["per_frame"])!=12):
        raise DataError("Frozen V2 FAIL / holdout protocol changed")
    return j

def read_source_depths(archive,rows):
    if archive.stat().st_size!=ARCHIVE_BYTES:
        raise DataError("Original official archive byte count changed")
    hasher=hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):
            hasher.update(block)
    if hasher.hexdigest()!=ARCHIVE_SHA:
        raise DataError("Original official archive SHA changed")
    by_path={ROOT+"/"+r["depth_filename"]:r for r in rows}
    if len(by_path)!=12:
        raise DataError("Repeated frozen depth path")
    found={}
    with tarfile.open(archive,"r:gz") as stream:
        for item in stream:
            if item.name not in by_path:
                continue
            if item.name in found or not item.isfile() or not 0<item.size<=8*1024*1024:
                raise DataError("Duplicate or invalid original depth member")
            handle=stream.extractfile(item)
            if handle is None:
                raise DataError("Missing source depth data")
            data=handle.read(item.size+1)
            if len(data)!=item.size or digest(data)!=by_path[item.name]["depth_sha256"]:
                raise DataError("Frozen measured depth source hash mismatch")
            with Image.open(io.BytesIO(data)) as image:
                raw=np.asarray(image)
            if raw.shape!=(480,640) or raw.dtype.kind not in "ui":
                raise DataError("Not a real 640x480 single-channel sensor PNG")
            found[item.name]=(raw.astype(np.float64)/5000)[::4,::4].astype(np.float32)
    if set(found)!=set(by_path):
        raise DataError("At least one original frozen holdout missing")
    return found

def evaluate(artifact_dir,source_archive,evidence_file):
    reference=json.loads(evidence_file.read_text())
    if reference.get("schema")!="echo4d-desk-v2-real-failed-gate-evidence-1":
        raise DataError("Wrong frozen evidence revision")
    items=from_artifact(artifact_dir,reference)
    score=items["desk-heldout-metrics.json"]
    source={x["name"]:x for x in items["source-cohort-plan.json"]["selected_frames"]}
    metric={x["name"]:x for x in items["metric-trajectory-from-sensor.json"]["frames"]}
    depths=read_source_depths(source_archive,score["per_frame"])
    cameras=_cameras(artifact_dir/"cameras.txt")
    if len(cameras)!=1: raise DataError("Invalid original COLMAP camera count")
    camera=next(iter(cameras.values())).scaled(4)
    mesh=read_echo_ply(artifact_dir/"mesh/mesh.ply")
    results=[]
    for expected,name in zip(score["per_frame"],HOLDOUT,strict=True):
        if (expected["name"]!=name or name not in metric or name not in source or
            expected["depth_filename"]!=source[name]["source_depth_path"] or
            abs(expected["rgb_depth_delta_s"])>.03 or
            expected["source_rgb_t_s"]!=source[name]["t_tum_rgb_s"]):
            raise DataError("Original V2 pairing or holdout identity changed")
        pose=rigid_matrix(metric[name]["camera_to_metric_sfm_world"])
        prediction,_=rasterize_mesh_depth(*mesh,camera,pose,far=4.)
        measured=depths[ROOT+"/"+expected["depth_filename"]]
        r=bucketize(prediction,measured)
        old=expected["score"]
        if (r["valid_pixels"]!=old["sensor_valid_pixels"] or
            r["overlap_pixels"]!=old["overlap_pixels"] or
            not np.isclose(r["abs_error_sum_m"]/r["overlap_pixels"],
                old["mae_m"],atol=1e-6,rtol=0) or
            not np.isclose(np.sqrt(r["squared_error_sum_m2"]/r["overlap_pixels"]),
                old["rmse_m"],atol=1e-6,rtol=0)):
            raise DataError("Does not reproduce exact original V2 frame score: "+name)
        results.append({"name":name,"measured_source_hash":expected["depth_sha256"],
                        "rgb_depth_delta_s":expected["rgb_depth_delta_s"],**r})
    n=sum(v["valid_pixels"] for v in results)
    hit=sum(v["overlap_pixels"] for v in results)
    mae=sum(v["abs_error_sum_m"] for v in results)/hit
    rmse=float(np.sqrt(sum(v["squared_error_sum_m2"] for v in results)/hit))
    if (n!=score["valid_measured_pixels"] or hit!=score["overlap_pixels"] or
        not np.isclose(mae,score["mae_m"],atol=1e-6,rtol=0) or
        not np.isclose(rmse,score["rmse_m"],atol=1e-6,rtol=0)):
        raise DataError("Total score disagrees with original machine run")
    aggregates={}
    full_sq=sum(v["squared_error_sum_m2"] for v in results)
    for kind in results[0]["partitions"]:
        aggregates[kind]=[]
        for i in range(len(results[0]["partitions"][kind])):
            records=[v["partitions"][kind][i] for v in results]
            valid=sum(x["valid_pixels"] for x in records)
            overlap=sum(x["overlap_pixels"] for x in records)
            squared=sum(x["squared_error_sum_m2"] for x in records)
            aggregates[kind].append({
                "label":records[0]["label"],"valid_pixels":valid,
                "overlap_pixels":overlap,"missing_pixels":valid-overlap,
                "coverage":overlap/valid if valid else None,
                "rmse_m":float(np.sqrt(squared/overlap)) if overlap else None,
                "global_squared_error_share":squared/full_sq if full_sq else None})
    return {
        "version":"desk-v2-frozen-pixel-residual-attribution-1",
        "status":"ANALYSIS_ONLY_ORIGINAL_V2_STILL_FAILED",
        "original_run":reference["provenance"]["run_url"],
        "original_mesh_sha256":reference["provenance"]["mesh_sha256"],
        "original_archive_sha256":ARCHIVE_SHA,
        "all_12_frozen_views_preserved":True,
        "counts":{"measured":n,"overlap":hit},
        "coverage":hit/n,"mae_m":mae,"rmse_m":rmse,
        "original_gate_m":.15,"original_gate_pass":False,
        "aggregate":aggregates,"per_frame":results,
        "limitations":[
            "No SfM/refit/mesh change; all originally failed measurements kept",
            "Same-sensor Z is not independent scanned-surface geometry truth",
            "Depth edge proxy and screen position only describe association, not causality"],
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("evidence-dir","source-archive","frozen-evidence","output"):
        p.add_argument("--"+arg,type=Path,required=True)
    a=p.parse_args()
    try:
        out=evaluate(a.evidence_dir,a.source_archive,a.frozen_evidence)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        if a.output.exists(): raise DataError("No overwrite of historical diagnostics")
        a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
        print(json.dumps({"status":out["status"],"views":12,"rmse_m":out["rmse_m"]}))
        return 0
    except Exception as e:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        (a.output.parent/"failure.json").write_text(json.dumps({
            "status":"DIAGNOSTIC_NOT_COMPLETED","error":str(e),
            "original_v2_result":"FAIL_UNCHANGED"},indent=2)+"\n")
        print("DIAGNOSTIC_NOT_COMPLETED:",str(e))
        return 2

if __name__=="__main__":
    raise SystemExit(main())
