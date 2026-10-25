"""Strict same-input TUM 48/12 A/B: GT-pose vs estimated-pose sensor-depth TSDF.

A uses supplied TUM pose ONLY after B has fused/scored. GT poses are mapped
into B's measured-metric world with ONE training-view rigid SE(3) transform
(no similarity-scale fit). Both branches fuse EXACT same measured depth bytes,
with identical camera model, bounds, and TSDF configuration. They use the SAME
mesh triangle-depth evaluator for the same twelve withheld sensor Z views.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from echo4d.cli import reconstruct_command
from echo4d.metric_holdout import evaluate_sensor_metric_holdouts
from echo4d.model import DataError, file_sha256, rigid_matrix, load_recording, safe_path
from echo4d.output import output_transaction, write_json
from tum_public_roundtrip import ROOT, PINNED_ARCHIVE_SHA256
from tum_rgb_sfm import gt_after_sfm
from tum_sensor_metric_mesh import experiment as run_frozen_B, DOWNSAMPLE, MAX_RGB_DEPTH_GAP_S, MAX_DEPTH_M

GT_ANCHOR_NAME = "frame_00015.png"  # frozen B training and scale anchor
MIN_COVERAGE = .80
MAX_OVERLAP_RMSE_M = .15


def align_gt_to_metric_estimated_world(gt_poses: dict, metric_frames: list[dict],
                                      training: list[str], withheld: list[str],
                                      anchor: str=GT_ANCHOR_NAME
                                      ) -> tuple[dict[str,list[float]], list[float]]:
    """Use a single training-view 6-DoF reference for world gauge, no Sim(3) scale."""
    if len(training) != 48 or len(withheld) != 12 or set(training)&set(withheld):
        raise DataError("A/B must preserve frozen 48 train and 12 disjoint holdout views")
    if anchor not in training or anchor in withheld:
        raise DataError("Rigid world-gauge anchor must be a frozen training view")
    metric = {frame["name"]:frame for frame in metric_frames}
    if len(metric) != len(metric_frames):
        raise DataError("Duplicate metric pose names")
    needed = set(training)|set(withheld)
    if not needed.issubset(gt_poses) or not needed.issubset(metric):
        raise DataError("At least one A/B frozen view lacks GT or measured-scale SfM pose")
    origin_est=rigid_matrix(metric[anchor]["camera_to_metric_sfm_world"])
    origin_gt=rigid_matrix(gt_poses[anchor])
    gauge = origin_est @ np.linalg.inv(origin_gt)
    rigid_matrix(gauge.flatten(order="F"))
    result={}
    for name in needed:
        pose=gauge @ rigid_matrix(gt_poses[name])
        rigid_matrix(pose.flatten(order="F"))
        result[name]=pose.flatten(order="F").tolist()
    if not np.allclose(rigid_matrix(result[anchor]), origin_est, atol=1e-6):
        raise DataError("Single-view rigid gauge did not align A/B worlds")
    return result, gauge.flatten(order="F").tolist()


def make_A_recording(B_dir:Path, A_dir:Path, split:dict, metric:dict,
                     gt_mapped:dict[str,list[float]]) -> dict:
    """Copy *the same bytes* of training RGB and Z, changing only cameraToWorld."""
    original=json.loads((B_dir/"recording.json").read_text(encoding="utf-8"))
    source_frames=original["frames"]
    train=[r["name"] for r in split["train"]]
    metric_map={x["name"]:x for x in metric["frames"]}
    if len(source_frames)!=48 or len(train)!=48 or len(set(train))!=48:
        raise DataError("B training recording differs from frozen split")
    changes=[]
    with output_transaction(A_dir) as stage:
        (stage/"depth").mkdir()
        (stage/"color").mkdir()
        for i,(frame,name) in enumerate(zip(source_frames,train)):
            if name not in gt_mapped or name not in metric_map:
                raise DataError("Missing frozen training pose")
            b_pose=rigid_matrix(frame["cameraToWorld"])
            bound=rigid_matrix(metric_map[name]["camera_to_metric_sfm_world"])
            if not np.allclose(b_pose,bound,atol=1e-6,rtol=1e-7):
                raise DataError(f"Input recording train order/pose mismatch: {name}")
            receipt={"name":name}
            for key in ("color","depth"):
                inp=safe_path(B_dir,frame[key])
                dest=stage/frame[key]
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(inp,dest)
                digest=file_sha256(inp)
                if file_sha256(dest)!=digest:
                    raise DataError("A/B measured-depth or RGB byte equality failed")
                receipt[key+"_sha256"]=digest
            frame["cameraToWorld"]=gt_mapped[name]
            changes.append(receipt)
        original["name"]="TUM-GT-pose-same-48-training-measured-Z"
        original["notes"]="A: true TUM GT camera poses rigidly aligned to B metric world; same Z/RGB byte hashes, same camera and TSDF. NOT product estimated pose."
        write_json(stage/"recording.json",original)
        write_json(stage/"a-source-evidence.json",{
            "version":"echo-fair-ab-1",
            "gt_pose_is_A_only":True,
            "rigid_world_gauge":"one fixed training frame, no GT scale optimization",
            "source_B_recording_sha256":file_sha256(B_dir/"recording.json"),
            "training_pixel_inputs_identical":True,
            "frame_inputs":changes
        })
        record=load_recording(stage/"recording.json")
        for _ in record.iter_frames():pass
    return {"A_recording_sha256":file_sha256(A_dir/"recording.json"),
            "B_recording_sha256":file_sha256(B_dir/"recording.json"),
            "A_input_evidence_sha256":file_sha256(A_dir/"a-source-evidence.json"),
            "training_frames":len(train),"identical_sensor_depth_and_color_bytes":True}


def score_gate(result:dict)->dict:
    return {"coverage_minimum":MIN_COVERAGE,
            "overlap_rmse_max_m":MAX_OVERLAP_RMSE_M,
            "passes":bool(result["coverage"]>=MIN_COVERAGE
                          and result["rmse_m"]<=MAX_OVERLAP_RMSE_M)}


def run_experiment(args:argparse.Namespace, receipt:dict)->None:
    report=args.report_dir.resolve()
    report.mkdir(parents=True,exist_ok=True)
    b={}
    b["git_sha"]=receipt["git_sha"]
    run_frozen_B(argparse.Namespace(work=args.work,report_dir=report),b)
    receipt["B_mesh_sha256"]=b["mesh_result"]["mesh_sha256"]
    receipt["B_split_sha256"]=b["split_sha256"]
    receipt["source_dataset_sha256"]=b["metric_stage"]["source_sha256"]
    if b["metric_stage"]["source_sha256"]!=PINNED_ARCHIVE_SHA256:
        raise DataError("Pinned TUM source hash has changed")

    # B upstream #14 reads GT only AFTER the sensor metric scale is locked,
    # for a separate pose-audit; it does not feed B pose/TSDF. Here the GT
    # camera poses are applied to A ONLY after B Mesh and scores are sealed.
    receipt["B_mesh_and_score_finalized_before_GT_poses_used_for_A"]=True
    receipt["upstream_B_GT_was_pose_diagnostic_only_after_scale_lock"]=True
    work=args.work.resolve()
    source_json=json.loads((report/"metric"/"report.json").read_text(encoding="utf-8"))
    source=source_json["test_video"]["source_rgb_manifest"]
    gt, missing=gt_after_sfm(work/"unpacked"/ROOT,source)
    split=json.loads((report/"b-48train-12holdout.json").read_text(encoding="utf-8"))
    metric=json.loads((report/"metric"/"metric-trajectory-from-sensor.json").read_text(encoding="utf-8"))
    train=[x["name"] for x in split["train"]]
    withheld=split["holdout"]
    gt_mapped,gauge=align_gt_to_metric_estimated_world(gt,metric["frames"],train,withheld)
    receipt["gt_groundtruth_txt_sha256"]=file_sha256(work/"unpacked"/ROOT/"groundtruth.txt")
    receipt["gauge_single_training_frame"]=GT_ANCHOR_NAME
    receipt["A_only_gt_rigid_se3_in_B_world"]=gauge
    receipt["A_GT_missing_source_frames"]=missing

    B_dir=work/"b-sensor-recording"
    A_dir=work/"a-known-pose-recording"
    input_provenance=make_A_recording(B_dir,A_dir,split,metric,gt_mapped)
    receipt["ab_identical_training_input_provenance"]=input_provenance
    a_mesh=report/"a-mesh"
    tsdf=b["tsdf_config"]
    a_result=reconstruct_command(argparse.Namespace(
        recording=A_dir/"recording.json",output=a_mesh,
        bounds=b["pre_registered_bounds"],voxel=tsdf["voxel_m"],
        truncation=None,max_depth=tsdf["max_depth_m"],
        max_voxels=tsdf["max_voxels"],min_observations=tsdf["min_observations"],
        stride=1,limit=None,reference=None,threshold=.05,
        no_scene=False,save_volume=False,quiet=True))
    if a_result["frames"]!=48 or a_result["vertices"]<1 or a_result["triangles"]<1:
        raise DataError("A known-pose mesh is empty or changed frozen frame count")
    receipt["A_mesh_sha256"]=a_result["mesh_sha256"]

    # Score both with the exact SAME function, camera, thresholds and TUM depth bytes.
    sfm=work/"rgb-only-colmap"
    info=json.loads((sfm/"report.json").read_text(encoding="utf-8"))
    model=sfm/info["result"]["sfm_model_dir"]
    data=work/"unpacked"/ROOT
    holdout_args=(B_dir/"recording.json",metric,split,source,data,model)
    b_score=evaluate_sensor_metric_holdouts(report/"mesh"/"mesh.ply",*holdout_args,
        downsample=DOWNSAMPLE,max_gap_s=MAX_RGB_DEPTH_GAP_S,max_depth_m=MAX_DEPTH_M)
    a_score=evaluate_sensor_metric_holdouts(a_mesh/"mesh.ply",
        A_dir/"recording.json",metric,split,source,data,model,
        downsample=DOWNSAMPLE,max_gap_s=MAX_RGB_DEPTH_GAP_S,max_depth_m=MAX_DEPTH_M,
        evaluation_poses={name:gt_mapped[name] for name in withheld})

    if b_score["gt_camera_poses_loaded"] or not a_score["gt_camera_poses_loaded"]:
        raise DataError("A/B pose source label is inconsistent")
    if b_score["holdout_frames"]!=12 or a_score["holdout_frames"]!=12:
        raise DataError("Some frozen depth views were silently removed")
    if b_score["valid_measured_pixels"]!=a_score["valid_measured_pixels"]:
        raise DataError("A and B do not share identical withheld sensor observation counts")
    for ar,br in zip(a_score["per_frame"],b_score["per_frame"]):
        if (ar["name"]!=br["name"] or ar["depth_sha256"]!=br["depth_sha256"]
                or ar["score"]["sensor_valid_pixels"]!=br["score"]["sensor_valid_pixels"]):
            raise DataError("A/B heldout per-frame depth source changed")
    previous=b.get("mesh_heldout",{})
    if (abs(b_score["coverage"]-previous.get("coverage",-100))>1e-9
            or abs(b_score["rmse_m"]-previous.get("rmse_m",-100))>1e-9):
        raise DataError("Repeated B primary scoring disagrees with frozen B run")
    delta={key:float(b_score[key]-a_score[key])
           for key in ("coverage","mae_m","rmse_m","p95_abs_m","within_3cm","within_5cm","within_10cm","within_30cm")}
    comparison={
        "version":"echo-fair-estimated-vs-known-pose-48-12-1",
        "A":"TUM GT poses rigidly mapped to B-world at ONE training anchor; real measured RGB-D",
        "B":"RGB COLMAP estimated poses + 3 independent measured depth scale anchors; same real sensor RGB-D",
        "dataset_sha256":b["metric_stage"]["source_sha256"],
        "train_holdout_split_sha256":b["split_sha256"],
        "A_and_B_same_rgb_and_depth_bytes":True,
        "A_and_B_same_camera_tsdf_bounds_voxel":True,
        "world_gauge_gt_anchor":GT_ANCHOR_NAME,
        "world_gauge_rigid_no_rescaling":True,
        "holdout_not_fused":True,
        "A_mesh_sha256":a_result["mesh_sha256"],
        "B_mesh_sha256":b["mesh_result"]["mesh_sha256"],
        "A":a_score,"B":b_score,
        "B_minus_A":delta,
        "A_gate":score_gate(a_score),"B_gate":score_gate(b_score),
        "limitations":[
          "Both share same public RGB-D device and scene; no independent laser surface geometry.",
          "Only B uses learned/estimated SfM poses; A uses GT as explicit oracle baseline.",
          "One SE(3) training-anchor alignment fixes arbitrary world axes, NEVER the scale.",
          "The difference between A/B is joint pose+fusion+render impact, not pure pose-only error attribution.",
          "Overlap RMSE conditional on coverage; do not trade unobserved pixels for low RMSE.",
          "Source RGB data were repackaged into a fixed-frame test MP4; not original smartphone video."
        ]}
    write_json(report/"fair-ab-comparison.json",comparison)
    receipt["A_summary"]={k:a_score[k] for k in ("coverage","mae_m","rmse_m","p95_abs_m","holdout_frames")}
    receipt["B_summary"]={k:b_score[k] for k in ("coverage","mae_m","rmse_m","p95_abs_m","holdout_frames")}
    receipt["A_gate"]=comparison["A_gate"]
    receipt["B_gate"]=comparison["B_gate"]
    receipt["fair_ab_comparison_sha256"]=file_sha256(report/"fair-ab-comparison.json")
    if not (comparison["A_gate"]["passes"] and comparison["B_gate"]["passes"]):
        raise DataError("Frozen first-pass A/B coverage-or-RMSE research gate failed; see full comparison JSON")


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    args=p.parse_args()
    start=time.perf_counter()
    receipt={
        "experiment":"real-TUM-same-48train-12holdout-GT-pose-A-vs-sensor-metric-SfM-B-v1",
        "run_url": (f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/{os.getenv('GITHUB_RUN_ID')}"
                    if os.getenv("GITHUB_RUN_ID") else None),
        "git_sha":os.getenv("ECHO_HEAD_SHA") or os.getenv("GITHUB_SHA"),
        "started_utc":datetime.now(timezone.utc).isoformat(),
        "status":"running",
        "E3_scope":"real TUM RGB-D same 48/12; no independent scanned surface"
    }
    code=0
    try:
        run_experiment(args,receipt)
        receipt["status"]="passed_same_input_A_B_depth_view_consistency_not_independent_geometry"
    except Exception as exc:
        code=1
        receipt["status"]="failed"
        receipt["error"]={"type":type(exc).__name__,"message":str(exc)}
        print("FAIR_AB_FAILURE:",type(exc).__name__,str(exc),file=sys.stderr)
    finally:
        args.report_dir.mkdir(parents=True,exist_ok=True)
        receipt["elapsed_seconds"]=round(time.perf_counter()-start,3)
        write_json(args.report_dir/"fair-ab-run.json",receipt)
        lines=["# Frozen real TUM A/B GT-pose vs estimated-pose Mesh", "",
               f"- Status: {receipt['status']}",
               f"- [Actual run]({receipt.get('run_url')})",
               f"- GT(A) mesh sha256: {receipt.get('A_mesh_sha256','none')}",
               f"- Estimated(B) mesh sha256: {receipt.get('B_mesh_sha256','none')}",
               f"- Same sensor input files: {receipt.get('ab_identical_training_input_provenance',{}).get('identical_sensor_depth_and_color_bytes',False)}",
               f"- A coverage / RMSE: {receipt.get('A_summary',{}).get('coverage','not done')} / {receipt.get('A_summary',{}).get('rmse_m','not done')}",
               f"- B coverage / RMSE: {receipt.get('B_summary',{}).get('coverage','not done')} / {receipt.get('B_summary',{}).get('rmse_m','not done')}",
               f"- Failed gate or error: {receipt.get('error',{}).get('message','none')}","",
               "A uses real TUM GT poses; B uses COLMAP+external sensor scale. Same measured Z and scorer, NOT independent scanned-surface GT."]
        (args.report_dir/"FAIR_AB_SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return code


if __name__=="__main__":
    raise SystemExit(main())
