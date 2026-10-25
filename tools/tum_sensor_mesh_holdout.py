"""B-path real 12-view withheld-depth scoring, using estimated metric SfM poses.

First replays the frozen B-48/12 mesh experiment. After mesh creation, opens
the twelve withheld TUM depth frames ONLY for evaluation. Never substitutes
TUM GT camera poses, recalibrates scale, or changes mesh geometry from scores.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

from echo4d.colmap import _cameras
from echo4d.holdout import read_echo_ply, rasterize_mesh_depth, compare_holdout_depth
from echo4d.metric_anchor import read_measured_depth
from echo4d.model import DataError, file_sha256, load_recording, rigid_matrix, safe_path
from echo4d.tum import nearest_timestamp, read_rows
from tum_public_roundtrip import ROOT
from tum_sensor_metric_mesh import experiment as run_B_mesh, DOWNSAMPLE, split_names

# Prescribed in Issue #15 before this evaluator's first real-data run.
MIN_COVERAGE = 0.80
MAX_RMSE_M = 0.15
MAX_TIMESTAMP_GAP_S = 0.03
DEPTH_UNITS_PER_M = 5000.0
MAX_DEPTH_M = 4.0
ERROR_THRESHOLDS_M = (0.03, 0.05, 0.10, 0.30)


def validate_frozen_holdout_contract(split: dict, metric: dict,
                                     source: list[dict]) -> list[str]:
    """Reject mismatched 48/12 subsets and an unmeasured or leaked scale.

    Uses the parent experiment's frozen exact frame list, never the result's
    registration/accuracy statistics to pick favorable evaluation images.
    """
    train, withheld = split_names(source)
    rows = split.get("train")
    if not isinstance(rows, list) or len(rows) != 48:
        raise DataError("Frozen training split must have 48 frame-depth entries")
    if [x.get("name") for x in rows] != train or split.get("holdout") != withheld:
        raise DataError("Training/holdout frame names differ from frozen Issue #15 split")
    paths = [x.get("depth") for x in rows]
    if any(not isinstance(p, str) or not p for p in paths) or len(set(paths)) != 48:
        raise DataError("Training depth names must be present and unique")
    if (split.get("source_kind") != "measured_registered_z_depth" or
            split.get("depth_numeric_units_per_meter") != DEPTH_UNITS_PER_M):
        raise DataError("Incorrect measured camera-Z depth source or units")
    if (metric.get("version") != "echo-sfm-metric-scale-1" or
            metric.get("units") != "meters" or
            metric.get("metric_scale_estimated_from_measured_depth") is not True or
            metric.get("scale_is_from_external_gt_poses") is not False):
        raise DataError("Metric trajectory must be sensor-scaled without GT scale")
    scale = metric.get("scale_m_per_sfm_unit")
    if isinstance(scale, bool) or not isinstance(scale, (float, int)) or not math.isfinite(scale) or scale <= 0:
        raise DataError("Missing physically measured positive metric scale")
    anchors = metric.get("anchor_diagnostics")
    if (not isinstance(anchors, list) or len(anchors) != 3 or
            any(not isinstance(x, dict) or not isinstance(x.get("name"), str) for x in anchors) or
            {x["name"] for x in anchors} & set(withheld)):
        raise DataError("Metric scale anchors missing, changed, or overlap holdout")
    if len({x["name"] for x in source}) != len(source):
        raise DataError("Duplicate source RGB image names")
    frames = metric.get("frames")
    if not isinstance(frames, list) or len({x.get("name") for x in frames}) != len(frames):
        raise DataError("Invalid or duplicate metric pose entries")
    if not set(withheld).issubset({x["name"] for x in frames}):
        raise DataError("Withheld views missing estimated metric camera poses")
    return withheld


def aggregate_scores(error_arrays: list[np.ndarray], valid: int,
                     per_frame: list[dict]) -> dict:
    if not per_frame or valid <= 0:
        raise DataError("No withheld frame evidence")
    if any(row["status"] != "measured" for row in per_frame):
        raise DataError("At least one withheld frame has no overlapping mesh surface")
    if not error_arrays:
        raise DataError("No valid withheld depth overlap")
    signed = np.concatenate(error_arrays)
    count = len(signed)
    if count != sum(frame["overlap_pixels"] for frame in per_frame):
        raise DataError("Withheld coverage accounting inconsistency")
    absolute = np.abs(signed)
    result = {
        "heldout_frames": len(per_frame),
        "sensor_valid_pixels": valid,
        "overlap_pixels": count,
        "missing_surface_pixels": valid-count,
        "observed_depth_coverage": count/valid,
        "mae_m": float(np.mean(absolute)),
        "rmse_m": float(np.sqrt(np.mean(signed*signed))),
        "bias_m": float(np.mean(signed)),
        "median_abs_m": float(np.median(absolute)),
        "p95_abs_m": float(np.percentile(absolute, 95)),
        "threshold_accuracy_on_overlap": {
            f"within_{int(round(k*100))}cm": float(np.mean(absolute <= k))
            for k in ERROR_THRESHOLDS_M
        },
        "per_frame": per_frame,
        "gate": {"coverage_minimum": MIN_COVERAGE,
                 "rmse_maximum_m": MAX_RMSE_M,
                 "passed": count/valid >= MIN_COVERAGE
                           and float(np.sqrt(np.mean(signed*signed))) <= MAX_RMSE_M},
        "interpretation": "Same-sensor withheld depth view-consistency; NOT independent true scene geometry."
    }
    return result


def evaluate_locked_B_run(work: Path, report_dir: Path, receipt: dict) -> dict:
    mesh_path = report_dir / "mesh" / "mesh.ply"
    split_file = report_dir / "b-48train-12holdout.json"
    metric_file = report_dir / "metric" / "metric-trajectory-from-sensor.json"
    metric_receipt = report_dir / "metric" / "report.json"
    if not all(x.is_file() for x in (mesh_path, split_file, metric_file, metric_receipt)):
        raise DataError("B-path prerequisite files missing; no independent holdout scoring")
    split = json.loads(split_file.read_text(encoding="utf-8"))
    metric = json.loads(metric_file.read_text(encoding="utf-8"))
    first = json.loads(metric_receipt.read_text(encoding="utf-8"))
    source = first["test_video"]["source_rgb_manifest"]
    withheld = validate_frozen_holdout_contract(split, metric, source)
    names = {s["name"]: s for s in source}
    estimated = {s["name"]: s for s in metric["frames"]}
    if any(name not in names or name not in estimated for name in withheld):
        raise DataError("Missing source timestamp or estimated metric camera for withheld frame")

    tum = work / "unpacked" / ROOT
    depth_index = tum / "depth.txt"
    rows = read_rows(depth_index, 2)
    ts = np.asarray([float(v[0]) for v in rows])
    if not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
        raise DataError("Measured depth index timestamps must be finite and increasing")
    # Compare canonical paths, not only text: aliases must not let heldout
    # depths reuse measurements seen by the TSDF training stage.
    training_depth_paths = {
        safe_path(tum, entry["depth"]).resolve() for entry in split["train"]
    }
    sfm_report = json.loads((work / "rgb-only-colmap" / "report.json").read_text(encoding="utf-8"))
    cameras = _cameras(work / "rgb-only-colmap" / sfm_report["result"]["sfm_model_dir"] / "cameras.txt")
    selected_ids = {estimated[name]["camera_id"] for name in withheld}
    if len(selected_ids) != 1 or not selected_ids.issubset(cameras):
        raise DataError("Withheld frames do not share known SfM camera intrinsics")
    camera = cameras[next(iter(selected_ids))].scaled(DOWNSAMPLE)
    training_camera = load_recording(work / "b-sensor-recording" / "recording.json").camera
    if any(not np.isclose(getattr(camera, attr), getattr(training_camera, attr),
                          rtol=0, atol=1e-8)
           for attr in ("width", "height", "fx", "fy", "cx", "cy")):
        raise DataError("Holdout intrinsics differ from frozen B-path TSDF camera")
    vertices, faces = read_echo_ply(mesh_path)
    if len(vertices) < 3 or len(faces) < 1:
        raise DataError("B-path reconstructed surface is empty")
    all_residuals, per_frame, total_valid = [], [], 0
    read_depth_files = set()
    # Withheld camera poses are from estimated SfM, NEVER from TUM groundtruth.txt.
    for name in withheld:
        measured_rgb_time = float(names[name]["t_tum_rgb_s"])
        k = nearest_timestamp(ts, measured_rgb_time, MAX_TIMESTAMP_GAP_S)
        if k is None:
            raise DataError(f"Missing synchronized withheld TUM sensor view: {name}")
        gap = float(ts[k] - measured_rgb_time)
        rel_path = rows[k][1]
        if rel_path in read_depth_files:
            raise DataError("Repeated withheld depth source image")
        read_depth_files.add(rel_path)
        p = safe_path(tum, rel_path)
        if p.resolve() in training_depth_paths:
            raise DataError(f"Holdout depth reuses training measurement: {name}")
        data = read_measured_depth(
            p, DEPTH_UNITS_PER_M,
            (cameras[next(iter(selected_ids))].height,
             cameras[next(iter(selected_ids))].width))
        z_observed = data[::DOWNSAMPLE, ::DOWNSAMPLE].astype(np.float32)
        pose = rigid_matrix(estimated[name]["camera_to_metric_sfm_world"])
        prediction, raster = rasterize_mesh_depth(
            vertices, faces, camera, pose, near=.15, far=MAX_DEPTH_M)
        if prediction.shape != z_observed.shape:
            raise DataError("Predicted camera-Z pixels are not aligned with the sensor frame")
        mask = (z_observed >= .15) & (z_observed <= MAX_DEPTH_M)
        overlap = mask & (prediction >= .15) & (prediction <= MAX_DEPTH_M)
        valid = int(mask.sum())
        covered = int(overlap.sum())
        total_valid += valid
        row = {
            "name": name, "sensor_timestamp_delta_s": gap,
            "sensor_depth_sha256": file_sha256(p),
            "sensor_valid_pixels": valid, "overlap_pixels": covered,
            "missing_surface_pixels": valid-covered,
            "observed_depth_coverage": covered/valid if valid else 0,
            "raster_diagnostics": raster,
            "source_pose": "sensor-metric SfM; NOT TUM GT"
        }
        if covered and valid:
            details = compare_holdout_depth(prediction, z_observed, max_depth=MAX_DEPTH_M)
            residual = prediction[overlap].astype(np.float64) - z_observed[overlap].astype(np.float64)
            all_residuals.append(residual)
            row.update({"status": "measured", "mae_m": details["mae_m"],
                        "rmse_m": details["rmse_m"], "p95_abs_m": details["p95_abs_m"],
                        "within_3cm":details["within_3cm"],
                        "within_5cm":details["within_5cm"],
                        "within_10cm":details["within_10cm"]})
        else:
            row["status"] = "no_overlap"
        per_frame.append(row)
        receipt["progress_evaluated_frames"] = per_frame
    if len(per_frame) != 12:
        raise DataError("Partial withheld evaluation")
    return aggregate_scores(all_residuals, total_valid, per_frame)


def experiment(args: argparse.Namespace, receipt: dict) -> None:
    args.report_dir.mkdir(parents=True, exist_ok=True)
    receipt["b_path_started_before_holdout_access"] = True
    b = {"git_sha": receipt["git_sha"]}
    # Uses previously frozen #15 input and gates, including the original 48/12 split.
    run_B_mesh(args, b)
    receipt["b_mesh_result"] = b["mesh_result"]
    receipt["b_mesh_stage_status"] = "passed"
    receipt["b_dataset_sha256"] = b["metric_stage"]["source_sha256"]
    receipt["b_fixed_split_sha256"] = b["split_sha256"]
    receipt["b_mesh_sha256"] = b["mesh_result"]["mesh_sha256"]
    receipt["b_mesh_generated_before_holdout_read"] = True
    heldout = evaluate_locked_B_run(args.work.resolve(), args.report_dir.resolve(), receipt)
    receipt["holdout_evaluation"] = heldout
    receipt.pop("progress_evaluated_frames", None)
    (args.report_dir / "holdout-depth-comparison.json").write_text(
        json.dumps(heldout, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    if not heldout["gate"]["passed"]:
        raise DataError("B-48/12 withheld-depth gate failed; preserve numbers, no retuning")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    receipt = {
        "experiment": "ECHO4D-B-mesh-estimated-poses-sensor-depth-12-view-holdout-v1",
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": os.environ.get("ECHO_HEAD_SHA") or os.environ.get("GITHUB_SHA"),
        "run_url": (f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/"
                    f"{os.getenv('GITHUB_RUN_ID')}") if os.getenv("GITHUB_RUN_ID") else None,
        "strict_48_12_split": True,
        "sensor_based_metric_scale": True,
        "uses_tum_gt_pose_to_score_mesh": False,
        "limits": ["12 holdout depths never used for TSDF fusion or scale calibration",
                   "Earlier #14 sparse diagnostic viewed some of these sensor holdout frames",
                   "Measured and predicted camera Z refer to the same TUM device convention but time offsets remain",
                   "No independent laser or dimensional geometry GT; mesh has unobserved regions",
                   "Per-pixel RMSE conditioned on overlap; missing coverage reported separately"]
    }
    code = 0
    try:
        experiment(args, receipt)
        receipt["status"] = "passed_sensor_B_mesh_12view_depth_gate_not_independent_surface_truth"
    except Exception as exc:
        code = 1
        receipt["status"] = "failed"
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
        print("B_HOLDOUT_FAILED:", type(exc).__name__, str(exc), file=sys.stderr)
    finally:
        receipt["elapsed_seconds"] = round(time.perf_counter()-started,3)
        args.report_dir.mkdir(parents=True,exist_ok=True)
        (args.report_dir/"holdout-run-report.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
        h = receipt.get("holdout_evaluation", {})
        summary = [
            "# B-path estimated pose / real depth: 12-view heldout depth experiment","",
            f"- Status: {receipt['status']}",
            f"- [Actions run]({receipt['run_url']})",
            f"- Mesh SHA256: {receipt.get('b_mesh_sha256','not available')}",
            f"- Heldout frames: {h.get('heldout_frames','not completed')}",
            f"- Sensor valid pixels: {h.get('sensor_valid_pixels','not completed')}",
            f"- Mesh-vs-sensor overlap: {h.get('overlap_pixels','not completed')}",
            f"- Observed depth coverage: {h.get('observed_depth_coverage','not completed')}",
            f"- Overlap RMSE m: {h.get('rmse_m','not completed')}",
            f"- Overlap MAE m: {h.get('mae_m','not completed')}",
            f"- Gate passes: {h.get('gate',{}).get('passed','not completed')}",
            f"- Failure: {receipt.get('error',{}).get('message','none')}","",
            "Only measured TUM heldout Z view consistency; no independent geometry/laser reference."
        ]
        (args.report_dir/"HOLDOUT_SUMMARY.md").write_text("\n".join(summary)+"\n",encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
