"""Pre-registered E3 attempt: actual RGB SfM + 3 sensor anchors + measured depth + TSDF.

No RGB-only metric claim. No holdout depth is opened during SfM, metric
calibration (except #14's distinct sparse diagnostics), dataset conversion,
or mesh fusion. There is no independent scanned-surface GT here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from echo4d.cli import reconstruct_command
from echo4d.metric_rgbd import convert_metric_sensor
from echo4d.model import DataError, file_sha256, load_recording
from echo4d.tum import nearest_timestamp, read_rows
from tum_metric_anchor import experiment as run_sensor_metric_anchor
from tum_public_roundtrip import ROOT, PINNED_ARCHIVE_SHA256

# Fixed in Issue #15 before this real experiment executes.
SELECTED_COUNT = 60
HOLDOUT_INDICES = (4, 5, 9, 10, 14, 19, 24, 25, 29, 34, 35, 55)
ANCHOR_INDICES = (15, 45, 75)
MAX_RGB_DEPTH_GAP_S = .03
DOWNSAMPLE = 4
PIXEL_SAMPLE_STEP = 10
Z_RANGE = (.2, 3.5)
BBOX_PERCENTILES = (2.0, 98.0)
BBOX_MARGIN_M = .3
VOXEL_M = .07
MAX_VOXELS = 1_500_000
MAX_DEPTH_M = 4.0
MIN_OBSERVATIONS = 2


def split_names(source: list[dict]) -> tuple[list[str], list[str]]:
    if len(source) != 90:
        raise DataError("Expected 90 fixed TUM RGB source entries")
    if len(set(HOLDOUT_INDICES)) != 12 or any(i in HOLDOUT_INDICES for i in ANCHOR_INDICES):
        raise DataError("Pre-registered holdout split is invalid")
    selected = [entry["name"] for entry in source[:SELECTED_COUNT]]
    if len(set(selected)) != SELECTED_COUNT:
        raise DataError("Duplicate selected source frame name")
    holdout = [selected[i] for i in HOLDOUT_INDICES]
    training = [s for s in selected if s not in set(holdout)]
    if len(training) != 48 or len(holdout) != 12:
        raise DataError("Pre-registered train/holdout counts changed")
    return training, holdout


def build_split(source: list[dict], tum: Path) -> dict:
    training, holdout = split_names(source)
    depth_rows = read_rows(tum / "depth.txt", 2)
    times = np.asarray([float(row[0]) for row in depth_rows])
    mapping = {row["name"]: row for row in source}
    used_files, train = set(), []
    for name in training:
        rgb_time = float(mapping[name]["t_tum_rgb_s"])
        found = nearest_timestamp(times, rgb_time, MAX_RGB_DEPTH_GAP_S)
        if found is None:
            raise DataError(f"No synchronized sensor depth for {name}")
        path = depth_rows[found][1]
        if path in used_files:
            raise DataError("A measured depth image is reused by two training RGB images")
        used_files.add(path)
        train.append({"name": name, "depth": path,
                      "timestamp_delta_seconds": float(times[found] - rgb_time)})
    return {
        "source_kind": "measured_registered_z_depth",
        "source_description": "TUM Freiburg1 XYZ registered measured RGB-D Z sensor; official archive hash pinned; external to RGB SfM",
        "depth_numeric_units_per_meter": 5000,
        "train": train, "holdout": holdout,
        "protocol": {"source_rgb_frames": 90, "selected_frames": SELECTED_COUNT,
                     "holdout_indices": HOLDOUT_INDICES,
                     "external_scale_anchor_indices": ANCHOR_INDICES,
                     "max_rgb_depth_gap_s": MAX_RGB_DEPTH_GAP_S},
    }


def depth_only_training_bounds(recording_path: Path) -> tuple[float, ...]:
    """Fixed bounds from training measurements; never consult holdout or GT."""
    recording = load_recording(recording_path)
    cam = recording.camera
    points = []
    for _, frame, _ in recording.iter_frames():
        z = frame.depth[::PIXEL_SAMPLE_STEP, ::PIXEL_SAMPLE_STEP]
        ys, xs = np.mgrid[0:frame.depth.shape[0]:PIXEL_SAMPLE_STEP,
                          0:frame.depth.shape[1]:PIXEL_SAMPLE_STEP]
        valid = np.isfinite(z) & (z >= Z_RANGE[0]) & (z <= Z_RANGE[1])
        if not np.any(valid):
            continue
        depth = z[valid].astype(float)
        u, v = xs[valid], ys[valid]
        local = np.stack([(u - cam.cx)*depth/cam.fx,
                          (v - cam.cy)*depth/cam.fy, depth], axis=1)
        xyz = (frame.camera_to_world[:3,:3] @ local.T).T + frame.camera_to_world[:3,3]
        points.append(xyz)
    if not points:
        raise DataError("No measured training depth points to compute fixed bounds")
    cloud = np.concatenate(points)
    if len(cloud) < 100:
        raise DataError("Training depth support insufficient to compute TSDF bounds")
    lo = np.percentile(cloud, BBOX_PERCENTILES[0], axis=0) - BBOX_MARGIN_M
    hi = np.percentile(cloud, BBOX_PERCENTILES[1], axis=0) + BBOX_MARGIN_M
    if np.any(hi <= lo) or not np.isfinite([*lo, *hi]).all():
        raise DataError("Invalid metric estimated-volume bounds")
    shape = np.ceil((hi-lo)/VOXEL_M).astype(np.int64)
    if np.prod(shape, dtype=np.int64) > MAX_VOXELS:
        raise DataError("Pre-registered bounds exceed 1.5M voxels; do not silently crop")
    return (float(lo[0]), float(lo[1]), float(lo[2]),
            float(hi[0]), float(hi[1]), float(hi[2]))


def experiment(args: argparse.Namespace, report: dict) -> None:
    work = args.work.resolve()
    target = args.report_dir.resolve()
    if work.exists():
        raise DataError("Work directory must not already exist")
    (target / "metric").mkdir(parents=True)
    metric_report = {
        "experiment": "reused TUM 90-frame actual RGB SfM plus sensor metric anchor",
        "commands": [], "git_sha": report["git_sha"], "status": "running"
    }
    run_sensor_metric_anchor(argparse.Namespace(work=work,report_dir=target/"metric"),metric_report)
    report["metric_stage"] = {
        "source_sha256": metric_report["dataset"]["sha256"],
        "scale_m_per_sfm_unit": metric_report["metric_sensor_anchor"]["scale_m_per_sfm_unit"],
        "metric_se3_heldout_rmse_m": metric_report["metric_se3_pose_evaluation"]["heldout_position_rmse_m"],
        "gt_only_for_post_calibration_evaluation": True,
    }
    # Direct calls bypass tum_metric_anchor.main(), so finalize the nested
    # receipt explicitly; an E3-complete run must never say status=running.
    metric_report["status"] = "passed_sensor_metric_calibration_not_rgb_only_metric"
    (target/"metric"/"report.json").write_text(
        json.dumps(metric_report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    if metric_report["dataset"]["sha256"] != PINNED_ARCHIVE_SHA256:
        raise DataError("Dataset hash differs from pinned public TUM archive")
    sfm = work/"rgb-only-colmap"
    sfm_report = json.loads((sfm/"report.json").read_text(encoding="utf-8"))
    model = sfm/sfm_report["result"]["sfm_model_dir"]
    source = metric_report["test_video"]["source_rgb_manifest"]
    tum = work/"unpacked"/ROOT
    split = build_split(source,tum)
    split_path = target/"b-48train-12holdout.json"
    split_path.write_text(json.dumps(split,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    metric_path = target/"metric"/"metric-trajectory-from-sensor.json"
    input_dir = work/"b-sensor-recording"
    conversion = convert_metric_sensor(
        metric_path, sfm/"trajectory-scale-unknown.json", model, sfm/"images",
        tum, split_path, input_dir, downsample=DOWNSAMPLE)
    report["dataset_conversion"] = conversion
    report["split_sha256"] = file_sha256(split_path)
    report["input_manifest_sha256"] = file_sha256(input_dir/"recording.json")
    report["sensor_provenance_sha256"] = file_sha256(input_dir/"sensor-provenance.json")
    bounds = depth_only_training_bounds(input_dir/"recording.json")
    report["pre_registered_bounds"] = bounds
    report["tsdf_config"] = {"voxel_m":VOXEL_M, "max_voxels":MAX_VOXELS,
                            "max_depth_m":MAX_DEPTH_M, "min_observations":MIN_OBSERVATIONS,
                            "percentiles":BBOX_PERCENTILES, "margin_m":BBOX_MARGIN_M}
    output = target/"mesh"
    result = reconstruct_command(argparse.Namespace(
        recording=input_dir/"recording.json", output=output,
        bounds=bounds, voxel=VOXEL_M, truncation=None, max_depth=MAX_DEPTH_M,
        max_voxels=MAX_VOXELS, min_observations=MIN_OBSERVATIONS,
        stride=1, limit=None, reference=None, threshold=.05,
        no_scene=False, save_volume=False, quiet=True))
    report["mesh_result"] = result
    if result["frames"] != 48 or result["vertices"] <= 0 or result["triangles"] <= 0:
        raise DataError("First B-path real mesh is empty or frame count differs from frozen train set")
    report["mesh_report_sha256"] = file_sha256(output/"report.json")
    report["input_kind"] = "estimated RGB SfM poses + independent measured sensor scale and per-frame measured dense depth"
    report["no_heldout_depth_used_in_mesh"] = True
    report["mesh_geometry_independently_validated"] = False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    args = parser.parse_args()
    report = args.report_dir.resolve()
    report.mkdir(parents=True, exist_ok=True)
    receipt = {
        "experiment": "ECHO4D-TUM-48train-12heldout-estimated-pose-sensor-depth-B-mesh-v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": os.environ.get("ECHO_HEAD_SHA") or os.environ.get("GITHUB_SHA"),
        "run_url": (f"https://github.com/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/"
                    f"{os.environ.get('GITHUB_RUN_ID')}") if os.environ.get("GITHUB_RUN_ID") else None,
        "status":"running",
        "declared_limitations": [
            "Not RGB-only depth: actual sensor supplies 3 metric scale anchors plus 48 dense training depth maps.",
            "TUM real RGB re-encoded to test MP4; not native phone footage.",
            "GT used only after scale calibration in reused anchor-stage analysis, not by SfM or fusion.",
            "No holdout frame depth read during dataset conversion, bounding or fusion.",
            "This stage does not compute heldout mesh projection accuracy or independent scanned-surface truth.",
            "Original #9 48/12 was a different source-frame selection; scores not comparable."
        ]
    }
    started = time.perf_counter()
    exit_code = 0
    try:
        experiment(args,receipt)
        receipt["status"] = "passed_real_sensor_depth_estimated_pose_mesh_no_independent_accuracy_claim"
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
        exit_code = 1
        print("SENSOR_MESH_FAILURE:",type(exc).__name__, str(exc), file=sys.stderr)
    finally:
        receipt["elapsed_seconds"] = round(time.perf_counter()-started,3)
        (report/"report.json").write_text(
            json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
        summary = [
            "# Echo4D B-path real TUM sensor-measured depth Mesh attempt", "",
            f"- Status: {receipt['status']}",
            f"- [Actual Actions run]({receipt['run_url']})",
            f"- Estimated-pose sensor-scale: {receipt.get('metric_stage',{}).get('scale_m_per_sfm_unit','not completed')}",
            f"- Train/holdout: {receipt.get('dataset_conversion',{}).get('training_frames','N/A')}/12",
            f"- Mesh vertices: {receipt.get('mesh_result',{}).get('vertices','not computed')}",
            f"- Mesh triangles: {receipt.get('mesh_result',{}).get('triangles','not computed')}",
            f"- Mesh SHA256: {receipt.get('mesh_result',{}).get('mesh_sha256','none')}",
            f"- Failure: {receipt.get('error',{}).get('message','none')}", "",
            "Not RGB-only, no heldout Mesh projection score and no independent surface GT.",
        ]
        (report/"SUMMARY.md").write_text("\n".join(summary)+"\n",encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
