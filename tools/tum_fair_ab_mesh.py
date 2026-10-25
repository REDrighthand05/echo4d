"""Frozen same-48-train/12-heldout A/B TUM mesh comparison.

A: original TUM ground-truth camera poses + SAME 48 measured depth maps.
B: estimated RGB COLMAP poses + 3 *measured sensor* scale anchors + SAME depths.
Only injects GT poses into A AFTER B Mesh/scoring completes; the B anchor
stage may read GT earlier strictly for post-calibration diagnostics. The 12 withheld Z
frames are never used to build either Mesh or optimize B's metric scale.
This is same-sensor view consistency, NOT independent scanned-surface GT.
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
from echo4d.colmap import _cameras
from echo4d.holdout import read_echo_ply, rasterize_mesh_depth, compare_holdout_depth
from echo4d.metric_anchor import read_measured_depth
from echo4d.model import DataError, file_sha256, load_recording, rigid_matrix, safe_path
from echo4d.tum import nearest_timestamp, read_rows
from tum_public_roundtrip import ROOT, PINNED_ARCHIVE_SHA256
from tum_rgb_sfm import gt_after_sfm
from tum_sensor_metric_mesh import (
    depth_only_training_bounds, VOXEL_M, MAX_VOXELS, MAX_DEPTH_M,
    MIN_OBSERVATIONS, DOWNSAMPLE, MAX_RGB_DEPTH_GAP_S)
from tum_sensor_mesh_holdout import (
    aggregate_scores, experiment as execute_B, validate_frozen_holdout_contract)


def make_A_recording(b_record: Path, provenance: Path, split: dict,
                     gt: dict[str, list[float]]) -> Path:
    """Create GT-pose-only view of identical training pixels; don't copy depth."""
    b = json.loads(Path(b_record).read_text(encoding="utf-8"))
    audit = json.loads(Path(provenance).read_text(encoding="utf-8"))
    receipts = audit["train"]
    if len(b["frames"]) != 48 or len(receipts) != 48:
        raise DataError("A/B requires the same 48 B-source measured depth frames")
    expected = {r["name"] for r in split["train"]}
    if {r["name"] for r in receipts} != expected or len(expected) != 48:
        raise DataError("B input provenance and frozen training names disagree")
    if len({r["name"] for r in receipts}) != 48:
        raise DataError("Duplicated B training source name")
    # Fail before publishing or altering any A pose if GT has missing entries.
    missing = [r["name"] for r in receipts if r["name"] not in gt]
    if missing:
        raise DataError(f"GT unavailable for frozen training frame {missing[0]}")
    for frame, receipt in zip(b["frames"], receipts):
        name = receipt["name"]
        if name not in gt:
            raise DataError(f"GT unavailable for frozen training frame {name}")
        # All other fields (depth and RGB relative paths, timestamps and camera)
        # are UNCHANGED. A differs from B only in calibrated camera poses.
        frame["cameraToWorld"] = rigid_matrix(gt[name]).flatten(order="F").tolist()
    b["name"] = "A-GT-poses-same-B-measured-Z-train48"
    b["notes"] = "A baseline only: GT camera poses; exact same measured RGB/depth as B. No heldout Z fused."
    out = Path(b_record).with_name("recording.A-groundtruth-poses.json")
    if out.exists():
        raise DataError("A baseline recording already exists")
    out.write_text(json.dumps(b, indent=2, ensure_ascii=False, allow_nan=False)+"\n",
                   encoding="utf-8")
    parsed = load_recording(out)
    if len(parsed) != 48:
        raise DataError("A baseline camera/depth manifest failed ECHO validation")
    for _ in parsed.iter_frames():
        pass
    return out


def score_A_holdout(
    mesh_file: Path, train_record: Path, tum: Path,
    source: list[dict], split: dict, gt: dict[str, list[float]],
    model_dir: Path,
) -> dict:
    """Use the exact same triangle-Z and weighted pixel scoring as #26 B."""
    camera = load_recording(train_record).camera
    cameras = _cameras(model_dir / "cameras.txt")
    if len(cameras) != 1:
        raise DataError("A/B require identical single calibrated pinhole camera")
    original = next(iter(cameras.values()))
    scaled = original.scaled(DOWNSAMPLE)
    if any(not np.isclose(getattr(camera, k), getattr(scaled, k), atol=1e-8, rtol=0)
           for k in ("width", "height", "fx", "fy", "cx", "cy")):
        raise DataError("A camera calibration differs from frozen B calibration")
    vertices, faces = read_echo_ply(mesh_file)
    rows = read_rows(tum / "depth.txt", 2)
    times = np.asarray([float(row[0]) for row in rows])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise DataError("Unordered depth timestamps")
    sources = {x["name"]: x for x in source}
    training_depth = {safe_path(tum, row["depth"]).resolve()
                      for row in split["train"]}
    seen, all_residuals, frame_scores = set(), [], []
    total_valid = 0
    for name in split["holdout"]:
        if name not in gt:
            raise DataError(f"GT missing for A holdout {name}; cannot pick easier frame")
        rgb_time = float(sources[name]["t_tum_rgb_s"])
        index = nearest_timestamp(times, rgb_time, MAX_RGB_DEPTH_GAP_S)
        if index is None:
            raise DataError(f"No time-matched heldout sensor Z for {name}")
        depth_file = safe_path(tum, rows[index][1]).resolve()
        if depth_file in seen or depth_file in training_depth:
            raise DataError("A holdout reused a training/other withheld measured depth")
        seen.add(depth_file)
        measured = read_measured_depth(
            depth_file, float(split["depth_numeric_units_per_meter"]),
            (original.height, original.width))[::DOWNSAMPLE, ::DOWNSAMPLE].astype(np.float32)
        estimate, raster = rasterize_mesh_depth(
            vertices, faces, camera, rigid_matrix(gt[name]),
            near=.15, far=MAX_DEPTH_M)
        valid_mask = (measured >= .15) & (measured <= MAX_DEPTH_M)
        overlap = valid_mask & (estimate >= .15) & (estimate <= MAX_DEPTH_M)
        count = int(np.count_nonzero(overlap))
        valid = int(np.count_nonzero(valid_mask))
        total_valid += valid
        row = {"name": name, "sensor_valid_pixels": valid,
               "overlap_pixels": count,
               "missing_surface_pixels": valid - count,
               "observed_depth_coverage": count / valid if valid else 0,
               "sensor_timestamp_delta_s": float(times[index] - rgb_time),
               "sensor_depth_sha256": file_sha256(depth_file),
               "source_pose": "TUM GT pose — A reference baseline only",
               "raster_diagnostics": raster,
               "status": "measured" if count else "no_overlap"}
        if count:
            stats = compare_holdout_depth(estimate, measured, max_depth=MAX_DEPTH_M)
            residual = estimate[overlap].astype(np.float64) - measured[overlap].astype(np.float64)
            all_residuals.append(residual)
            row.update({k: stats[k] for k in (
                "mae_m", "rmse_m", "p95_abs_m", "within_3cm", "within_5cm", "within_10cm")})
        frame_scores.append(row)
    scores = aggregate_scores(all_residuals, total_valid, frame_scores)
    if scores["heldout_frames"] != 12:
        raise DataError("A failed to score every original withheld frame")
    return scores


def assert_same_comparison_inputs(a: dict, b: dict) -> None:
    """Strictly require same names, sensor bytes and measured-valid pixel counts."""
    if (a["heldout_frames"], a["sensor_valid_pixels"]) != (
            b["heldout_frames"], b["sensor_valid_pixels"]):
        raise DataError("A/B valid source pixel counts differ")
    if len(a["per_frame"]) != 12 or len(b["per_frame"]) != 12:
        raise DataError("A/B missing some fixed heldout view")
    for left, right in zip(a["per_frame"], b["per_frame"]):
        if (left["name"] != right["name"] or
                left["sensor_depth_sha256"] != right["sensor_depth_sha256"] or
                left["sensor_valid_pixels"] != right["sensor_valid_pixels"]):
            raise DataError("A/B used different heldout images, pixel masks or order")


def run(args: argparse.Namespace, report: dict) -> None:
    args.report_dir.mkdir(parents=True, exist_ok=False)
    b_receipt = {"git_sha": report["git_sha"], "status": "running"}
    # B mesh and 12-view heldout run finishes BEFORE this function reads GT.
    execute_B(args, b_receipt)
    report["B_mesh_scored_before_A_gt_pose_injection"] = True
    report["B_mesh_sha256"] = b_receipt["b_mesh_sha256"]
    report["B_score"] = b_receipt["holdout_evaluation"]
    report["B_split_sha256"] = b_receipt["b_fixed_split_sha256"]
    report["dataset_sha256"] = b_receipt["b_dataset_sha256"]
    if report["dataset_sha256"] != PINNED_ARCHIVE_SHA256:
        raise DataError("A/B unexpected public dataset archive SHA256")

    root = args.work.resolve()
    output = args.report_dir.resolve()
    split_path = output / "b-48train-12holdout.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    metric_receipt = json.loads((output / "metric" / "report.json").read_text(encoding="utf-8"))
    source = metric_receipt["test_video"]["source_rgb_manifest"]
    metric = json.loads((output / "metric" / "metric-trajectory-from-sensor.json").read_text(encoding="utf-8"))
    validate_frozen_holdout_contract(split, metric, source)
    tum = root / "unpacked" / ROOT
    # The groundtruth is an explicit A/reference input; B cannot touch it here.
    gt, unmatched = gt_after_sfm(tum, source)
    if any(n not in gt for n in
           [r["name"] for r in split["train"]] + split["holdout"]):
        raise DataError("A reference GT has missing frozen train/heldout poses")
    report["A_gt_groundtruth_txt_sha256"] = file_sha256(tum / "groundtruth.txt")
    report["A_gt_unmatched_other_names"] = unmatched
    original_record = root / "b-sensor-recording" / "recording.json"
    train_gt = make_A_recording(
        original_record, root / "b-sensor-recording" / "sensor-provenance.json", split, gt)
    report["A_train_record_sha256"] = file_sha256(train_gt)
    bounds = depth_only_training_bounds(train_gt)
    report["A_bounds_from_same_training_sensor_Z_only"] = bounds
    model_receipt = json.loads((root / "rgb-only-colmap" / "report.json").read_text(encoding="utf-8"))
    model = root / "rgb-only-colmap" / model_receipt["result"]["sfm_model_dir"]
    result = reconstruct_command(argparse.Namespace(
        recording=train_gt, output=output / "A-GT-pose-mesh",
        bounds=bounds, voxel=VOXEL_M, truncation=None, max_depth=MAX_DEPTH_M,
        max_voxels=MAX_VOXELS, min_observations=MIN_OBSERVATIONS,
        stride=1, limit=None, reference=None, threshold=.05,
        no_scene=False, save_volume=False, quiet=True))
    if result["frames"] != 48 or not result["vertices"] or not result["triangles"]:
        raise DataError("A GT-pose mesh failed frozen 48-train reconstruction")
    report["A_mesh"] = result
    a = score_A_holdout(output / "A-GT-pose-mesh" / "mesh.ply",
                        train_gt, tum, source, split, gt, model)
    assert_same_comparison_inputs(a, report["B_score"])
    report["A_score"] = a
    report["score_input_identity_verified"] = True
    report["A_B_difference_B_minus_A"] = {
        "coverage_fraction": report["B_score"]["observed_depth_coverage"] - a["observed_depth_coverage"],
        "overlap_rmse_m": report["B_score"]["rmse_m"] - a["rmse_m"],
        "overlap_mae_m": report["B_score"]["mae_m"] - a["mae_m"]
    }
    # Relative differences are diagnostic, not a causal localization-only effect;
    # reconstructed mesh and bounds can change nonlinearly with the poses.
    (output / "A_B_metrics.json").write_text(
        json.dumps({"A": a, "B": report["B_score"], "difference_B_minus_A": report["A_B_difference_B_minus_A"],
                    "identical_heldout_sensor_data": True,
                    "protocol": "Frozen 48/12; A GT pose, B sensor-scaled SfM pose; same training RGB/D",
                    "limitations": ["Same TUM sensor/scene, not independent scanned geometry.",
                                    "A GT poses only used after the complete B workflow.",
                                    "Mesh/bounds differ naturally with input poses; differences are not pure localization-only causal attribution."]},
                   indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    report["A_B_metric_file_sha256"] = file_sha256(output / "A_B_metrics.json")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, required=True)
    args = p.parse_args()
    started = time.perf_counter()
    report = {"schema": "echo4d-frozen-A-B-mesh-1",
              "status": "running",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "git_sha": os.getenv("ECHO_HEAD_SHA") or os.getenv("GITHUB_SHA"),
              "run_url": (f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/"
                          f"{os.getenv('GITHUB_RUN_ID')}") if os.getenv("GITHUB_RUN_ID") else None,
              "training": 48, "withheld_views": 12,
              "A_pose": "TUM GT, evaluation reference baseline",
              "B_pose": "COLMAP real RGB + 3 separate measured depth scale anchors",
              "depth_in_both": "same measured TUM registered RGB-D sensor, camera Z, not RGB-only",
              "no_external_scanned_surface_GT": True}
    code = 0
    try:
        run(args, report)
        report["status"] = "passed_frozen_same_input_comparison_not_independent_surface_truth"
    except Exception as exc:
        code = 1
        report["status"] = "failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)[:1500]}
        print("A_B_FAILED:", type(exc).__name__, str(exc), file=sys.stderr)
    finally:
        output = args.report_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        report["duration_seconds"] = round(time.perf_counter() - started, 3)
        (output / "A_B_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n",
            encoding="utf-8")
        a = report.get("A_score", {})
        b = report.get("B_score", {})
        lines = [
            "# Frozen TUM RGB-D A/B comparison (48 train / 12 heldout)", "",
            f"Status: **{report['status']}**",
            f"Git head: {report['git_sha']}",
            f"Run: {report['run_url']}",
            f"Dataset SHA256: {report.get('dataset_sha256','not finished')}",
            f"Split SHA256: {report.get('B_split_sha256','not finished')}",
            f"A GT-pose Mesh SHA256: {report.get('A_mesh',{}).get('mesh_sha256','not finished')}",
            f"B estimated-pose Mesh SHA256: {report.get('B_mesh_sha256','not finished')}",
            f"A GT-pose coverage / overlap RMSE(m): {a.get('observed_depth_coverage','-')} / {a.get('rmse_m','-')}",
            f"B SfM-pose coverage / overlap RMSE(m): {b.get('observed_depth_coverage','-')} / {b.get('rmse_m','-')}",
            f"Identical heldout sensor file hashes and valid pixels: {report.get('score_input_identity_verified',False)}",
            f"Failure: {report.get('failure',{}).get('message','none')}",
            "",
            "**External measured depth is used for BOTH; only A uses GT camera poses. Neither path is RGB-only dense depth.**",
            "Shared same-sensor observations are NOT independent geometric truth."
        ]
        (output / "A_B_SUMMARY.md").write_text("\n\n".join(lines)+"\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
