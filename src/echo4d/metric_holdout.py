"""Held-out measured-Z view consistency for B-path sensor-scale SfM Mesh.

Evaluation-only: this module is called AFTER TSDF fusion and never changes its
metric scale, camera poses, bounds, or mesh. No TUM ground-truth pose is read.
Source depth is the same physical RGB-D sensor, not independent scan truth.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .colmap import _cameras
from .holdout import read_echo_ply, rasterize_mesh_depth, compare_holdout_depth
from .metric_anchor import read_measured_depth
from .model import DataError, file_sha256, load_recording, rigid_matrix, safe_path
from .tum import nearest_timestamp, read_rows


def validate_holdout_split(split: dict, source: list[dict],
                           metric: dict) -> dict[str, dict]:
    """Require *all* 12 predeclared holdouts to have registered, metric poses."""
    train = split.get("train")
    withheld = split.get("holdout")
    if (not isinstance(train, list) or len(train) != 48 or
            not isinstance(withheld, list) or len(withheld) != 12):
        raise DataError("Expected fixed 48 training / 12 withheld RGB views")
    training_names = [row.get("name") for row in train]
    if len(set(training_names)) != 48 or len(set(withheld)) != 12 or set(training_names) & set(withheld):
        raise DataError("Train and holdout names must be distinct and disjoint")
    if split.get("source_kind") != "measured_registered_z_depth":
        raise DataError("Holdout evaluation expects measured sensor depth")
    source_map = {x["name"]: x for x in source}
    metric_map = {x["name"]: x for x in metric["frames"]}
    if len(source_map) != len(source) or len(metric_map) != len(metric["frames"]):
        raise DataError("Duplicate source or metric pose names")
    if not set(withheld).issubset(source_map) or not set(withheld).issubset(metric_map):
        raise DataError("Some withheld frames have no original RGB source or registered metric pose")
    if not isinstance(metric.get("scale_m_per_sfm_unit"), (float, int)) or not math.isfinite(
            metric["scale_m_per_sfm_unit"]) or metric["scale_m_per_sfm_unit"] <= 0:
        raise DataError("Holdout metric trajectory requires a previously measured scale")
    if metric.get("metric_scale_estimated_from_measured_depth") is not True:
        raise DataError("Not a measured-depth anchored metric trajectory")
    anchors = [x.get("name") for x in metric["anchor_diagnostics"]]
    if set(anchors) & set(withheld):
        raise DataError("Scale anchor is also in heldout evaluation group")
    return {name: metric_map[name] for name in withheld}


def evaluate_sensor_metric_holdouts(
    mesh_path: Path,
    training_recording: Path,
    metric: dict,
    split: dict,
    source: list[dict],
    dataset_root: Path,
    model_path: Path,
    downsample: int = 4,
    max_gap_s: float = .03,
    max_depth_m: float = 4.,
    known_poses_for_A: dict[str, list[float]] | None = None,
) -> dict:
    metric_poses = validate_holdout_split(split, source, metric)
    # A/B uses the identical triangle rasterizer and pixel scoring method.
    # Ground-truth pose substitution is permitted ONLY for the explicitly
    # labeled A arm, and never alters the estimated SfM / measured-scale B arm.
    if known_poses_for_A is not None:
        if set(known_poses_for_A) != set(split["holdout"]):
            raise DataError("A reference poses must cover exactly all twelve heldout views")
        for name in split["holdout"]:
            rigid_matrix(known_poses_for_A[name])
    dataset_root = Path(dataset_root).resolve()
    camera = load_recording(training_recording).camera
    cams = _cameras(Path(model_path) / "cameras.txt")
    if len(cams) != 1:
        raise DataError("Only one registered camera is supported by B-path holdout")
    original = next(iter(cams.values()))
    if not 1 <= downsample <= 16:
        raise DataError("Invalid holdout downsample")
    scaled = original.scaled(downsample)
    if any(not np.isclose(getattr(camera, key), getattr(scaled, key), atol=1e-8)
           for key in ("width", "height", "fx", "fy", "cx", "cy")):
        raise DataError("Holdout camera must exactly match B-path training calibration")
    vertices, faces = read_echo_ply(mesh_path)
    index = read_rows(dataset_root / "depth.txt", 2)
    timestamps = np.asarray([float(row[0]) for row in index])
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise DataError("TUM measured depth index is not strictly ordered")
    source_lookup = {x["name"]: x for x in source}
    used_train = {x["depth"] for x in split["train"]}
    used_eval = set()
    per_frame, residuals = [], []
    total_valid = total_overlap = 0
    for name in split["holdout"]:  # preserve frozen order; do NOT select easy frames
        source_time = float(source_lookup[name]["t_tum_rgb_s"])
        if not math.isfinite(source_time):
            raise DataError("Invalid source RGB time in holdout")
        found = nearest_timestamp(timestamps, source_time, max_gap_s)
        if found is None:
            raise DataError(f"No measured holdout Z frame within 30ms: {name}")
        relative = index[found][1]
        if relative in used_train or relative in used_eval:
            raise DataError("Depth frame was used in training or twice in holdout")
        used_eval.add(relative)
        entry = metric_poses[name]
        model_id = entry.get("camera_id")
        if model_id not in cams:
            raise DataError("Holdout pose references an unknown camera")
        pose = rigid_matrix(known_poses_for_A[name] if known_poses_for_A is not None
                            else entry["camera_to_metric_sfm_world"])
        measured = read_measured_depth(safe_path(dataset_root, relative),
                                      float(split["depth_numeric_units_per_meter"]),
                                      (original.height, original.width))
        downsampled = measured[::downsample, ::downsample].astype(np.float32)
        predicted, raster = rasterize_mesh_depth(vertices, faces, camera, pose,
                                                 far=max_depth_m)
        score = compare_holdout_depth(predicted, downsampled, max_depth=max_depth_m)
        valid = (downsampled >= .15) & (downsampled <= max_depth_m)
        overlap = valid & (predicted >= .15) & (predicted <= max_depth_m)
        residuals.append((predicted[overlap] - downsampled[overlap]).astype(np.float64))
        total_valid += score["sensor_valid_pixels"]
        total_overlap += score["overlap_pixels"]
        per_frame.append({
            "name": name, "source_rgb_t_s": source_time,
            "depth_filename": relative,
            "depth_sha256": file_sha256(safe_path(dataset_root, relative)),
            "rgb_depth_delta_s": float(timestamps[found] - source_time),
            "score": score, "raster": raster,
        })
    if len(per_frame) != 12 or total_valid <= 0 or total_overlap <= 0:
        raise DataError("Incomplete withheld measured depth evaluation")
    errors = np.concatenate(residuals)
    absolute = np.abs(errors)
    coverage = total_overlap / total_valid
    rmse = float(np.sqrt(np.mean(errors**2)))
    return {
        "version": "echo-b-mesh-heldout-1",
        "source_kind": "measured_registered_z_depth",
        "mesh_sha256": file_sha256(mesh_path),
        "pose_kind": ("TUM measured ground-truth camera pose (A reference arm)"
                      if known_poses_for_A is not None else
                      "RGB COLMAP + external three-frame sensor depth scale"),
        "gt_camera_poses_loaded": known_poses_for_A is not None,
        "holdout_frames": 12,
        "valid_measured_pixels": total_valid,
        "overlap_pixels": total_overlap,
        "missing_surface_pixels": total_valid - total_overlap,
        "coverage": float(coverage),
        "mae_m": float(np.mean(absolute)),
        "bias_m": float(np.mean(errors)),
        "rmse_m": rmse,
        "median_abs_m": float(np.median(absolute)),
        "p95_abs_m": float(np.percentile(absolute, 95)),
        "within_3cm": float(np.mean(absolute <= .03)),
        "within_5cm": float(np.mean(absolute <= .05)),
        "within_10cm": float(np.mean(absolute <= .10)),
        "within_30cm": float(np.mean(absolute <= .30)),
        "screening_gate_coverage_gte_80pct": bool(coverage >= .80),
        "screening_gate_overlap_rmse_lte_15cm": bool(rmse <= .15),
        "screening_gate_passed": bool(coverage >= .80 and rmse <= .15),
        "per_frame": per_frame,
        "limitations": [
            "Holdout sensor depths were not fused into TSDF or used to compute bounds.",
            "Scale came from 3 separate measured sensor-depth anchors, never GT Sim(3).",
            ("A arm uses TUM ground-truth poses ONLY as explicit reference." if
             known_poses_for_A is not None else
             "B arm uses estimated metric SfM poses, not TUM ground-truth camera poses."),
            "Depth and scale use the same public RGB-D sensor: NOT external laser-scan surface truth.",
            "Neighboring frames and underlying sensor share errors and scene observations.",
            "MAE/RMSE are conditional on overlapping valid pixels; missing geometry is reported by coverage.",
            "First pass screening thresholds are research-only and do not certify consumer-level 3D accuracy.",
        ],
    }
