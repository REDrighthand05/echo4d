"""Measured-depth B-path: sensor-scaled COLMAP poses + dense registered Z -> ECHO input.

This is NOT RGB-only dense depth or a geometry validation. Holdout images are
explicitly excluded and never read here. The caller supplies calibrated RGB
images and *measured* Z depth for training views only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .colmap import _cameras, _frame_name
from .metric_anchor import read_measured_depth
from .model import DataError, file_sha256, load_recording, rigid_matrix, safe_path
from .output import output_transaction, write_json


def _objects_by_name(items: Any, label: str) -> dict[str, dict]:
    if not isinstance(items, list):
        raise DataError(f"{label} must be a list")
    result = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise DataError(f"{label} entries need names")
        name = _frame_name(item["name"])
        if name in result:
            raise DataError(f"Duplicate {label} frame: {name}")
        result[name] = item
    return result


def convert_metric_sensor(
    metric_path: Path, sfm_path: Path, colmap_dir: Path,
    image_root: Path, depth_root: Path, split_path: Path,
    output: Path, downsample: int = 4,
) -> dict:
    """Write an ECHO dataset only after every training frame passes validation."""
    if type(downsample) is not int or not 1 <= downsample <= 16:
        raise DataError("Downsample must be an integer from 1 to 16")
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    sfm = json.loads(sfm_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))

    if (metric.get("version") != "echo-sfm-metric-scale-1"
            or metric.get("units") != "meters"
            or metric.get("metric_scale_estimated_from_measured_depth") is not True
            or metric.get("scale_is_from_external_gt_poses") is not False
            or metric.get("source_trajectory_sha256") != file_sha256(sfm_path)):
        raise DataError("Metric trajectory is not tied to sensor-measured SfM")
    if (sfm.get("version") != "echo-sfm-trajectory-1"
            or sfm.get("units") != "unknown"
            or sfm.get("metric_scale_estimated") is not False):
        raise DataError("Original trajectory must remain explicitly unknown-scale")
    scale = metric.get("scale_m_per_sfm_unit")
    if (isinstance(scale, bool) or not isinstance(scale, (int, float))
            or not math.isfinite(scale) or not 0 < scale <= 10000):
        raise DataError("No valid measured scale")
    if not isinstance(metric.get("anchor_diagnostics"), list) or len(metric["anchor_diagnostics"]) != 3:
        raise DataError("Measured scale needs three documented anchor views")

    for kind in ("images", "cameras"):
        model_file = colmap_dir / (kind + ".txt")
        if sfm.get("model_" + kind + "_txt_sha256") != file_sha256(model_file):
            raise DataError(f"Mixed COLMAP model: {kind}.txt hash mismatch")
    cams = _cameras(colmap_dir / "cameras.txt")

    if (split.get("source_kind") != "measured_registered_z_depth"
            or not isinstance(split.get("source_description"), str)
            or len(split["source_description"].strip()) < 12):
        raise DataError("Training depth must have explicit measured provenance")
    units = split.get("depth_numeric_units_per_meter")
    if isinstance(units, bool) or not isinstance(units, (int, float)) or not math.isfinite(units) or units <= 0:
        raise DataError("Measured depth requires numeric units per meter")
    training = _objects_by_name(split.get("train"), "train")
    holdout = split.get("holdout")
    if (not isinstance(holdout, list) or len(holdout) < 1
            or any(not isinstance(name, str) for name in holdout)):
        raise DataError("Declare at least one withheld validation frame")
    withheld = [_frame_name(name) for name in holdout]
    if len(set(withheld)) != len(withheld) or set(withheld) & set(training):
        raise DataError("Training/holdout names must be unique and disjoint")
    if len(training) < 2:
        raise DataError("At least two measured training frames are needed")

    src_frames = _objects_by_name(sfm.get("frames"), "original SfM")
    dst_frames = _objects_by_name(metric.get("frames"), "sensor-metric")
    if set(src_frames) != set(dst_frames) or not set(training).union(withheld).issubset(dst_frames):
        raise DataError("Training or holdout differs from registered SfM frames")
    anchors = [item.get("name") for item in metric["anchor_diagnostics"]]
    if any(name in withheld for name in anchors):
        raise DataError("A scale-calibration anchor cannot also be a dense holdout")

    prepared = []
    camera_ids = set()
    seen_depths = set()
    for name, item in training.items():
        base, entry = src_frames[name], dst_frames[name]
        if base.get("camera_id") != entry.get("camera_id"):
            raise DataError(f"Changed SfM camera: {name}")
        camera_id = entry.get("camera_id")
        if camera_id not in cams:
            raise DataError(f"Unknown camera calibration for {name}")
        camera_ids.add(camera_id)
        t = entry.get("t_s")
        if (isinstance(t, bool) or not isinstance(t, (int, float))
                or not math.isfinite(t) or t < 0 or
                not math.isclose(t, base.get("t_s"), abs_tol=1e-8)):
            raise DataError(f"Metric frame timestamp modified: {name}")
        src_pose = rigid_matrix(base["camera_to_sfm_world"])
        dst_pose = rigid_matrix(entry["camera_to_metric_sfm_world"])
        if (not np.allclose(src_pose[:3, :3], dst_pose[:3, :3], atol=1e-7, rtol=0)
                or not np.allclose(src_pose[:3, 3] * scale, dst_pose[:3, 3],
                                   atol=1e-6, rtol=1e-7)):
            raise DataError(f"Metric pose not a pure sensor-scale transform: {name}")
        delta = item.get("timestamp_delta_seconds")
        if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or abs(delta) > 0.03:
            raise DataError(f"RGB/depth timestamp exceeds 30ms: {name}")
        depth_rel = item.get("depth")
        if not isinstance(depth_rel, str) or depth_rel in seen_depths:
            raise DataError("Depth filename missing or reused")
        seen_depths.add(depth_rel)
        image_path = safe_path(image_root, name)
        depth_path = safe_path(depth_root, depth_rel)
        with Image.open(image_path) as color:
            if color.size != (cams[camera_id].width, cams[camera_id].height):
                raise DataError(f"RGB resolution differs from SfM model: {name}")
        # The calibration sensor's depth Z is real metres; do not infer it from RGB.
        measured = read_measured_depth(depth_path, float(units),
                                       (cams[camera_id].height, cams[camera_id].width))
        prepared.append((float(t), name, dst_pose, image_path, depth_path, measured, float(delta)))

    if len(camera_ids) != 1:
        raise DataError("ECHO input requires one fixed camera model")
    camera = cams[next(iter(camera_ids))].scaled(downsample)
    prepared.sort(key=lambda row: row[0])
    if any(b[0] <= a[0] for a, b in zip(prepared, prepared[1:])):
        raise DataError("Training frame timestamps must strictly increase")
    first_t = prepared[0][0]
    with output_transaction(output) as target:
        (target / "color").mkdir()
        (target / "depth").mkdir()
        entries, receipts = [], []
        for i, (t, name, pose, rgb_path, depth_path, measured, delta) in enumerate(prepared):
            rgb = target / "color" / f"{i:05d}.png"
            z = target / "depth" / f"{i:05d}.npy"
            with Image.open(rgb_path) as im:
                sampled_rgb = np.asarray(im.convert("RGB"))[::downsample, ::downsample]
            Image.fromarray(sampled_rgb).save(rgb)
            np.save(z, measured[::downsample, ::downsample].astype(np.float32),
                    allow_pickle=False)
            entries.append({"t": t-first_t, "cameraToWorld": pose.flatten(order="F").tolist(),
                            "color": f"color/{i:05d}.png",
                            "depth": f"depth/{i:05d}.npy"})
            receipts.append({"name": name, "sensor_depth_path": str(depth_path.relative_to(depth_root.resolve())),
                             "sensor_depth_sha256": file_sha256(depth_path),
                             "source_rgb_sha256": file_sha256(rgb_path),
                             "rgb_depth_timestamp_delta_s": delta})
        write_json(target / "recording.json", {
            "version": "echo-dataset-1", "name": "sfm-with-real-sensor-depth",
            "source": "recorded", "units": "meters", "depth_scale": 1,
            "duration": max(entries[-1]["t"], .001), "intrinsics": camera.to_dict(),
            "frames": entries,
            "notes": "Estimated SfM poses scaled by independent measured depth; all dense Z measurements external. Not RGB-only."
        })
        write_json(target / "sensor-provenance.json", {
            "version": "echo-sensor-mesh-input-1",
            "metric_trajectory_sha256": file_sha256(metric_path),
            "sfm_trajectory_sha256": file_sha256(sfm_path),
            "depth_split_sha256": file_sha256(split_path),
            "colmap_cameras_sha256": file_sha256(colmap_dir / "cameras.txt"),
            "sensor_scale_m_per_sfm_unit": float(scale),
            "sensor_claim_as_supplied": split["source_description"],
            "train": receipts, "holdout_names_not_loaded": withheld,
            "limitations": ["Holdout depth not read", "No external geometry ground truth",
                            "Only fixed PINHOLE intrinsics and registered camera-Z depth",
                            "External measured scale, not monocular RGB metrification"]
        })
        recording = load_recording(target / "recording.json")
        for _ in recording.iter_frames():
            pass
    return {"output": str(output.resolve()), "training_frames": len(entries),
            "heldout_frames": len(withheld), "sensor_scale_m_per_sfm_unit": float(scale)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("metric-trajectory", "sfm-trajectory", "colmap-model", "images",
                "depth-root", "split", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--downsample", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(convert_metric_sensor(
            args.metric_trajectory, args.sfm_trajectory, args.colmap_model,
            args.images, args.depth_root, args.split, args.output, args.downsample),
            ensure_ascii=False))
        return 0
    except (DataError, OSError, KeyError, TypeError, ValueError, MemoryError) as exc:
        import sys
        print(json.dumps({"error": str(exc), "type": type(exc).__name__}),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
