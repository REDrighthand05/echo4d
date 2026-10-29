"""Replay #105's 96 cached DA2 RGB depths, filter unsupported ghosts, rebuild Mesh.

No model inference or GPU. The 0-sensor candidate is fused before the inherited
ladder opens the measured sensor archive. The original #105 Mesh is A/B control.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from echo4d.model import Camera, DataError, file_sha256
import tum_room_96view_da2_complete as base
from tum_room_depth_support import filter_all
from tum_room_pose_overlap_neighbors import choose_overlap_neighbors
from tum_room_da2_global_scale_replay import checked_replay, five_view_comparison

OLD_ZERO_SHA256 = "eae89399da53cc34a3c61a85daada404a7959fda935ea3dc0b592a3090dc62cb"
_cached_predictions = Path(os.environ.get("ECHO_ROOM96_FROZEN_PREDICTIONS", ""))
_old_overlap_artifact = Path(os.environ.get("ECHO_ROOM96_OLD_OVERLAP", ""))


def replay_and_filter(replay: Path, selected: dict, source_ids: list[int],
                      frozen_predicted: dict[int, dict], output: Path):
    """Read only cached predicted Z and source RGB; no sensor depth available."""
    root = _cached_predictions.resolve()
    manifest_path = root / "recording.json"
    if not manifest_path.is_file() or output.exists():
        raise DataError("Frozen prediction manifest missing or output already exists")
    original = json.loads(manifest_path.read_text())
    if len(source_ids) != 96 or len(original.get("frames", [])) != 96:
        raise DataError("Not the fixed 96-view cohort")
    camera = Camera.from_dict(selected["intrinsics"]).scaled(2)
    if original["intrinsics"] != camera.to_dict():
        raise DataError("Predicted RGB camera calibration changed")
    assoc = json.loads((replay / "source/association.json").read_text())
    poses, depths = [], []
    for i, (idx, frame, source_frame) in enumerate(zip(
            source_ids, original["frames"], selected["frames"])):
        if frame.get("source_anchor_index") != idx:
            raise DataError(f"Frozen prediction frame identity changed at {i}")
        a = np.asarray(frame["cameraToWorld"], dtype=np.float64)
        b = np.asarray(source_frame["cameraToWorld"], dtype=np.float64)
        if a.shape != (16,) or b.shape != (16,) or not np.allclose(a, b, atol=1e-7):
            raise DataError(f"Frozen metric pose changed at {i}")
        pose = a.reshape((4, 4), order="F")
        depth_file = (root / frame["depth"]).resolve()
        color_file = (root / frame["color"]).resolve()
        if (not depth_file.is_relative_to(root) or not color_file.is_relative_to(root)
                or not depth_file.is_file() or not color_file.is_file()):
            raise DataError("Invalid frozen relative paths")
        z = np.load(depth_file, allow_pickle=False)
        if (z.shape != (240, 320) or z.dtype != np.float32
                or not np.isfinite(z).all() or (z < 0).any()):
            raise DataError(f"Frozen predicted metric Z invalid at {i}")
        original_rgb = replay / "source" / assoc["frames"][idx]["rgb_path"]
        if file_sha256(original_rgb) != assoc["frames"][idx]["rgb_sha256"]:
            raise DataError("Original 640x480 RGB hash mismatch")
        with Image.open(original_rgb) as im:
            if im.size != (640, 480):
                raise DataError("Original RGB dimensions changed")
            expected = np.asarray(im.convert("RGB").resize(
                (320, 240), Image.Resampling.BILINEAR))
        with Image.open(color_file) as im:
            observed = np.asarray(im.convert("RGB"))
        if not np.array_equal(expected, observed):
            raise DataError(f"Frozen RGB differs from original source at {i}")
        poses.append(pose)
        depths.append(z)
    K = np.array([[camera.fx, 0., camera.cx], [0., camera.fy, camera.cy],
                  [0., 0., 1.]], dtype=np.float64)
    filtered, receipts = filter_all(depths, poses, K, choose_overlap_neighbors)
    shutil.copytree(root, output)
    for frame, z in zip(original["frames"], filtered):
        np.save(output / frame["depth"], z, allow_pickle=False)
    record = json.loads((output / "recording.json").read_text())
    record["name"] += " / occlusion-aware cross-view ghost rejection"
    record["notes"] += ("; predicted Z filtered only by independent known-pose "
                        "RGB-predicted Z views; no measured sensor depth or prior Mesh")
    (output / "recording.json").write_text(json.dumps(
        record, indent=2, allow_nan=False) + "\n")
    removed = sum(r["removed"] for r in receipts)
    before = sum(r["valid_before"] for r in receipts)
    report = {
        "source_run": 40826766484,
        "source_artifact": "tum-room-96-overlap-depth-40826766484",
        "source_predicted_manifest_sha256": file_sha256(manifest_path),
        "source_rgb_poses_verified": 96,
        "model_inference_executed_this_run": False,
        "new_cpu_inference_frames": 0,
        "predicted_nonempty_frames": sum(int(np.any(z > 0)) for z in filtered),
        "frozen_predicted_frames": 96,
        "cross_view_filter": {
            "before_valid_pixels": before,
            "after_valid_pixels": before - removed,
            "removed_pixels": removed,
            "removed_fraction": float(removed / before) if before else 0.,
            "algorithm": "occlusion-aware 2-independent-front-contradictions; original immutable DA2 maps",
            "absolute_tolerance_m": .07,
            "relative_tolerance": .045,
            "receipts": receipts,
            "historical_visual_fail": True,
        },
    }
    (output / "cross-view-filter-report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"FILTERED_REAL_PREDICTED_Z removed={removed}/{before}", flush=True)
    return output / "recording.json", report


def main():
    base.build_complete_rgb_prediction = replay_and_filter
    base.main()
    # Direct old-vs-new Mesh comparison, not only measured-vs-new.
    import sys
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--archive", type=Path)
    args = p.parse_args(sys.argv[1:])
    old = _old_overlap_artifact.resolve() / "mesh-sensor-0/mesh.ply"
    new = args.output.resolve() / "mesh-sensor-0/mesh.ply"
    if not old.is_file() or file_sha256(old) != OLD_ZERO_SHA256:
        raise DataError("Prior real 0-sensor Mesh missing or not exact #105 control")
    frozen, replay, _ = checked_replay(args.source.resolve())
    comparison = args.output.resolve() / "visual-old-vs-new"
    comparison.mkdir(exist_ok=False)
    views = five_view_comparison(args.source.resolve(), replay, old, new,
                                 comparison, frozen)
    report_path = args.output.resolve() / "report.json"
    report = json.loads(report_path.read_text())
    report["direct_old_new"] = {
        "old_mesh_sha256": OLD_ZERO_SHA256,
        "new_mesh_sha256": file_sha256(new),
        "five_views": views,
        "visual_sha256": file_sha256(comparison / "source-old-new-3plus2.png"),
        "visual_review": "UNREVIEWED / FAIL until actual source-photo and free-view inspection",
    }
    report["issue_74"] = "FAIL / NOT ACCEPTED"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
