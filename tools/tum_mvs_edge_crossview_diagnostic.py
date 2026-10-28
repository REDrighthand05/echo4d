"""Posthoc heldout sensor-Z error for removed vs retained observed-edge faces.

Only called AFTER both real Meshes exist. Sensor depth cannot affect selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from echo4d.holdout import read_echo_ply, rasterize_mesh_depth
from echo4d.model import Camera, file_sha256
from echo4d.tum import read_rows
from tum_public_roundtrip import ROOT
from tum_rgb_stereo import rgb_pose_frames
from tum_plane_sweep_depth_diagnostic import sensor_depth


def check(old: Path, new: Path, source_root: Path, output: Path) -> dict:
    old_mesh = old / "reconstruction/mesh.ply"
    new_mesh = new / "reconstruction/mesh.ply"
    if not old_mesh.is_file() or not new_mesh.is_file():
        raise FileNotFoundError("Both RGB-only-inferred Meshes must exist first")
    root = source_root / ROOT
    frames = rgb_pose_frames(root)
    rows = read_rows(root / "depth.txt", 2)
    times = np.array([float(r[0]) for r in rows], dtype=np.float64)
    camera = Camera(320, 240, 262.5, 262.5, 159.75, 119.75)
    meshes = [read_echo_ply(old_mesh), read_echo_ply(new_mesh)]
    output.mkdir(parents=True, exist_ok=False)
    details = []
    for index in (4, 24, 55):
        pose = frames[index]["pose"]
        old_z, _ = rasterize_mesh_depth(*meshes[0], camera, pose, far=4)
        new_z, _ = rasterize_mesh_depth(*meshes[1], camera, pose, far=4)
        sensor, info = sensor_depth(root, frames[index]["t_s"], rows, times)
        sensor_valid = (sensor >= .15) & (sensor <= 4)
        old_valid = (old_z >= .15) & (old_z <= 4)
        new_valid = (new_z >= .15) & (new_z <= 4)
        removed = sensor_valid & old_valid & ~new_valid
        retained = sensor_valid & old_valid & new_valid
        new_only = sensor_valid & ~old_valid & new_valid
        def metrics(mask, predicted):
            if not np.any(mask):
                return {"pixels": 0, "mae_m": None, "over_30cm_fraction": None}
            err = np.abs(predicted[mask].astype(np.float64) -
                         sensor[mask].astype(np.float64))
            return {"pixels": int(mask.sum()), "mae_m": float(err.mean()),
                    "over_30cm_fraction": float(np.mean(err > .30))}
        details.append({
            "heldout_index": index,
            "sensor": info,
            "old_covered": int(old_valid.sum()),
            "new_covered": int(new_valid.sum()),
            "removed": metrics(removed, old_z),
            "retained_old": metrics(retained, old_z),
            "retained_new": metrics(retained, new_z),
            "new_only": metrics(new_only, new_z),
        })
        # Red=removed old surface >30cm wrong, green=removed old <=30cm;
        # gray=still covered. Sensor-invalid remains black/unknown.
        pixels = np.zeros((*old_z.shape, 3), dtype=np.uint8)
        pixels[retained] = [120, 120, 120]
        pixels[removed & (np.abs(old_z - sensor) > .30)] = [240, 55, 55]
        pixels[removed & (np.abs(old_z - sensor) <= .30)] = [55, 190, 85]
        Image.fromarray(pixels).save(output / f"heldout-{index}-removed-errors.png")
    result = {
        "sensor_depth_used_for_reconstruction": False,
        "old_mesh_sha256": file_sha256(old_mesh),
        "new_mesh_sha256": file_sha256(new_mesh),
        "heldout": details,
        "visual_acceptance": "FAIL pending scene recognition",
        "warning": "Sensor-valid pixels are not independent scanned surface GT; sensor-invalid remains unknown.",
    }
    (output / "diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(check(a.old, a.new, a.source_root, a.output)))
