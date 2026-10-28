"""Post-hoc measured sensor-Z check of an ALREADY generated observed-edge Mesh.

No measured depth is read until both old/new RGB-inferred Mesh files exist.
The heldout sensor depth is evaluation-only and never affects reconstruction.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from echo4d.holdout import read_echo_ply, rasterize_mesh_depth
from echo4d.model import file_sha256
from echo4d.tum import read_rows
from tum_public_roundtrip import ROOT, PINNED_ARCHIVE_SHA256
from tum_rgb_stereo import rgb_pose_frames, stereo_resolution
from tum_plane_sweep_depth_diagnostic import score_pair, sensor_depth, error_image


def check(old: Path, new: Path, source_root: Path, output: Path) -> dict:
    if not (old / "reconstruction/mesh.ply").is_file() or not (
            new / "reconstruction/mesh.ply").is_file():
        raise FileNotFoundError("Both real Mesh files must exist before sensor Z opens")
    source = source_root / ROOT
    frames = rgb_pose_frames(source)
    rows = read_rows(source / "depth.txt", 2)
    times = np.array([float(r[0]) for r in rows], dtype=np.float64)
    camera, _, _ = stereo_resolution(320)
    meshes = [read_echo_ply(p / "reconstruction/mesh.ply") for p in (old, new)]
    output.mkdir(parents=True, exist_ok=False)
    scores, images = [], []
    for index in (4, 24, 55):
        pose = frames[index]["pose"]
        z0, _ = rasterize_mesh_depth(*meshes[0], camera, pose, far=4)
        z1, _ = rasterize_mesh_depth(*meshes[1], camera, pose, far=4)
        measured, source_info = sensor_depth(source, frames[index]["t_s"], rows, times)
        scores.append({"heldout_index": index, "sensor_source": source_info,
                       "comparison": score_pair(z0, z1, measured)})
        with Image.open(frames[index]["rgb"]) as photo:
            original = photo.convert("RGB").resize((320, 240), Image.Resampling.BILINEAR)
        images.append([original, error_image(z0, z1, measured)])
    canvas = Image.new("RGB", (640, 3 * 270 + 35), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 8), "Original RGB | new-only: green <=10cm, amber <=30cm, red >30cm, blue lost", fill="black")
    for i, pair in enumerate(images):
        y = 35 + i * 270
        for j, im in enumerate(pair):
            canvas.paste(im, (320 * j, y))
        draw.text((5, y + 242), f"heldout {scores[i]['heldout_index']}: sensor Z is diagnostic ONLY", fill="black")
    canvas.save(output / "heldout-sensor-error.png")
    gained = sum(s["comparison"]["new_only_sensor_agreement"]["pixels"] for s in scores)
    wrong = sum(s["comparison"]["new_only_sensor_agreement"]["pixels"] *
                s["comparison"]["new_only_sensor_agreement"]["over_30cm_fraction"]
                for s in scores if s["comparison"]["new_only_sensor_agreement"]["pixels"])
    lost = sum(s["comparison"]["lost_sensor_covered_pixels"] for s in scores)
    result = {
        "status": "posthoc_only_visual_acceptance_still_FAIL",
        "source_archive_sha256": PINNED_ARCHIVE_SHA256,
        "old_mesh_sha256": file_sha256(old / "reconstruction/mesh.ply"),
        "new_mesh_sha256": file_sha256(new / "reconstruction/mesh.ply"),
        "sensor_depth_used_for_reconstruction": False,
        "heldout": scores,
        "gained_sensor_valid_pixels": gained,
        "lost_sensor_valid_pixels": lost,
        "new_only_over_30cm_fraction": wrong / gained if gained else None,
        "error_image_sha256": file_sha256(output / "heldout-sensor-error.png"),
        "warning": "Sensor-invalid regions are unknown; sensor agreement is not independent scanned-surface truth.",
    }
    (output / "diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
    return {"gained": gained, "lost": lost,
            "new_only_over_30cm_fraction": result["new_only_over_30cm_fraction"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", required=True, type=Path)
    p.add_argument("--new", required=True, type=Path)
    p.add_argument("--source-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(check(a.old, a.new, a.source_root, a.output)))
