"""Second real scene: TUM Freiburg1 Desk, measured RGB-D, reference poses.

Compare the SAME moving-camera measured frames with 160x120 coarse TSDF and
320x240 fine TSDF. Neither mesh is a scan prior or an RGB-only estimate.
Historical desk metric failure belongs to a different frozen protocol.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
from PIL import Image

from echo4d.model import DataError, load_recording
from tum_desk_sfm import URL, ROOT, get_archive, extract_selected
from tum_desk_full_mesh import extract_sensor_depths
from tum_public_roundtrip import invoke
from tum_xyz_open3d_320 import fuse, export_glb, render, panel, free_view

# 8-frame raw depth stride spans the desk sequence's changing camera angles.
MAX_IMPORTED = 72
RECORDED = (11, 31, 53)
FREE = ((-1., 19), (1., 43))
# Freiburg1 RGB intrinsics. The existing import-tum adapter scales these by 2.
# Its pinhole model does not correct lens distortion: a known limitation.
FR1_INTRINSICS = (517.3, 516.5, 318.6, 255.3)


def split_indices(count: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Hold out every fifth imported view; coarse and fine share ALL train frames."""
    if count < 54 or count > MAX_IMPORTED:
        raise DataError("Desk needs at least 54 synchronized camera/depth pairs")
    train = tuple(i for i in range(count) if i % 5 != 0)
    holdout = tuple(i for i in range(count) if i % 5 == 0)
    if not set(RECORDED).issubset(train) or any(i >= count for _, i in FREE):
        raise DataError("Fixed desk comparison cameras unavailable")
    return train, holdout


def experiment(args, result):
    import open3d as o3d

    args.work.mkdir(parents=True, exist_ok=False)
    archive = args.work / "desk-source.tgz"
    get_archive(archive, result, offline=args.archive)
    root = args.work / "dataset"
    result["rgb_and_pose_extraction"] = extract_selected(archive, root)
    result["sensor_extraction"] = extract_sensor_depths(archive, root)

    imported = args.work / "imported-desk-320"
    invoke("import-TUM-desk-known-pose-RGBD", [
        sys.executable, "-m", "echo4d", "import-tum",
        str(root / ROOT), "--output", str(imported),
        "--intrinsics", *map(str, FR1_INTRINSICS),
        "--stride", "8", "--limit", str(MAX_IMPORTED),
        "--downsample", "2", "--world-up", "z",
    ], result, timeout=1200)
    record = load_recording(imported / "recording.json")
    if (record.camera.width, record.camera.height) != (320, 240):
        raise DataError("Desk RGB-D calibration is not 320x240")
    entries = list(record.entries)
    train, holdout = split_indices(len(entries))
    record.entries = [entries[i] for i in train]
    frames = [frame for _, frame, _ in record.iter_frames()]
    if len(frames) != len(train):
        raise DataError("Desk sensor training frame count changed")

    result["imported_views"] = len(entries)
    result["training_indices"] = list(train)
    result["holdout_indices"] = list(holdout)
    result["intrinsics_320"] = record.camera.to_dict()
    result["lens_distortion_correction"] = False
    result["geometry_source"] = "real Freiburg1 Desk RGB + sensor depth + TUM provided reference poses; no prior Mesh"

    # A single change from the 160 baseline: calibrated image resolution and
    # voxel size. Keep the measured frame set and known poses identical.
    coarse, _ = fuse(o3d, frames, record.camera, pixel_stride=2, voxel=.05)
    fine, _ = fuse(o3d, frames, record.camera, pixel_stride=1, voxel=.018)
    args.output.mkdir(parents=True, exist_ok=True)
    export_glb(coarse, args.output / "desk-measured-coarse-160.glb")
    export_glb(fine, args.output / "desk-measured-fine-320.glb")
    result["meshes"] = {
        "coarse_160": {"faces": len(coarse.triangles), "file": "desk-measured-coarse-160.glb"},
        "fine_320": {"faces": len(fine.triangles), "file": "desk-measured-fine-320.glb"},
    }

    association = json.loads((imported / "association.json").read_text())
    boards = []
    result["views"] = []
    for index in RECORDED:
        pose = np.asarray(entries[index]["cameraToWorld"]).reshape(4, 4, order="F")
        rgb_path = root / ROOT / association["frames"][index]["rgb_path"]
        with Image.open(rgb_path) as image:
            original = image.convert("RGB")
            original.save(args.output / f"real-desk-{index:02d}-640x480.png")
            reference = np.asarray(original.resize((320, 240), Image.Resampling.LANCZOS))
        name = f"recorded-{index:02d}-coarse-vs-fine.png"
        boards.append(panel(args.output / name, [
            reference, render(o3d, coarse, record.camera, pose),
            render(o3d, fine, record.camera, pose),
        ], ["REAL 640x480 RGB", "160 coarse TSDF", "320 fine TSDF"]))
        result["views"].append({"kind": "recorded", "index": index, "file": name})
        gc.collect()

    for side, index in FREE:
        pose = free_view(
            np.asarray(entries[index]["cameraToWorld"]).reshape(4, 4, order="F"), side)
        old = render(o3d, coarse, record.camera, pose)
        new = render(o3d, fine, record.camera, pose)
        name = f"free-{'left' if side < 0 else 'right'}-coarse-vs-fine.png"
        boards.append(panel(args.output / name, [
            np.full_like(new, 20), old, new,
        ], ["FREE / no real photo", "160 coarse TSDF", "320 fine TSDF"]))
        result["views"].append({"kind": "free", "index": index, "file": name,
                                "shift_m": .22, "yaw_deg": 12 * side})
        gc.collect()

    combined = Image.new("RGB", (960, 267 * len(boards)))
    for i, board in enumerate(boards):
        combined.paste(board, (0, 267 * i))
    combined.save(args.output / "desk-five-view-real-160-vs-320.png")
    result["human_visual_verdict"] = "PENDING; triangle count is not a visual verdict"
    result["free_orbit"] = "NOT ACCEPTED"
    result["issue_74"] = "FAIL unchanged"
    result["historical_desk_v2"] = "FAIL 16.7779 cm > 15 cm unchanged"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    result = {"scene": "TUM Freiburg1 Desk, different from Freiburg1 XYZ",
              "source": URL, "status": "running", "commands": [],
              "run_url": (f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/"
                          f"{os.getenv('GITHUB_RUN_ID')}" if os.getenv("GITHUB_RUN_ID") else None)}
    try:
        experiment(args, result)
        result["status"] = "real_mesh_and_five_views_exported_review_pending"
        code = 0
    except Exception as exc:
        traceback.print_exc()
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        code = 1
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"status": result["status"], "meshes": result.get("meshes"),
                          "error": result.get("error")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
