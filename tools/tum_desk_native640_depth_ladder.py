"""Real TUM Freiburg1 Desk: remove measured depth observations at fixed native resolution.

High-information control is the exact 640x480 / 12mm Open3D fusion from
tum_desk_native640_tsdf.py. Nested 24- and 12-frame subsets retain genuine
measured RGB-D and TUM reference poses, not RGB-inferred depth or a scan prior.
The other RGB photos are retained as source material, NOT fused as geometry.
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

from echo4d.model import Camera, DataError, load_recording
from tum_desk_sfm import URL, ROOT, get_archive, extract_selected
from tum_desk_full_mesh import extract_sensor_depths
from tum_public_roundtrip import invoke
from tum_desk_known_pose_open3d import (
    MAX_IMPORTED, RECORDED, FREE, FR1_INTRINSICS, split_indices,
)
from tum_desk_native640_tsdf import NATIVE_VOXEL_M, MAX_TRIANGLES, recorded_poses
from tum_xyz_open3d_320 import fuse, export_glb, render, panel, free_view


def nested_measured_indices(training: tuple[int, ...], budgets=(12, 24)) -> dict[int, tuple[int, ...]]:
    """Select temporally dispersed *nested* measured-depth subsets.

    The full training set and held-out depth frames must be determined BEFORE
    this function. No evaluation depth or generated depth is consulted.
    """
    if len(training) != len(set(training)) or tuple(sorted(training)) != training:
        raise ValueError("Training indices must be sorted and unique")
    if not budgets or tuple(sorted(set(budgets))) != tuple(budgets):
        raise ValueError("Budgets must be strictly increasing")
    if budgets[0] < 2 or budgets[-1] > len(training):
        raise ValueError("Invalid measured-depth budget")
    first = budgets[0]
    selected = {training[int(round(i * (len(training) - 1) / (first - 1)))]
                for i in range(first)}
    if len(selected) != first:
        raise ValueError("Uniform initial subset collided")
    result = {first: tuple(sorted(selected))}
    for budget in budgets[1:]:
        while len(selected) < budget:
            candidate = max((i for i in training if i not in selected),
                            key=lambda i: (min(abs(i - j) for j in selected), -i))
            selected.add(candidate)
        result[budget] = tuple(sorted(selected))
    return result


def experiment(args, receipt):
    import open3d as o3d

    args.work.mkdir(parents=True, exist_ok=False)
    archive = args.work / "desk-source.tgz"
    get_archive(archive, receipt, offline=args.archive)
    root = args.work / "dataset"
    receipt["rgb_pose_extraction"] = extract_selected(archive, root)
    receipt["measured_depth_extraction"] = extract_sensor_depths(archive, root)
    imported = args.work / "imported-native"
    invoke("import-TUM-Desk-640-measured-depth-ladder", [
        sys.executable, "-m", "echo4d", "import-tum", str(root / ROOT),
        "--output", str(imported), "--intrinsics", *map(str, FR1_INTRINSICS),
        "--stride", "8", "--limit", str(MAX_IMPORTED),
        "--downsample", "1", "--world-up", "z",
    ], receipt, timeout=1200)

    recording = load_recording(imported / "recording.json")
    camera = recording.camera
    if camera.to_dict() != Camera(640, 480, *FR1_INTRINSICS).to_dict():
        raise DataError("Native measured camera calibration differs from baseline")
    entries = list(recording.entries)
    train, holdout = split_indices(len(entries))
    subsets = nested_measured_indices(train)
    poses = recorded_poses(entries)
    recording.entries = [entries[i] for i in train]
    measured = [frame for _, frame, _ in recording.iter_frames()]
    if len(measured) != len(train):
        raise DataError("Training RGB-D frame count changed")
    by_index = dict(zip(train, measured))
    specs = [
        ("full-measured", train),
        ("measured-24", subsets[24]),
        ("measured-12", subsets[12]),
    ]
    receipt.update({
        "source_scene": "TUM Freiburg1 Desk",
        "geometry_source": "native registered 640x480 RGB + measured depth + TUM reference poses",
        "no_mesh_prior": True,
        "no_inferred_depth": True,
        "depth_removed": "Full measured training set reduced to nested 24 and 12 sensor depth frames",
        "unused_rgb": "Other source RGB frames are available but do not contribute geometry",
        "camera_calibration": camera.to_dict(),
        "reference_poses": "TUM provided, not estimated",
        "metric_scale": "Measured TUM sensor depths; meters",
        "lens_rectification": "off, identical to high-information native 640 baseline",
        "holdout_indices": list(holdout),
        "training_indices": list(train),
        "subsets": {str(k): list(v) for k, v in subsets.items()},
        "voxel_m": NATIVE_VOXEL_M,
        "meshes": {},
        "views": [],
    })
    args.output.mkdir(parents=True, exist_ok=True)
    render_camera = camera.scaled(2)
    rendered = {}

    # Process and release one mesh at a time; avoid holding three 640 meshes in RAM.
    for name, indices in specs:
        mesh, _ = fuse(o3d, [by_index[i] for i in indices], camera,
                       pixel_stride=1, voxel=NATIVE_VOXEL_M)
        faces = len(mesh.triangles)
        if not 100 <= faces <= MAX_TRIANGLES:
            raise DataError(f"{name}: invalid mesh face count {faces}")
        glb = f"desk-{name}-native640.glb"
        export_glb(mesh, args.output / glb)
        receipt["meshes"][name] = {
            "glb": glb, "measured_depth_frames": len(indices),
            "source_indices": list(indices), "triangles": faces,
            "vertices": len(mesh.vertices), "input_resolution": [640, 480],
            "voxel_m": NATIVE_VOXEL_M,
        }
        images = [render(o3d, mesh, render_camera, poses[i]) for i in RECORDED]
        images.extend(render(o3d, mesh, render_camera,
                             free_view(poses[i], side)) for side, i in FREE)
        rendered[name] = images
        print(f"{name}: exported real GLB and 5 matching camera views", flush=True)
        del mesh
        gc.collect()

    association = json.loads((imported / "association.json").read_text())
    boards = []
    names = [name for name, _ in specs]
    labels = ["FULL MEASURED", "24 MEASURED", "12 MEASURED"]
    for row, index in enumerate(RECORDED):
        path = root / ROOT / association["frames"][index]["rgb_path"]
        with Image.open(path) as image:
            original = image.convert("RGB")
            original.save(args.output / f"real-desk-{index:02d}-640x480.png")
            photo = np.asarray(original.resize((320, 240), Image.Resampling.LANCZOS))
        filename = f"recorded-{index:02d}-full-24-12.png"
        board = panel(args.output / filename,
                      [photo] + [rendered[n][row] for n in names],
                      ["REAL SOURCE RGB"] + labels)
        boards.append(board)
        receipt["views"].append({"kind": "recorded", "index": index, "file": filename})
    for offset, (side, index) in enumerate(FREE):
        row = len(RECORDED) + offset
        images = [rendered[n][row] for n in names]
        filename = f"free-{'left' if side < 0 else 'right'}-full-24-12.png"
        boards.append(panel(args.output / filename,
                            [np.full_like(images[0], 20)] + images,
                            ["FREE VIEW / NO RGB"] + labels))
        receipt["views"].append({
            "kind": "free", "reference_index": index, "file": filename,
            "shift_m": .22, "yaw_deg": 12 * side,
        })
    combined = Image.new("RGB", (1280, 267 * len(boards)))
    for i, board in enumerate(boards):
        combined.paste(board, (0, 267 * i))
    combined.save(args.output / "desk-native640-full-vs-24-vs-12-five-views.png")
    receipt["visual_verdict"] = "PENDING independent inspection of actual five views"
    receipt["full_room"] = "FAIL until continuous walls, floor and furniture verified"
    receipt["issue_74"] = "FAIL unchanged"
    receipt["historical_desk_v2"] = "FAIL 16.7779cm > 15cm unchanged"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    receipt = {
        "source": URL, "scene": "TUM Freiburg1 Desk", "status": "running",
        "commands": [],
        "run_url": (f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/"
                    f"{os.getenv('GITHUB_RUN_ID')}" if os.getenv("GITHUB_RUN_ID") else None),
    }
    try:
        experiment(args, receipt)
        receipt["status"] = "real_meshes_exported_visual_review_pending"
        code = 0
    except Exception as exc:
        traceback.print_exc()
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        code = 1
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "result.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"status": receipt["status"],
                          "meshes": receipt.get("meshes"),
                          "error": receipt.get("error")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
