"""Measured-depth removal on the same real TUM Freiburg1 XYZ scene as #161.

Compare 48 / 24 / 12 / 6 / 3 *real sensor depth frames* at identical 320x240
calibration, reference camera poses, Open3D TSDF voxel size and five cameras.
The full RGB source remains available; RGB-only frames are NOT inferred here.
No old mesh is used as a reconstruction prior. Full-room status remains FAIL.
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
from tum_public_roundtrip import ROOT, URL, download, extract, invoke
from tum_xyz_open3d_320 import (
    FRAME_COUNT, TRAIN_INDICES, RECORDED_INDICES,
    export_glb, free_view, fuse, panel, render,
)

VOXEL_M = .018
DEPTH_BUDGETS = (48, 24, 12, 6, 3)


def diverse_pose_order(entries, candidate_indices, *, orientation_m=.18):
    """Deterministic max-min sampling of reference poses; no sensor depth read.

    Distances combine camera-center separation in metres and optical-axis
    angular separation in radians, converted to metres. This is a simple
    sampling policy, not a claim that reference poses are RGB-estimated.
    """
    indices = tuple(candidate_indices)
    if len(indices) != len(set(indices)) or not indices:
        raise ValueError("Expected nonempty unique candidate indices")
    if orientation_m < 0:
        raise ValueError("Orientation weight must be nonnegative")
    poses = [np.asarray(entries[i]["cameraToWorld"], dtype=np.float64)
             .reshape(4, 4, order="F") for i in indices]
    centers = np.array([p[:3, 3] for p in poses])
    axes = np.array([p[:3, 2] / np.linalg.norm(p[:3, 2]) for p in poses])
    if not np.isfinite(centers).all() or not np.isfinite(axes).all():
        raise ValueError("Nonfinite camera poses")
    translation = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    angles = np.arccos(np.clip(axes @ axes.T, -1, 1))
    distance = translation + orientation_m * angles
    # Start with the most central real camera, then expand to diverse views.
    seed = int(np.argmin(np.sum(distance, axis=1)))
    selected = [seed]
    nearest = distance[seed].copy()
    nearest[seed] = -np.inf
    while len(selected) < len(indices):
        nxt = int(np.argmax(nearest))
        selected.append(nxt)
        nearest = np.minimum(nearest, distance[nxt])
        nearest[selected] = -np.inf
    return tuple(indices[j] for j in selected)


def make_recording(args, receipt):
    args.work.mkdir(parents=True, exist_ok=False)
    archive = args.work / "source.tgz"
    download(archive, receipt)
    extract(archive, args.work / "dataset")
    imported = args.work / "imported-320"
    invoke("import-TUM-320x240-60", [
        sys.executable, "-m", "echo4d", "import-tum",
        str(args.work / "dataset" / ROOT), "--output", str(imported),
        "--intrinsics", "525", "525", "319.5", "239.5",
        "--stride", "12", "--limit", "60", "--downsample", "2",
        "--world-up", "z",
    ], receipt, timeout=900)
    record = load_recording(imported / "recording.json")
    if len(record) != FRAME_COUNT or (record.camera.width, record.camera.height) != (320, 240):
        raise DataError("Expected 60 calibrated 320x240 TUM frames")
    return record, imported


def experiment(args, receipt):
    import open3d as o3d
    import trimesh

    record, imported = make_recording(args, receipt)
    all_entries = record.entries
    order = diverse_pose_order(all_entries, TRAIN_INDICES)
    chosen = {budget: tuple(sorted(order[:budget])) for budget in DEPTH_BUDGETS}
    assert all(set(chosen[small]).issubset(chosen[large])
               for small, large in ((3, 6), (6, 12), (12, 24), (24, 48)))
    assert set(chosen[48]) == set(TRAIN_INDICES)
    receipt["sensor_depth_indices"] = {str(k): list(v) for k, v in chosen.items()}
    receipt["depth_selection"] = "reference-pose farthest-point nested; 0.18 m/radian"
    receipt["untouched_holdout_indices"] = list(range(0, FRAME_COUNT, 5))
    receipt["rgb_source_count"] = FRAME_COUNT
    receipt["camera_pose_source"] = "TUM provided reference poses, NOT RGB SfM"
    receipt["mesh_prior"] = False
    receipt["learned_depth"] = False
    receipt["metric_scale"] = "TUM measured sensor depths + provided metric poses"

    meshes = {}
    args.output.mkdir(parents=True, exist_ok=True)
    for budget in DEPTH_BUDGETS:
        record.entries = [all_entries[i] for i in chosen[budget]]
        frames = [frame for _, frame, _ in record.iter_frames()]
        if len(frames) != budget:
            raise DataError("Missing measured depth frames")
        mesh, _ = fuse(o3d, frames, record.camera, pixel_stride=1, voxel=VOXEL_M)
        glb = args.output / f"tum-measured-{budget}.glb"
        export_glb(mesh, glb)
        # Reopen actual exported geometry; the screenshots render the same mesh.
        reopened = trimesh.load(str(glb), force="mesh", process=False)
        if len(reopened.faces) < 100 or not reopened.visual.kind == "vertex":
            raise RuntimeError("Exported GLB lacks usable colored geometry")
        meshes[budget] = mesh
        receipt.setdefault("meshes", {})[str(budget)] = {
            "glb": glb.name, "triangles": len(mesh.triangles),
            "vertices": len(mesh.vertices), "input_measured_depths": budget,
            "actual_reopened_faces": len(reopened.faces),
        }
        print(f"REAL TUM {budget} measured depth frames -> {len(mesh.triangles)} triangles", flush=True)
        del frames, reopened
        gc.collect()
    record.entries = all_entries
    association = json.loads((imported / "association.json").read_text())
    source_root = args.work / "dataset" / ROOT
    rows = []
    views = []
    for index in RECORDED_INDICES:
        pose = np.asarray(all_entries[index]["cameraToWorld"]).reshape(4, 4, order="F")
        with Image.open(source_root / association["frames"][index]["rgb_path"]) as im:
            original = im.convert("RGB").copy()
        original.save(args.output / f"real-original-{index:02d}-640x480.png")
        rgb = np.asarray(original.resize((320, 240), Image.Resampling.LANCZOS))
        rendered = [render(o3d, meshes[b], record.camera, pose) for b in DEPTH_BUDGETS]
        name = f"recorded-{index:02d}-48-24-12-6-3.png"
        rows.append(panel(args.output / name, [rgb] + rendered,
                          ["REAL RGB 640 source"] + [f"{b} sensor" for b in DEPTH_BUDGETS]))
        views.append({"type": "recorded", "index": index, "file": name,
                      "depth_frame_is_training": {str(b): index in chosen[b] for b in DEPTH_BUDGETS}})
        gc.collect()
    for side, index in ((-1., 19), (1., 39)):
        pose = free_view(np.asarray(all_entries[index]["cameraToWorld"]).reshape(
            4, 4, order="F"), side)
        rendered = [render(o3d, meshes[b], record.camera, pose) for b in DEPTH_BUDGETS]
        name = f"free-{'left' if side < 0 else 'right'}-48-24-12-6-3.png"
        rows.append(panel(args.output / name, [np.full_like(rendered[0], 20)] + rendered,
                          ["FREE VIEW (no photo)"] + [f"{b} sensor" for b in DEPTH_BUDGETS]))
        views.append({"type": "free", "index": index, "file": name,
                      "lateral_offset_m": .22, "yaw_deg": 12 * side})
        gc.collect()
    board = Image.new("RGB", (320 * (1 + len(DEPTH_BUDGETS)), 267 * len(rows)))
    for i, row in enumerate(rows):
        board.paste(row, (0, i * 267))
    board.save(args.output / "tum-48-24-12-6-3-five-views.png")
    receipt["views"] = views
    receipt["visual_verdict"] = "PENDING actual human inspection of 3+2 images"
    receipt["full_room"] = "FAIL until continuous scene and free traversal validated"
    receipt["issue_74"] = "FAIL unchanged"
    receipt["historical_desk"] = "FAIL 16.7779 cm > 15 cm unchanged"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    receipt = {
        "scene": "TUM Freiburg1 XYZ real RGB-D desk/office",
        "source": URL, "status": "running",
        "reference": "high-information measured-depth baseline PR #161",
        "run_url": (f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/"
                    f"{os.getenv('GITHUB_RUN_ID')}") if os.getenv("GITHUB_RUN_ID") else None,
        "commands": [],
    }
    try:
        experiment(args, receipt)
        receipt["status"] = "real_meshes_and_five_views_exported_visual_review_pending"
        code = 0
    except Exception as exc:
        traceback.print_exc()
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        code = 1
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "result.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"status": receipt["status"], "meshes": receipt.get("meshes"),
                          "error": receipt.get("error")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
