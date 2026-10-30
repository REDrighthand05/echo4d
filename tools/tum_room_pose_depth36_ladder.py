"""TUM Room: real measured-depth reduction 145 -> 72 -> 36, same five cameras.

Uses the existing real RGB-D import, pose-based frame selector, Open3D fusion,
and renderer. Original TUM reference poses and metric sensor depths; no prior Mesh.
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
from tum_desk_known_pose_open3d import FR1_INTRINSICS
from tum_public_roundtrip import invoke
from tum_room_loop_coverage import ROOT, URL, MAX_FRAMES, VOXEL_M, MAX_FACES, download, extract_sampled, sparse_reference_indices, choose_views
from tum_xyz_open3d_320 import fuse, export_glb, render, panel, free_view
from tum_room_pose_depth_ladder import select_pose_spread

def run(args, receipt):
    import open3d as o3d
    import trimesh
    args.work.mkdir(parents=True, exist_ok=False)
    archive = args.work / "room-source.tgz"
    download(archive, receipt, args.archive)
    root = args.work / "dataset" / ROOT
    receipt["extraction"] = extract_sampled(archive, root)
    imported = args.work / "imported-room-320"
    invoke("import-real-TUM-room-320", [
        sys.executable, "-m", "echo4d", "import-tum", str(root),
        "--output", str(imported), "--intrinsics", *map(str, FR1_INTRINSICS),
        "--stride", "1", "--limit", str(MAX_FRAMES),
        "--downsample", "2", "--world-up", "z"
    ], receipt, timeout=1200)
    recording = load_recording(imported / "recording.json")
    camera = recording.camera
    if (camera.width, camera.height) != (320, 240):
        raise DataError("Expected real 320x240 calibrated Room depth")
    entries = list(recording.entries)
    train = tuple(i for i in range(len(entries)) if i % 7 != 0)
    holdout = tuple(i for i in range(len(entries)) if i % 7 == 0)
    reference = sparse_reference_indices(train)
    if len(reference) < 110:
        raise DataError("Prior 145-depth Room baseline is unavailable")
    poses = {
        i: np.asarray(entries[i]["cameraToWorld"], dtype=np.float64)
             .reshape((4, 4), order="F")
        for i in reference
    }
    pose72 = select_pose_spread(reference, poses, 72)
    pose36 = select_pose_spread(reference, poses, 36)
    if (not set(pose36).issubset(reference)
            or not set(pose72).issubset(reference)
            or set(holdout) & (set(pose36) | set(pose72))):
        raise DataError("Sensor depth selection leaked holdouts")
    recorded, free = choose_views(len(entries), reference)
    camera_poses = {
        i: np.asarray(entries[i]["cameraToWorld"], dtype=np.float64)
             .reshape((4, 4), order="F")
        for i in recorded + free
    }
    # Read only the 145-frame training reference sensor depths; both reduced
    # cohorts are strict subsets, and untouched holdouts are never loaded.
    recording.entries = [entries[i] for i in reference]
    reference_frames = [frame for _, frame, _ in recording.iter_frames()]
    if len(reference_frames) != len(reference):
        raise DataError("Missing measured baseline frames")
    frames = dict(zip(reference, reference_frames))
    args.output.mkdir(parents=True, exist_ok=True)
    receipt.update({
        "scene": "TUM Freiburg1 Room full camera loop",
        "source": URL,
        "camera": camera.to_dict(),
        "reference_poses": "provided by TUM, NOT RGB-estimated",
        "scale": "TUM sensor depth in metres and reference pose translations",
        "prior_mesh": False,
        "predicted_depth": False,
        "gpu": False,
        "voxel_m": VOXEL_M,
        "holdout_indices": list(holdout),
        "cohorts": {"reference145": list(reference),
                    "pose72": list(pose72),
                    "pose36": list(pose36)},
        "visual_verdict": "PENDING visual review; full-room FAIL unchanged; depth36 is new unverified candidate",
        "meshes": {},
        "views": [],
    })
    rendered = {}
    for name, cohort in (("reference145", reference),
                         ("pose72", pose72),
                         ("pose36", pose36)):
        mesh, _ = fuse(o3d, [frames[i] for i in cohort],
                       camera, pixel_stride=1, voxel=VOXEL_M)
        if len(mesh.triangles) > MAX_FACES:
            mesh = mesh.simplify_quadric_decimation(MAX_FACES)
            mesh.compute_vertex_normals()
        if not 1000 < len(mesh.triangles) <= MAX_FACES:
            raise DataError(name + ": invalid measured-depth Mesh")
        filename = "room-" + name + "-measured.glb"
        export_glb(mesh, args.output / filename)
        # Reopen actual GLB rather than treating file existence as success.
        loaded = trimesh.load(str(args.output / filename),
                              force="mesh", process=False)
        if len(loaded.faces) < 1000 or loaded.visual.kind != "vertex":
            raise DataError(name + ": exported GLB is not a colored Mesh")
        rendered[name] = [render(o3d, mesh, camera, camera_poses[i])
                          for i in recorded]
        rendered[name] += [
            render(o3d, mesh, camera, free_view(camera_poses[i], side))
            for side, i in zip((-1., 1.), free)
        ]
        receipt["meshes"][name] = {
            "glb": filename, "measured_depth_frames": len(cohort),
            "triangles": len(mesh.triangles),
            "prior_mesh": False
        }
        print(name, "actual triangles", len(mesh.triangles), flush=True)
        del mesh, loaded
        gc.collect()
    association = json.loads((imported / "association.json").read_text())
    panels = []
    for row, i in enumerate(recorded + free):
        if row < 3:
            with Image.open(root / association["frames"][i]["rgb_path"]) as im:
                im.convert("RGB").save(args.output / f"source-{i:03d}-640x480.png")
                source = np.asarray(im.convert("RGB").resize(
                    (320, 240), Image.Resampling.LANCZOS))
            kind = "recorded"
        else:
            source = np.full((240, 320, 3), 20, dtype=np.uint8)
            kind = "free"
        names = ("reference145", "pose72", "pose36")
        image_file = f"{kind}-{row+1}-145-vs-72-vs-36.png"
        panels.append(panel(args.output / image_file,
            [source] + [rendered[name][row] for name in names],
            [("REAL RGB" if kind == "recorded" else "FREE / no photo"),
             "145 MEASURED", "72 POSE", "36 POSE"]))
        receipt["views"].append({"kind": kind, "source_index": i,
                                 "file": image_file})
    board = Image.new("RGB", (1280, 267 * len(panels)))
    for row, im in enumerate(panels):
        board.paste(im, (0, row * 267))
    board.save(args.output / "room-depth145-vs-pose72-vs-pose36-five-views.png")
    receipt["historical_fail"] = "full room FAIL; #74 FAIL; Desk 16.7779cm >15cm FAIL"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--archive", type=Path)
    args = p.parse_args()
    receipt = {"status": "running", "commands": [],
               "run_url": (
                   "https://github.com/" + os.environ["GITHUB_REPOSITORY"] +
                   "/actions/runs/" + os.environ["GITHUB_RUN_ID"]
                   if os.getenv("GITHUB_RUN_ID") else None)}
    try:
        run(args, receipt)
        receipt["status"] = "real_glbs_and_five_views_exported_visual_review_pending"
        status = 0
    except Exception as exc:
        traceback.print_exc()
        receipt["status"] = "failed"
        receipt["error"] = type(exc).__name__ + ": " + str(exc)
        status = 1
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "result.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"status": receipt["status"],
                          "meshes": receipt.get("meshes"),
                          "error": receipt.get("error")}), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
