"""Real Freiburg1 Desk: native 640x480 measured RGB-D TSDF vs 320x240.

Same RGB/depth pairs, TUM reference poses, held-out views, and camera path.
320 baseline uses exact even-pixel samples from native 640; 640 uses 12 mm
voxels instead of 18 mm to test the practical high-detail reconstruction.
No prior Mesh, inferred depth, view synthesis, or post-hoc hole filling.
This is a *combined* resolution/voxel A/B, not a causal one-variable test.
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

from echo4d.model import Camera, DataError, Frame, load_recording
from tum_desk_sfm import URL, ROOT, get_archive, extract_selected
from tum_desk_full_mesh import extract_sensor_depths
from tum_public_roundtrip import invoke
from tum_desk_known_pose_open3d import (
    MAX_IMPORTED, RECORDED, FREE, FR1_INTRINSICS, split_indices,
)
from tum_xyz_open3d_320 import fuse, export_glb, render, panel, free_view

BASELINE_VOXEL_M = .018
NATIVE_VOXEL_M = .012
MAX_TRIANGLES = 2_500_000


def even_pixel_frame(frame: Frame, camera: Camera) -> Frame:
    """Downsample with exact pinhole rays: (u,v)=(2i,2j) at native resolution."""
    if frame.depth.shape != (camera.height, camera.width):
        raise DataError("Native frame and camera resolution differ")
    if frame.color is None or frame.color.shape != (*frame.depth.shape, 3):
        raise DataError("Missing registered measured RGB")
    mask = None if frame.excluded is None else np.ascontiguousarray(frame.excluded[::2, ::2])
    result = Frame(frame.t, frame.camera_to_world,
                   np.ascontiguousarray(frame.depth[::2, ::2]),
                   np.ascontiguousarray(frame.color[::2, ::2]), mask)
    return result.validate(camera.scaled(2))


def recorded_poses(entries: list[dict]) -> dict[int, np.ndarray]:
    return {i: np.asarray(entries[i]["cameraToWorld"], dtype=np.float64).reshape(
        4, 4, order="F") for i in set(RECORDED) | {index for _, index in FREE}}


def run(args, result):
    import open3d as o3d

    args.work.mkdir(parents=True, exist_ok=False)
    archive = args.work / "desk-source.tgz"
    get_archive(archive, result, offline=args.archive)
    root = args.work / "dataset"
    result["rgb_pose_extraction"] = extract_selected(archive, root)
    result["measured_depth_extraction"] = extract_sensor_depths(archive, root)

    imported = args.work / "imported-desk-native-640"
    invoke("import-TUM-Desk-native-640-registered-RGBD", [
        sys.executable, "-m", "echo4d", "import-tum", str(root / ROOT),
        "--output", str(imported), "--intrinsics", *map(str, FR1_INTRINSICS),
        "--stride", "8", "--limit", str(MAX_IMPORTED),
        "--downsample", "1", "--world-up", "z",
    ], result, timeout=1200)
    recording = load_recording(imported / "recording.json")
    camera = recording.camera
    if (camera.width, camera.height) != (640, 480):
        raise DataError("Expected native 640x480 RGB-D camera")
    expected = Camera(640, 480, *FR1_INTRINSICS)
    if camera.to_dict() != expected.to_dict():
        raise DataError("Native camera intrinsics differ from TUM Freiburg1")
    entries = list(recording.entries)
    train, holdout = split_indices(len(entries))
    poses = recorded_poses(entries)
    recording.entries = [entries[i] for i in train]
    native_frames = [frame for _, frame, _ in recording.iter_frames()]
    if len(native_frames) != len(train):
        raise DataError("Measured training cohort changed")
    half_camera = camera.scaled(2)
    half_frames = [even_pixel_frame(frame, camera) for frame in native_frames]
    args.output.mkdir(parents=True, exist_ok=True)

    result.update({
        "source_scene": "TUM freiburg1_desk", "source_resolution": "640x480",
        "geometry_source": "real registered RGB + measured depth + TUM reference poses; NO prior mesh",
        "calibration_native": camera.to_dict(), "calibration_half": half_camera.to_dict(),
        "lens_rectification": "off in both arms; identical baseline pinhole convention",
        "training_indices": list(train), "heldout_indices": list(holdout),
        "experiment": "practical native 640 / 12mm TSDF vs even-pixel 320 / 18mm TSDF",
        "interpretation_limit": "Resolution and voxel size change together; do not attribute difference to either alone",
        "view_rendering": "Both meshes rendered at 320x240 with identical camera matrices and poses",
    })

    # Build and render one mesh at a time to limit peak Open3D memory.
    specs = [("baseline-320-18mm", half_frames, half_camera, BASELINE_VOXEL_M),
             ("native-640-12mm", native_frames, camera, NATIVE_VOXEL_M)]
    rendered: dict[str, list[np.ndarray]] = {}
    result["meshes"] = {}
    for name, frames, fusion_camera, voxel in specs:
        mesh, _ = fuse(o3d, frames, fusion_camera, pixel_stride=1, voxel=voxel)
        faces = len(mesh.triangles)
        if not 100 <= faces <= MAX_TRIANGLES:
            raise DataError(f"{name}: unreasonable triangle count {faces}")
        export_glb(mesh, args.output / f"desk-{name}.glb")
        result["meshes"][name] = {
            "glb": f"desk-{name}.glb", "triangles": faces,
            "vertices": len(mesh.vertices), "voxel_m": voxel,
            "input_resolution": [fusion_camera.width, fusion_camera.height],
            "training_frames": len(frames),
        }
        frames_rendered = []
        for index in RECORDED:
            frames_rendered.append(render(o3d, mesh, half_camera, poses[index]))
        for side, index in FREE:
            frames_rendered.append(render(o3d, mesh, half_camera,
                                          free_view(poses[index], side)))
        rendered[name] = frames_rendered
        print(f"Exported {name}: {faces} actual triangles, five fixed views", flush=True)
        del mesh
        gc.collect()

    association = json.loads((imported / "association.json").read_text())
    boards = []
    result["views"] = []
    for view_number, index in enumerate(RECORDED):
        photo_path = root / ROOT / association["frames"][index]["rgb_path"]
        with Image.open(photo_path) as image:
            original = image.convert("RGB")
            original.save(args.output / f"real-desk-{index:02d}-640x480.png")
            reference = np.asarray(original.resize((320, 240), Image.Resampling.LANCZOS))
        name = f"recorded-{index:02d}-native-ab.png"
        boards.append(panel(args.output / name, [reference,
            rendered["baseline-320-18mm"][view_number],
            rendered["native-640-12mm"][view_number]],
            ["REAL RGB 640x480", "320 TSDF / 18mm", "640 TSDF / 12mm"]))
        result["views"].append({"kind": "recorded", "index": index, "file": name})
    for free_number, (side, index) in enumerate(FREE):
        view_number = len(RECORDED) + free_number
        before = rendered["baseline-320-18mm"][view_number]
        after = rendered["native-640-12mm"][view_number]
        name = f"free-{'left' if side < 0 else 'right'}-native-ab.png"
        boards.append(panel(args.output / name, [np.full_like(after, 20), before, after],
            ["FREE / no photo", "320 TSDF / 18mm", "640 TSDF / 12mm"]))
        result["views"].append({"kind": "free", "index": index, "file": name,
                                "shift_m": .22, "yaw_deg": 12 * side})
    combined = Image.new("RGB", (960, 267 * len(boards)))
    for i, board in enumerate(boards):
        combined.paste(board, (0, 267 * i))
    combined.save(args.output / "desk-five-view-native640-vs-320.png")
    result["visual_verdict"] = "PENDING independent human inspection of real artifact"
    result["free_orbit"] = "NOT ACCEPTED until reviewed"
    result["issue_74"] = "FAIL unchanged"
    result["historical_desk_v2"] = "FAIL 16.7779 cm > 15 cm unchanged"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    result = {"scene": "TUM Freiburg1 Desk", "source": URL,
              "status": "running", "commands": [],
              "run_url": (f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/"
                          f"{os.getenv('GITHUB_RUN_ID')}" if os.getenv("GITHUB_RUN_ID") else None)}
    try:
        run(args, result)
        result["status"] = "real_colored_mesh_and_five_views_exported_review_pending"
        code = 0
    except Exception as exc:
        traceback.print_exc()
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        code = 1
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"status": result["status"], "meshes": result.get("meshes"),
                          "error": result.get("error")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
