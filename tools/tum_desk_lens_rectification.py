"""Real Freiburg1 Desk: same measured RGB-D frames, pinhole vs calibrated lens rectification.

Both RGB and the pre-registered measured depth are remapped with one Freiburg1
Brown-Conrady camera map. No inferred depth, no scan prior, no new input frames.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import traceback

import cv2
import numpy as np
from PIL import Image

from echo4d.model import Camera, DataError, Frame, load_recording
from tum_desk_sfm import URL, ROOT, get_archive, extract_selected
from tum_desk_full_mesh import extract_sensor_depths
from tum_public_roundtrip import invoke
from tum_desk_known_pose_open3d import MAX_IMPORTED, RECORDED, FREE, FR1_INTRINSICS, split_indices
from tum_xyz_open3d_320 import fuse, export_glb, render, panel, free_view

# Published TUM Freiburg1 RGB distortion in OpenCV order: k1,k2,p1,p2,k3.
FR1_DISTORTION = (0.2624, -0.9531, -0.0054, 0.0026, 1.1633)


def lens_map(camera: Camera, distortion=FR1_DISTORTION) -> tuple[np.ndarray, np.ndarray]:
    """Ideal pinhole output pixel -> distorted original RGB/depth source pixel."""
    if camera.width < 2 or camera.height < 2:
        raise ValueError("Image too small for rectification")
    k = np.array([[camera.fx, 0, camera.cx],
                  [0, camera.fy, camera.cy], [0, 0, 1]], np.float64)
    d = np.asarray(distortion, dtype=np.float64)
    if d.shape != (5,) or not np.isfinite(d).all():
        raise ValueError("Expected finite 5-element Brown-Conrady distortion")
    return cv2.initUndistortRectifyMap(k, d, np.eye(3), k,
                                      (camera.width, camera.height), cv2.CV_32FC1)


def rectify_pair(depth: np.ndarray, color: np.ndarray, maps, excluded=None):
    """Nearest-neighbor measured depth, bilinear RGB, zero remains unmeasured."""
    mx, my = maps
    if depth.shape != mx.shape or mx.shape != my.shape or color.shape != (*depth.shape, 3):
        raise ValueError("Depth, RGB and camera maps must have matching dimensions")
    if excluded is not None and excluded.shape != depth.shape:
        raise ValueError("Excluded mask dimensions differ")
    d = cv2.remap(np.ascontiguousarray(depth.astype(np.float32)), mx, my,
                  cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    c = cv2.remap(np.ascontiguousarray(color), mx, my,
                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ex = None if excluded is None else cv2.remap(
        np.ascontiguousarray(excluded.astype(np.uint8)), mx, my,
        cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1).astype(bool)
    return np.ascontiguousarray(d), np.ascontiguousarray(c), ex


def rectify_frame(frame: Frame, maps) -> Frame:
    if frame.color is None:
        raise DataError("Cannot jointly rectify depth without registered RGB")
    depth, color, excluded = rectify_pair(frame.depth, frame.color, maps, frame.excluded)
    return Frame(frame.t, frame.camera_to_world, depth, color, excluded)


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
        raise DataError("Expected 320x240 RGB-D camera")
    entries = list(record.entries)
    train, holdout = split_indices(len(entries))
    record.entries = [entries[i] for i in train]
    original_frames = [frame for _, frame, _ in record.iter_frames()]
    maps = lens_map(record.camera)
    rectified_frames = [rectify_frame(frame, maps) for frame in original_frames]
    for frame in rectified_frames:
        frame.validate(record.camera)

    result["geometry_source"] = "TUM Freiburg1 Desk measured RGB-D + TUM reference poses; no prior Mesh"
    result["lens_coefficients"] = list(FR1_DISTORTION)
    result["rectification"] = "Same map on RGB and registered sensor depth; nearest depth, bilinear RGB; no hole fill"
    result["training_indices"] = list(train)
    result["heldout_indices"] = list(holdout)
    result["camera_320"] = record.camera.to_dict()
    result["ablation"] = "Same frames/poses, 320x240, 0.018m TSDF; only lens rectification changes"
    old, _ = fuse(o3d, original_frames, record.camera, pixel_stride=1, voxel=.018)
    new, _ = fuse(o3d, rectified_frames, record.camera, pixel_stride=1, voxel=.018)
    args.output.mkdir(parents=True, exist_ok=True)
    export_glb(old, args.output / "desk-measured-320-unrectified.glb")
    export_glb(new, args.output / "desk-measured-320-rectified.glb")
    result["meshes"] = {
        "unrectified": {"triangles": len(old.triangles), "glb": "desk-measured-320-unrectified.glb"},
        "rectified": {"triangles": len(new.triangles), "glb": "desk-measured-320-rectified.glb"},
    }
    association = json.loads((imported / "association.json").read_text())
    boards = []
    result["views"] = []
    full_maps = lens_map(Camera(640, 480, *FR1_INTRINSICS))
    for index in RECORDED:
        pose = np.asarray(entries[index]["cameraToWorld"]).reshape(4, 4, order="F")
        rgb_path = root / ROOT / association["frames"][index]["rgb_path"]
        with Image.open(rgb_path) as image:
            original = image.convert("RGB")
            original.save(args.output / f"real-original-{index:02d}-640x480.png")
        rectified_rgb = cv2.remap(np.asarray(original), *full_maps, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        Image.fromarray(rectified_rgb).save(args.output / f"real-rectified-{index:02d}-640x480.png")
        photo = np.asarray(Image.fromarray(rectified_rgb).resize((320, 240), Image.Resampling.LANCZOS))
        name = f"recorded-{index:02d}-lens-ab.png"
        boards.append(panel(args.output / name, [
            photo, render(o3d, old, record.camera, pose),
            render(o3d, new, record.camera, pose),
        ], ["REAL rectified RGB", "uncorrected TSDF", "lens-corrected TSDF"]))
        result["views"].append({"kind": "recorded", "index": index, "file": name})
        gc.collect()
    for side, index in FREE:
        pose = free_view(np.asarray(entries[index]["cameraToWorld"]).reshape(4, 4, order="F"), side)
        old_rgb = render(o3d, old, record.camera, pose)
        new_rgb = render(o3d, new, record.camera, pose)
        name = f"free-{'left' if side < 0 else 'right'}-lens-ab.png"
        boards.append(panel(args.output / name, [
            np.full_like(new_rgb, 20), old_rgb, new_rgb,
        ], ["FREE / no photo", "uncorrected TSDF", "lens-corrected TSDF"]))
        result["views"].append({"kind": "free", "index": index, "file": name})
        gc.collect()
    combined = Image.new("RGB", (960, 267 * len(boards)))
    for i, board in enumerate(boards):
        combined.paste(board, (0, 267 * i))
    combined.save(args.output / "desk-five-view-lens-ab.png")
    result["human_visual_verdict"] = "PENDING; must inspect actual mesh renderings"
    result["free_orbit"] = "NOT ACCEPTED"
    result["issue_74"] = "FAIL unchanged"
    result["historical_desk_v2"] = "FAIL 16.7779 cm > 15 cm unchanged"


def main():
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
        (args.output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"status": result["status"], "meshes": result.get("meshes"),
                          "error": result.get("error")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
