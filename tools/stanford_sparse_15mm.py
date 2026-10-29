"""Stanford Lounge: real 640x480 / 480-view RGB-D at 15 mm sparse TSDF.

Only the voxel length changes versus the frozen 25 mm observed-only Mesh.
Source images, calibrated camera intrinsics, recorded extrinsics, and 480
selected frame IDs are identical. Pre-existing scanned Mesh is NEVER fused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FROZEN_VOXEL_M = 0.025
CANDIDATE_VOXEL_M = 0.015
EXPECTED_BASELINE_FACES = 454009


def frozen_indices(total: int, baseline: list[int], extra: list[int]) -> list[int]:
    """Refuse a comparison if a different frame set sneaks into the TSDF."""
    if total < 480 or len(baseline) != 360 or len(extra) != 120:
        raise ValueError("Expected original 360 + 120 recorded frames")
    if (len(set(baseline)) != 360 or len(set(extra)) != 120
            or set(baseline) & set(extra)):
        raise ValueError("Repeated frame in frozen Stanford selection")
    combined = sorted(baseline + extra)
    if combined[0] < 0 or combined[-1] >= total or len(combined) != 480:
        raise ValueError("Recorded Stanford frame IDs out of range")
    return combined


def assert_native_intrinsics(k, color_shape, depth_shape):
    """No hidden 640->160/320 resizing or stale principal point."""
    w, h, fx, fy, cx, cy = k
    if ((w, h) != (640, 480)
            or tuple(color_shape[:2]) != (480, 640)
            or tuple(depth_shape[:2]) != (480, 640)
            or not np.isfinite([fx, fy, cx, cy]).all()
            or fx <= 0 or fy <= 0
            or not 0 <= cx < w or not 0 <= cy < h):
        raise ValueError("Requires native calibrated 640x480 Stanford RGB-D")


def run(cache: Path, source: Path, source_report: Path, output: Path):
    import open3d as o3d
    import trimesh
    from open3d_official_room_baseline import spaced_real_frames
    from stanford_lounge_second_room import integrate, lounge_recorded_poses
    from stanford_gap_guided_depth import five_views, to_open3d
    from redwood_prior_first_room import export_colored

    if output.exists():
        raise FileExistsError("Use a fresh output directory")
    frozen = json.loads(source_report.read_text())
    if (frozen.get("after_observed_faces") != EXPECTED_BASELINE_FACES
            or frozen.get("existing_prior_geometry_used") is not False
            or frozen.get("new_frames") != 120):
        raise ValueError("Unexpected frozen 480-frame measured-only source")

    data = o3d.data.LoungeRGBDImages(data_root=str(cache))
    if len(data.color_paths) != len(data.depth_paths):
        raise ValueError("Missing Stanford depth or color frames")
    trajectory = o3d.io.read_pinhole_camera_trajectory(data.trajectory_log_path)
    k, poses, mapping = lounge_recorded_poses(
        trajectory, list(data.color_paths), data.trajectory_log_path)
    assert_native_intrinsics(
        k, np.asarray(o3d.io.read_image(data.color_paths[0])).shape,
        np.asarray(o3d.io.read_image(data.depth_paths[0])).shape)
    if len(poses) != len(data.color_paths):
        raise ValueError("Recorded camera poses are not frame-aligned")
    indices = frozen_indices(
        len(data.color_paths),
        spaced_real_frames(len(data.color_paths), 360),
        frozen["selected_frame_indices"])

    before_model = trimesh.load(str(source), force="mesh", process=False)
    if len(before_model.faces) != EXPECTED_BASELINE_FACES:
        raise ValueError("Frozen measured-only GLB changed")
    before = to_open3d(o3d, before_model)

    # The only changed fusion parameter is voxel length; integrate() uses
    # Open3D's sparse ScalableTSDFVolume, original RGB-D and exact same poses.
    after = integrate(o3d, data, k, poses, indices, CANDIDATE_VOXEL_M)
    output.mkdir(parents=True)
    if not o3d.io.write_triangle_mesh(
            str(output / "OBSERVED-15mm-native.ply"), after):
        raise RuntimeError("Failed to export native 15mm measured Mesh")

    vertices = np.asarray(after.vertices)
    triangles = np.asarray(after.triangles)
    colors = np.asarray(after.vertex_colors)
    if len(triangles) < 10000 or not np.isfinite(vertices).all():
        raise RuntimeError("15mm fusion returned invalid or empty geometry")
    export_colored(output / "OBSERVED-15mm-full.glb",
                   vertices, triangles, colors)
    # Only the convenience preview is simplified; the native GLB and the
    # five real comparison renders retain the original full triangles.
    if len(triangles) > 300000:
        preview = after.simplify_quadric_decimation(300000)
    else:
        preview = after
    export_colored(output / "OBSERVED-15mm-preview.glb",
                   np.asarray(preview.vertices), np.asarray(preview.triangles),
                   np.asarray(preview.vertex_colors))

    views = five_views(o3d, data, poses, k, before, after, output)
    # Reused raycaster defaults to old 360-vs-480 labels. These A/B
    # images are instead the SAME 480 recorded frames at two voxel sizes.
    from PIL import Image, ImageDraw
    for view in views:
        image_path = output / view["file"]
        with Image.open(image_path) as original:
            canvas = original.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, canvas.width, 27), fill=(24, 24, 24))
        label = ("REAL PHOTO / 480 FRAMES 25mm / SAME 480 FRAMES 15mm"
                 if view["kind"] == "recorded" else
                 "ROOM-SCALE FREE / 480 FRAMES 25mm / SAME 480 FRAMES 15mm")
        draw.text((8, 8), label, fill="white")
        canvas.save(image_path)
    report = {
        "scene": "Stanford Lounge, public Open3D LoungeRGBDImages",
        "input": "real native 640x480 RGB-D, logged metric camera poses",
        "recorded_pose_mapping": mapping,
        "intrinsics_640x480": list(k),
        "frozen_measured_frame_count": len(indices),
        "baseline_25mm_observed_faces": len(before.triangles),
        "candidate_15mm_observed_faces": len(triangles),
        "candidate_preview_faces": len(preview.triangles),
        "baseline_voxel_m": FROZEN_VOXEL_M,
        "candidate_voxel_m": CANDIDATE_VOXEL_M,
        "tsdf_truncation_policy": "4 times voxel length in both configurations",
        "voxel_method": "Open3D 0.19 ScalableTSDFVolume, sparse voxel blocks",
        "source_frame_set_identical": True,
        "preexisting_mesh_prior_used": False,
        "inferred_geometry_used": False,
        "native_full_glb": "OBSERVED-15mm-full.glb",
        "three_real_two_room_scale_free": views,
        "visual_acceptance": "NOT REVIEWED: require human side-by-side",
        "issue_74": "FAIL unchanged",
        "desk_v2": "FAIL 16.7779cm > 15cm unchanged"
    }
    (output / "result.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--source-report", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    try:
        run(args.cache, args.source, args.source_report, args.output)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "FAILED.txt").write_text(
            f"{type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    main()
