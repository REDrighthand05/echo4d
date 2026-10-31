"""Real Stanford Lounge: globally consistent measured-plane regularization BEFORE TSDF.

A/B uses identical 144 real RGB-D frames, known poses, 320x240 intrinsics,
20 mm TSDF. A world-space plane is inferred only from the baseline measured
mesh; per-frame sensor depth is snapped to it only within 35 mm of an actual
measurement and with local 3x3 support. No supplied scan or RGB prediction.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback

import numpy as np
from PIL import Image
from scipy import ndimage

from echo4d.model import Camera, DataError, Frame
from lounge_spatial_prior_real import VOXEL_M, read_frames, select_views
from tum_xyz_open3d_320 import export_glb, free_view, fuse, panel, render


def detect_global_planes(o3d, measured, *, max_planes=4, seed=41):
    """Find broad planes from REAL measured TSDF vertices, in world metres."""
    vertices = np.asarray(measured.vertices)
    if len(vertices) < 5000:
        raise DataError("Measured baseline mesh too small for plane support")
    rng = np.random.default_rng(seed)
    subset = vertices[rng.choice(len(vertices), min(90000, len(vertices)),
                                 replace=False)]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(subset)
    planes, descriptions = [], []
    for _ in range(max_planes):
        if len(cloud.points) < 5000:
            break
        equation, inliers = cloud.segment_plane(
            distance_threshold=.025, ransac_n=3, num_iterations=450)
        if len(inliers) < 3000:
            break
        support = np.asarray(cloud.points)[inliers]
        n = np.asarray(equation[:3], dtype=np.float64)
        scale = float(np.linalg.norm(n))
        if scale < 1e-8:
            break
        equation = np.r_[n / scale, float(equation[3]) / scale]
        cloud = cloud.select_by_index(inliers, invert=True)
        if float(np.ptp(support, axis=0).max()) < .80:
            continue
        planes.append(equation)
        descriptions.append(dict(normal=equation[:3].tolist(),
                                 offset_m=float(equation[3]),
                                 measured_support_points=len(inliers),
                                 support_extent_m=np.ptp(support, axis=0).tolist()))
    return planes, descriptions


def regularize_depth(depth, camera, camera_to_world, planes, *,
                     tolerance_m=.035):
    """Snap only locally supported measured pixels near a global measured plane.

    A plane is NEVER allowed to fill an invalid sensor pixel, cross a depth
    discontinuity, or create geometry outside the original observed rays.
    """
    src = np.asarray(depth, dtype=np.float32)
    pose = np.asarray(camera_to_world, dtype=np.float64)
    if src.ndim != 2 or src.shape != (camera.height, camera.width):
        raise ValueError("Depth dimensions differ from calibrated camera")
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Invalid camera-to-world pose")
    if tolerance_m <= 0:
        raise ValueError("Invalid metric tolerance")
    yy, xx = np.indices(src.shape, dtype=np.float64)
    ray_x = (xx - camera.cx) / camera.fx
    ray_y = (yy - camera.cy) / camera.fy
    valid_depth = np.isfinite(src) & (src >= .2) & (src <= 4.)
    best_error = np.full(src.shape, tolerance_m, dtype=np.float64)
    best_target = src.astype(np.float64).copy()
    for plane in planes:
        equation = np.asarray(plane, dtype=np.float64)
        if equation.shape != (4,) or not np.isfinite(equation).all():
            raise ValueError("Invalid plane equation")
        normal_cam = pose[:3, :3].T @ equation[:3]
        denominator = (normal_cam[0] * ray_x + normal_cam[1] * ray_y
                       + normal_cam[2])
        numerator = -(equation[:3] @ pose[:3, 3] + equation[3])
        with np.errstate(divide="ignore", invalid="ignore"):
            target = numerator / denominator
        error = np.abs(target - src)
        supported = (valid_depth & (np.abs(denominator) > .10)
                     & np.isfinite(target) & (target >= .2) & (target <= 4.)
                     & (error < tolerance_m))
        # The interior must have neighboring REAL sensor measurements on
        # the same plane. This protects object silhouettes and thin edges.
        supported = ndimage.binary_erosion(
            supported, structure=np.ones((3, 3), dtype=bool),
            border_value=0)
        better = supported & (error < best_error)
        best_error[better] = error[better]
        best_target[better] = target[better]
    changed = (best_error < tolerance_m) & (np.abs(best_target - src) > .001)
    corrected = src.copy()
    corrected[changed] = best_target[changed].astype(np.float32)
    return np.ascontiguousarray(corrected), dict(
        changed_pixels=int(changed.sum()),
        changed_mean_m=(float(np.mean(np.abs(best_target[changed] - src[changed])))
                        if changed.any() else 0.),
        valid_measured_pixels=int(valid_depth.sum()))


def regularize_frames(frames, camera, planes):
    updated, count, weighted_delta, valid = [], 0, 0., 0
    for frame in frames:
        depth, stats = regularize_depth(
            frame.depth, camera, frame.camera_to_world, planes)
        updated.append(Frame(t=frame.t, camera_to_world=frame.camera_to_world,
                             depth=depth, color=frame.color).validate(camera))
        n = stats["changed_pixels"]
        count += n
        weighted_delta += n * stats["changed_mean_m"]
        valid += stats["valid_measured_pixels"]
    return updated, dict(changed_pixels=count, valid_measured_pixels=valid,
                         changed_fraction=(count / valid if valid else 0.),
                         mean_adjustment_m=(weighted_delta / count if count else 0.))


def change_view(before, after):
    """Magenta pixel mask is a visualization of changed renders, not geometry."""
    difference = np.abs(before.astype(np.int16) - after.astype(np.int16))
    changed = np.max(difference, axis=2) > 22
    image = after.copy()
    image[changed] = ((image[changed].astype(np.uint16) +
                       np.array([255, 0, 255], np.uint16)) // 2).astype(np.uint8)
    return image


def experiment(work, output, receipt):
    import open3d as o3d
    work.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    dataset = o3d.data.LoungeRGBDImages(data_root=str(work))
    params = o3d.io.read_pinhole_camera_trajectory(
        dataset.trajectory_log_path).parameters
    _, selected, poses, pose_indices = select_views(params, dataset.color_paths)
    intr = params[pose_indices[0]].intrinsic
    k = intr.intrinsic_matrix
    full = Camera(intr.width, intr.height, float(k[0, 0]), float(k[1, 1]),
                  float(k[0, 2]), float(k[1, 2]))
    if (full.width, full.height) != (640, 480):
        raise DataError("Unexpected Lounge RGB-D camera calibration")
    half = full.scaled(2)
    for i in selected:
        other = params[pose_indices[i]].intrinsic
        if ((other.width, other.height) != (full.width, full.height)
                or not np.allclose(other.intrinsic_matrix, k, atol=1e-3)):
            raise DataError("Per-frame camera intrinsics differ")
    frames, originals, photos = read_frames(
        dataset, selected, poses, full, half)
    source_frames = [frames[i] for i in selected]
    baseline, _ = fuse(o3d, source_frames, half,
                       pixel_stride=1, voxel=VOXEL_M)
    print("Measured baseline triangles:", len(baseline.triangles), flush=True)
    planes, descriptions = detect_global_planes(o3d, baseline)
    print("Broad measured planes:", len(planes), flush=True)
    adjusted, stats = regularize_frames(source_frames, half, planes)
    print("Real depth pixels regularized:", stats, flush=True)
    improved, _ = fuse(o3d, adjusted, half, pixel_stride=1, voxel=VOXEL_M)
    print("Plane-regularized triangles:", len(improved.triangles), flush=True)
    for name, mesh in (
        ("MEASURED-144-baseline.glb", baseline),
        ("MEASURED-144-global-plane-regularized.glb", improved),
    ):
        export_glb(mesh, output / name)
    labels = ["REAL / FREE", "MEASURED BASE", "PLANE TSDF", "CHANGED PIXELS"]
    boards, views = [], []
    for i in photos:
        Image.fromarray(originals[i]).save(output / f"REAL-lounge-{i:04d}.png")
        photo = np.asarray(Image.fromarray(originals[i]).resize(
            (half.width, half.height), Image.Resampling.LANCZOS))
        before = render(o3d, baseline, half, poses[i])
        after = render(o3d, improved, half, poses[i])
        name = f"recorded-{i:04d}.png"
        boards.append(panel(output / name, [
            photo, before, after, change_view(before, after)], labels))
        views.append(dict(kind="recorded", frame=i, file=name))
    for i, side in ((photos[0], -1.), (photos[-1], 1.)):
        pose = free_view(poses[i], side)
        before = render(o3d, baseline, half, pose)
        after = render(o3d, improved, half, pose)
        name = f"free-{i:04d}.png"
        boards.append(panel(output / name, [
            np.full_like(before, 20), before, after,
            change_view(before, after)], labels))
        views.append(dict(kind="free", frame=i, file=name))
    board = Image.new("RGB", (half.width * 4, (half.height + 27) * 5))
    for row, image in enumerate(boards):
        board.paste(image, (0, row * (half.height + 27)))
    board.save(output / "lounge-five-view-global-plane-tsdf.png")
    receipt.update(
        status="real_meshes_and_five_views_exported_review_pending",
        original_resolution=[full.width, full.height],
        fusion_resolution=[half.width, half.height],
        measured_frames=len(selected), voxel_m=VOXEL_M,
        baseline_triangles=len(baseline.triangles),
        regularized_triangles=len(improved.triangles),
        global_planes=descriptions, regularization=stats,
        original_photo_indices=photos, views=views,
        provenance="Both meshes use measured RGB-D only; plane constraint inferred from baseline measured mesh; no supplied PRIOR",
        visual_verdict="PENDING human review; free orbit FAIL until visually validated",
        issue_74="FAIL unchanged",
        old_desk="FAIL 16.7779cm > 15cm unchanged")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = dict(scene="Real Stanford Lounge 640x480 RGB-D",
                   source="Open3D LoungeRGBDImages; 144 sensor frames and known poses",
                   status="running",
                   run_url=(f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/"
                            f"{os.getenv('GITHUB_RUN_ID')}" if os.getenv("GITHUB_RUN_ID") else None))
    try:
        experiment(args.work, args.output, receipt)
        code = 0
    except Exception as exc:
        traceback.print_exc()
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        code = 1
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "result.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"status": receipt["status"],
                          "error": receipt.get("error")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
