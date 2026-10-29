"""Stanford Lounge: bounded per-frame RGB-D ICP against a FROZEN measured Mesh.

The frozen 480-frame 25mm TSDF is a *self-derived geometric reference*, not
an independent scan or a new source of geometry. No prior or inferred surfaces
are fused. Compare original logged poses with bounded ICP-refined poses using
the exact same 480 native 640x480 measured RGB-D frames and five cameras.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stanford_sparse_15mm import (
    EXPECTED_BASELINE_FACES, assert_native_intrinsics, frozen_indices,
)

VOXEL_M = 0.025
MAX_TRANSLATION_M = 0.030
MAX_ROTATION_DEG = 1.5
MIN_FITNESS = 0.35
MAX_RMSE_M = 0.055


def pose_correction_accepted(transform, fitness, rmse):
    """Reject nonrigid, excessive, or weak ICP corrections; never silently clamp."""
    t = np.asarray(transform, dtype=np.float64)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        return False, "invalid_transform", 0.0, 0.0
    rot = t[:3, :3]
    if (not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(rot.T @ rot, np.eye(3), atol=2e-3)
            or abs(np.linalg.det(rot) - 1) > 2e-3):
        return False, "nonrigid_transform", 0.0, 0.0
    distance = float(np.linalg.norm(t[:3, 3]))
    angle = float(np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1))))
    if distance > MAX_TRANSLATION_M or angle > MAX_ROTATION_DEG:
        return False, "correction_too_large", distance, angle
    if not np.isfinite([fitness, rmse]).all() or fitness < MIN_FITNESS or rmse > MAX_RMSE_M:
        return False, "weak_registration", distance, angle
    return True, "accepted", distance, angle


def corrected_world_to_camera(extrinsic, world_correction):
    """Open3D logs world->camera E; ICP maps old world point to corrected world.

    If X_new = T @ X_old, then E_new = E_old @ inverse(T).
    """
    e = np.asarray(extrinsic, dtype=np.float64)
    t = np.asarray(world_correction, dtype=np.float64)
    if e.shape != (4, 4) or t.shape != (4, 4):
        raise ValueError("Expected 4x4 camera and ICP transforms")
    return e @ np.linalg.inv(t)


def refine_poses(o3d, data, k, poses, indices, reference):
    """Small, independently bounded frame-to-reference ICP; no trajectory chaining."""
    target = reference.sample_points_uniformly(
        number_of_points=200000, use_triangle_normal=True)
    if not target.has_normals():
        raise RuntimeError("Frozen measured reference has no target normals")
    intrinsic = o3d.camera.PinholeCameraIntrinsic(*k)
    output = list(poses)
    stats = []
    for n, i in enumerate(indices):
        raw = o3d.io.read_image(data.depth_paths[i])
        depth = np.asarray(raw)
        assert_native_intrinsics(
            k, np.asarray(o3d.io.read_image(data.color_paths[i])).shape, depth.shape)
        source = o3d.geometry.PointCloud.create_from_depth_image(
            raw, intrinsic, depth_scale=1000., depth_trunc=4., stride=4)
        if len(source.points) < 200:
            stats.append({"frame": int(i), "state": "too_few_sensor_points"})
            continue
        source.transform(np.linalg.inv(poses[i]))
        source = source.voxel_down_sample(.045)
        if len(source.points) < 200:
            stats.append({"frame": int(i), "state": "too_few_downsampled_points"})
            continue
        result = o3d.pipelines.registration.registration_icp(
            source, target, .085, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=8))
        accepted, reason, distance, angle = pose_correction_accepted(
            result.transformation, result.fitness, result.inlier_rmse)
        if accepted:
            output[i] = corrected_world_to_camera(poses[i], result.transformation)
        stats.append({
            "frame": int(i), "state": reason,
            "fitness": float(result.fitness),
            "rmse_m": float(result.inlier_rmse),
            "translation_m": distance,
            "rotation_deg": angle,
        })
        if (n + 1) % 40 == 0:
            print(f"ICP {n+1}/{len(indices)} accepted "
                  f"{sum(s['state']=='accepted' for s in stats)}", flush=True)
    return output, stats


def run(cache: Path, source: Path, source_report: Path, output: Path):
    import open3d as o3d
    import trimesh
    from open3d_official_room_baseline import spaced_real_frames
    from stanford_lounge_second_room import integrate, lounge_recorded_poses
    from stanford_gap_guided_depth import five_views, to_open3d
    from redwood_prior_first_room import export_colored
    from PIL import Image, ImageDraw

    if output.exists():
        raise FileExistsError("Use a new result directory")
    frozen = json.loads(source_report.read_text())
    if (frozen.get("after_observed_faces") != EXPECTED_BASELINE_FACES
            or frozen.get("existing_prior_geometry_used") is not False
            or frozen.get("new_frames") != 120):
        raise ValueError("Frozen measured-only baseline changed")
    model = trimesh.load(str(source), force="mesh", process=False)
    if len(model.faces) != EXPECTED_BASELINE_FACES:
        raise ValueError("Unexpected frozen source GLB triangle count")
    before = to_open3d(o3d, model)
    before.compute_triangle_normals()

    data = o3d.data.LoungeRGBDImages(data_root=str(cache))
    if (len(data.color_paths) != len(data.depth_paths)
            or any(Path(a).stem != Path(b).stem for a, b in
                   zip(data.color_paths, data.depth_paths))):
        raise ValueError("Unsynchronized real RGB/depth frames")
    trajectory = o3d.io.read_pinhole_camera_trajectory(data.trajectory_log_path)
    k, poses, mapping = lounge_recorded_poses(
        trajectory, list(data.color_paths), data.trajectory_log_path)
    if len(poses) != len(data.color_paths):
        raise ValueError("Missing frame-aligned logged poses")
    indices = frozen_indices(
        len(data.color_paths),
        spaced_real_frames(len(data.color_paths), 360),
        frozen["selected_frame_indices"])
    assert_native_intrinsics(
        k, np.asarray(o3d.io.read_image(data.color_paths[0])).shape,
        np.asarray(o3d.io.read_image(data.depth_paths[0])).shape)

    refined, stats = refine_poses(o3d, data, k, poses, indices, before)
    accepted = sum(s["state"] == "accepted" for s in stats)
    print(f"Accepted {accepted}/{len(indices)} bounded per-frame corrections", flush=True)
    # Frozen reference is NOT added to the TSDF. Only sensor RGB-D is integrated.
    after = integrate(o3d, data, k, refined, indices, VOXEL_M)
    if not after.has_triangles() or not after.has_vertex_colors():
        raise RuntimeError("No real measured colored mesh after ICP")
    vertices = np.asarray(after.vertices)
    triangles = np.asarray(after.triangles)
    colors = np.asarray(after.vertex_colors)
    if len(triangles) < 10000 or not np.isfinite(vertices).all():
        raise RuntimeError("Invalid refined native mesh")
    output.mkdir(parents=True)
    if not o3d.io.write_triangle_mesh(
            str(output / "OBSERVED-25mm-bounded-icp-native.ply"), after):
        raise RuntimeError("PLY export failed")
    export_colored(output / "OBSERVED-25mm-bounded-icp-full.glb",
                   vertices, triangles, colors)
    views = five_views(o3d, data, poses, k, before, after, output)
    for view in views:
        path = output / view["file"]
        with Image.open(path) as img:
            canvas = img.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, canvas.width, 27), fill=(24, 24, 24))
        draw.text((8, 8),
                  ("REAL PHOTO / ORIGINAL 25mm / BOUNDED ICP 25mm"
                   if view["kind"] == "recorded" else
                   "FREE CAMERA / ORIGINAL 25mm / BOUNDED ICP 25mm"),
                  fill="white")
        canvas.save(path)
    report = {
        "scene": "Stanford Lounge, Open3D public RGB-D",
        "source": "480 native 640x480 measured RGB-D, logged metric poses",
        "intrinsics": list(k),
        "pose_mapping": mapping,
        "frames_identical": True,
        "voxel_m_both": VOXEL_M,
        "tsdf_truncation_m_both": .1,
        "original_observed_faces": len(before.triangles),
        "bounded_icp_observed_faces": len(triangles),
        "accepted_pose_corrections": accepted,
        "rejected_pose_corrections": len(indices) - accepted,
        "correction_limit_m": MAX_TRANSLATION_M,
        "correction_limit_deg": MAX_ROTATION_DEG,
        "pose_corrections": stats,
        "geometry_reference": "FROZEN measured 480-frame 25mm mesh, ICP target ONLY",
        "reference_mesh_faces_fused": 0,
        "preexisting_scan_prior_used": False,
        "inferred_geometry_used": False,
        "five_comparison_views": views,
        "visual_acceptance": "NOT REVIEWED; no improvement claimed",
        "issue_74": "FAIL unchanged",
        "desk_v2": "FAIL 16.7779cm > 15cm unchanged",
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k != "pose_corrections"}), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--source-report", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    try:
        run(a.cache, a.source, a.source_report, a.output)
    except Exception as exc:
        a.output.mkdir(parents=True, exist_ok=True)
        (a.output / "FAILED.txt").write_text(
            f"{type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    main()
