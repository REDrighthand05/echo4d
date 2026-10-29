"""Measured-depth loop-overlap pose correction and real Redwood Bedroom TSDF A/B.

Only original RGB-D, native intrinsics and logged camera poses enter geometry.
A frozen observed-only Mesh is used solely for same-camera A/B rendering.
No reference scan, predicted depth, or existing Mesh is used for registration.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from open3d_official_room_baseline import read_poses, spaced_real_frames
from redwood_dense_native_review import export_full
from redwood_gap_guided_fusion import compare, integrate

BASELINE_FACES = 553208
MAX_TRANSLATION_M = .03
MAX_ROTATION_DEG = 1.5
MAX_CONSENSUS_TRANSLATION_M = .025
MAX_CONSENSUS_ROTATION_DEG = 1.0


def rotation_degrees(rotation):
    r = np.asarray(rotation, dtype=float)
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def validate_delta(delta, max_translation=MAX_TRANSLATION_M,
                   max_rotation=MAX_ROTATION_DEG):
    t = np.asarray(delta, dtype=float)
    return bool(t.shape == (4, 4) and np.isfinite(t).all()
                and np.allclose(t[3], [0, 0, 0, 1], atol=1e-5)
                and np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-3)
                and abs(np.linalg.det(t[:3, :3]) - 1) < 1e-3
                and np.linalg.norm(t[:3, 3]) <= max_translation
                and rotation_degrees(t[:3, :3]) <= max_rotation)


def consensus_delta(a, b):
    """Accept only two independently measured, compatible small corrections."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not (validate_delta(a) and validate_delta(b)):
        return None
    if (np.linalg.norm(a[:3, 3] - b[:3, 3]) > MAX_CONSENSUS_TRANSLATION_M
            or rotation_degrees(a[:3, :3] @ b[:3, :3].T)
            > MAX_CONSENSUS_ROTATION_DEG):
        return None
    rotation_mean = (a[:3, :3] + b[:3, :3]) / 2
    u, _, vt = np.linalg.svd(rotation_mean)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = (a[:3, 3] + b[:3, 3]) / 2
    return result if validate_delta(result) else None


def corrected_world_to_camera(original_world_to_camera, world_delta):
    """ICP transforms source world points; convert to corrected camera extrinsic."""
    old = np.asarray(original_world_to_camera, dtype=float)
    delta = np.asarray(world_delta, dtype=float)
    if old.shape != (4, 4) or not validate_delta(delta):
        raise ValueError("Invalid camera pose correction")
    return old @ np.linalg.inv(delta)


def loop_targets(selected, poses, max_distance=.75, min_frame_gap=50):
    """Find two independently observed, overlapping original sensor views."""
    camera_to_world = [np.linalg.inv(poses[i]) for i in selected]
    centers = np.array([p[:3, 3] for p in camera_to_world])
    forward = np.array([p[:3, 2] for p in camera_to_world])
    result = {}
    for j, idx in enumerate(selected):
        dist = np.linalg.norm(centers - centers[j], axis=1)
        dot = forward @ forward[j]
        candidates = [q for q in range(len(selected))
                      if abs(selected[q] - idx) >= min_frame_gap
                      and dist[q] <= max_distance and dot[q] >= .65]
        candidates.sort(key=lambda q: (float(dist[q]) + .15 * (1 - dot[q]), selected[q]))
        chosen = []
        for q in candidates:
            if not chosen or abs(selected[q] - selected[chosen[0]]) >= min_frame_gap:
                chosen.append(q)
            if len(chosen) == 2:
                break
        if len(chosen) == 2:
            result[j] = [selected[q] for q in chosen]
    return result


def measured_world_cloud(o3d, path, world_to_camera, k, pixel_stride=8):
    """Use original 16-bit measured depth, sampled at native pixel coordinates."""
    w, h, fx, fy, cx, cy = k
    raw = np.asarray(o3d.io.read_image(str(path)))
    if raw.shape != (h, w) or raw.dtype != np.uint16:
        raise ValueError("Original depth must be calibrated 16-bit millimeters")
    vv, uu = np.mgrid[pixel_stride // 2:h:pixel_stride,
                      pixel_stride // 2:w:pixel_stride]
    z = raw[vv, uu].astype(np.float64) / 1000.
    valid = (z > .25) & (z < 4.5)
    zz, xx, yy = z[valid], uu[valid], vv[valid]
    xyz = np.column_stack(((xx - cx) * zz / fx,
                           (yy - cy) * zz / fy, zz))
    c2w = np.linalg.inv(world_to_camera)
    world = xyz @ c2w[:3, :3].T + c2w[:3, 3]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(world)
    return cloud


def register_pair(o3d, source, target, threshold=.075):
    """ICP on measured surfaces; compare against unchanged logged alignment."""
    if len(source.points) < 300 or len(target.points) < 300:
        return None, {"reason": "too_few_depth_samples"}
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=.12, max_nn=24))
    reg = o3d.pipelines.registration
    before = reg.evaluate_registration(source, target, threshold, np.eye(4))
    result = reg.registration_icp(
        source, target, threshold, np.eye(4),
        reg.TransformationEstimationPointToPlane(),
        reg.ICPConvergenceCriteria(max_iteration=25))
    delta = result.transformation
    accepted = (validate_delta(delta) and result.fitness >= .35
                and result.inlier_rmse < .045
                and before.inlier_rmse > 0
                and result.inlier_rmse <= before.inlier_rmse * .92
                and result.fitness >= before.fitness * .95)
    return (delta if accepted else None), {
        "before_fitness": round(float(before.fitness), 4),
        "after_fitness": round(float(result.fitness), 4),
        "before_rmse_m": round(float(before.inlier_rmse), 5),
        "after_rmse_m": round(float(result.inlier_rmse), 5),
        "delta_translation_m": round(float(np.linalg.norm(delta[:3, 3])), 5),
        "delta_rotation_deg": round(rotation_degrees(delta[:3, :3]), 4),
        "accepted": bool(accepted),
    }


def run(cache, source_ply, out, budget=600):
    import open3d as o3d
    data = o3d.data.BedroomRGBDImages(data_root=str(cache))
    colors, depths = list(data.color_paths), list(data.depth_paths)
    if len(colors) != len(depths) or len(colors) < budget:
        raise ValueError("Missing original real RGB-D frames")
    if any(Path(c).stem != Path(d).stem for c, d in zip(colors, depths)):
        raise ValueError("Color/depth sensor frame IDs disagree")
    trajectory = o3d.io.read_pinhole_camera_trajectory(data.trajectory_log_path)
    k, poses = read_poses(trajectory, colors, data.trajectory_log_path)
    if k[:2] != (640, 480):
        raise ValueError("Native Redwood calibration must be 640x480")
    baseline = o3d.io.read_triangle_mesh(str(source_ply))
    if not baseline.has_vertex_colors() or len(baseline.triangles) != BASELINE_FACES:
        raise ValueError("Frozen 600-frame 4.5m observed baseline mismatch")

    selected = spaced_real_frames(len(colors), budget)
    candidates = loop_targets(selected, poses)
    corrected = list(poses)
    pair_cache = {}
    pair_records = []
    accepted = []
    print("Measured-depth overlapping pose candidates: " + str(len(candidates)), flush=True)
    for j, targets in candidates.items():
        idx = selected[j]
        if idx not in pair_cache:
            pair_cache[idx] = measured_world_cloud(o3d, depths[idx], poses[idx], k)
        source = pair_cache[idx]
        proposals = []
        records = []
        for other in targets:
            if other not in pair_cache:
                pair_cache[other] = measured_world_cloud(
                    o3d, depths[other], poses[other], k)
            delta, info = register_pair(o3d, source, pair_cache[other])
            proposals.append(delta)
            records.append({"target_frame": other, **info})
        agreed = (consensus_delta(*proposals)
                  if all(delta is not None for delta in proposals) else None)
        if agreed is not None:
            corrected[idx] = corrected_world_to_camera(poses[idx], agreed)
            accepted.append(idx)
        pair_records.append({"frame": idx, "targets": records,
                             "consensus_accepted": agreed is not None})
        if len(pair_records) % 50 == 0:
            print("Depth ICP " + str(len(pair_records)) + "/" + str(len(candidates))
                  + ", accepted " + str(len(accepted)), flush=True)
    del pair_cache
    out.mkdir(parents=True, exist_ok=False)
    if not accepted:
        (out / "NO_CORRECTIONS.txt").write_text(
            "No two independent depth overlaps supported a safe pose correction.\n")
        raise RuntimeError("No supported geometry correction; no fake Mesh improvement")

    print("Re-fusing real 600-frame measured depth with " + str(len(accepted))
          + " corrected poses", flush=True)
    candidate = integrate(o3d, data, selected, corrected, k)
    if not o3d.io.write_triangle_mesh(str(out / "redwood-measured-pose-refused.ply"),
                                      candidate):
        raise RuntimeError("Could not export real observed-only Mesh")
    exported = export_full(candidate, out)
    (out / exported["glb"]).rename(out / "redwood-measured-pose-refused.glb")
    views = compare(o3d, data, poses, k, baseline, candidate, out)
    board = Image.new("RGB", (960, 268 * len(views)), (26, 26, 26))
    for j, entry in enumerate(views):
        path = out / entry["file"]
        with Image.open(path) as file:
            im = file.convert("RGB").copy()
        ImageDraw.Draw(im).rectangle((0, 0, im.width, 27), fill=(26, 26, 26))
        label = ("REAL RGB / ORIGINAL POSES / DEPTH-ICP POSES"
                 if entry["kind"] == "recorded"
                 else "FREE VIEW / ORIGINAL POSES / DEPTH-ICP POSES")
        ImageDraw.Draw(im).text((8, 8), label, fill="white")
        im.save(path)
        board.paste(im, (0, j * 268))
    board.save(out / "five-view-measured-pose-ab.png")
    report = {
        "scene": "Real Redwood Bedroom",
        "input": "native 640x480 RGB-D, measured depth, logged metric poses",
        "geometry_action": "two-independent-target measured depth ICP pose correction",
        "same_600_frames": len(selected),
        "original_poses": len(poses),
        "candidate_frames_with_two_loop_targets": len(candidates),
        "accepted_pose_corrections": len(accepted),
        "accepted_frame_ids": accepted,
        "original_observed_faces": len(baseline.triangles),
        "candidate_observed_faces": len(candidate.triangles),
        "preexisting_reference_scan_used": False,
        "frozen_mesh_used_only_for_AB": True,
        "human_visual_acceptance": "PENDING",
        "free_orbit": "FAIL until directly verified",
        "issue_74": "FAIL unchanged",
        "old_desk": "FAIL 16.7779cm > 15cm unchanged",
        "views": views,
        "color_glb": "redwood-measured-pose-refused.glb",
    }
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "pair-depth-diagnostics.json").write_text(
        json.dumps(pair_records, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget", type=int, default=600)
    args = parser.parse_args()
    if not 3 <= args.budget <= 600:
        parser.error("Use 3..600 original RGB-D frames")
    try:
        run(args.cache, args.source_ply, args.output, args.budget)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "FAILURE.txt").write_text(
            type(exc).__name__ + ": " + str(exc) + "\n")
        raise


if __name__ == "__main__":
    main()
