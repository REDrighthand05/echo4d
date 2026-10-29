"""Real TUM fr1_room: remove 83/95 known camera poses while keeping RGB + sensor Z.

Mature OpenCV SIFT / PnP-RANSAC estimates relative metric camera motion from
observed RGB keypoints and independently measured RGB-D. Known poses enter only
as explicitly counted anchors; all other GT poses are read AFTER reconstruction
for diagnostics. The prior Mesh is never an input to candidate fusion.
"""
from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from echo4d.model import DataError, file_sha256
from echo4d.holdout import read_echo_ply
from sparse_open3d_fusion import fuse_sparse
from tum_highres_tsdf import face_components
from tum_room_da2_global_scale_replay import (
    FROZEN_ARCHIVE_SHA256, checked_replay, five_view_comparison, mesh_glb)
from tum_room_sensor_anchor_ladder import verified_sensor_depths
from tum_room_sparse_quality import get_archive

FROZEN_RGB_RUN = 40832708084
FROZEN_ASSOC_RUN = 40790769632


def camera_to_world(frame):
    return np.asarray(frame["cameraToWorld"], dtype=np.float64).reshape(4, 4).T


def camera_matrix(intr):
    return np.array([[intr["fx"], 0, intr["cx"]],
                     [0, intr["fy"], intr["cy"]], [0, 0, 1]], dtype=np.float64)


def anchor_indices(count, n):
    if count < 1 or count > n:
        raise ValueError("Anchor count must be between one and frame count")
    return np.unique(np.linspace(0, n - 1, count, dtype=int)).tolist()


def pair_pnp(kp_a, des_a, kp_b, des_b, depth_a, K, matcher=None):
    """Observed 3D in camera A -> observed 2D in camera B; never uses GT pose."""
    if des_a is None or des_b is None or len(des_a) < 12 or len(des_b) < 12:
        return None
    matcher = matcher or cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(des_a, des_b, k=2)
    good = [m for m, n in matches if m.distance < 0.72 * n.distance]
    if len(good) < 12:
        return None
    src, dst = [], []
    h, w = depth_a.shape
    for m in good:
        x, y = kp_a[m.queryIdx].pt
        xi, yi = round(x), round(y)
        if not (0 <= xi < w and 0 <= yi < h):
            continue
        z = float(depth_a[yi, xi])
        if not np.isfinite(z) or not 0.25 < z < 4.0:
            continue
        src.append(((x - K[0, 2]) * z / K[0, 0],
                    (y - K[1, 2]) * z / K[1, 1], z))
        dst.append(kp_b[m.trainIdx].pt)
    if len(src) < 12:
        return None
    obj = np.asarray(src, dtype=np.float32)
    pixels = np.asarray(dst, dtype=np.float32)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj, pixels, K, None, iterationsCount=300,
        reprojectionError=3.0, confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 12:
        return None
    ids = inliers.ravel()
    if len(ids) / len(obj) < 0.45:
        return None
    # Refine on actual inliers, not a learned or synthetic pose prior.
    rvec, tvec = cv2.solvePnPRefineLM(obj[ids], pixels[ids], K, None, rvec, tvec)
    projected, _ = cv2.projectPoints(obj[ids], rvec, tvec, K, None)
    residual = np.linalg.norm(projected.reshape(-1, 2) - pixels[ids], axis=1)
    median_px = float(np.median(residual))
    R, _ = cv2.Rodrigues(rvec)
    degrees = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    displacement = float(np.linalg.norm(tvec))
    if median_px > 2.5 or degrees > 125 or displacement > 1.5:
        return None
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = tvec.ravel()
    return transform, {"inliers": int(len(ids)), "matches_with_depth": len(obj),
                       "median_reprojection_px": median_px,
                       "rotation_degrees": degrees,
                       "relative_translation_m": displacement}


def build_pose_graph(colors, depths, K, max_gap=2):
    sift = cv2.SIFT_create(nfeatures=3000)
    features = [sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), None)
                for im in colors]
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    graph = [[] for _ in colors]
    tested, successful = 0, 0
    for i in range(len(colors)):
        for j in range(i + 1, min(len(colors), i + max_gap + 1)):
            tested += 1
            forward = pair_pnp(*features[i], *features[j], depths[i], K, matcher)
            reverse = pair_pnp(*features[j], *features[i], depths[j], K, matcher)
            if forward is None and reverse is None:
                continue
            # Select the better independently estimated direction; invert
            # only the observed relative transform, not any ground-truth pose.
            if reverse is not None and (forward is None or
                    reverse[1]["inliers"] > forward[1]["inliers"]):
                T = np.linalg.inv(reverse[0])
                quality = reverse[1]
            else:
                T, quality = forward
            successful += 1
            cost = (1.0 + 25.0 / quality["inliers"] +
                    quality["median_reprojection_px"] / 4.0 + (j - i) * 0.25)
            graph[i].append((j, T, cost))
            graph[j].append((i, np.linalg.inv(T), cost))
    return graph, {"tested_pairs": tested, "successful_pairs": successful}


def estimate_poses(graph, reference_poses, anchors):
    """Multi-source shortest reliable paths, fixed only at declared anchors."""
    n = len(graph)
    estimated = [None] * n
    cost = [float("inf")] * n
    provenance = [None] * n
    queue = []
    for a in anchors:
        estimated[a] = reference_poses[a].copy()
        cost[a] = 0.0
        provenance[a] = {"kind": "known_metric_anchor", "anchor": a}
        heapq.heappush(queue, (0.0, a))
    locked = set(anchors)
    while queue:
        current_cost, i = heapq.heappop(queue)
        if current_cost > cost[i] + 1e-8:
            continue
        for j, T_i_to_j, edge_cost in graph[i]:
            if j in locked:
                continue
            proposed_cost = current_cost + edge_cost
            if proposed_cost >= cost[j]:
                continue
            estimated[j] = estimated[i] @ np.linalg.inv(T_i_to_j)
            cost[j] = proposed_cost
            provenance[j] = {"kind": "rgbd_sift_pnp", "from": i,
                             "cost": proposed_cost}
            heapq.heappush(queue, (proposed_cost, j))
    return estimated, provenance


def prepare_candidate(source, manifest, depths, poses, provenance, output):
    if output.exists():
        raise DataError("Refusing to overwrite existing candidate")
    (output / "color").mkdir(parents=True)
    (output / "depth").mkdir(parents=True)
    frames = []
    for i, (original, z, pose) in enumerate(zip(manifest["frames"], depths, poses)):
        if pose is None:
            continue
        if not np.isfinite(pose).all() or abs(np.linalg.det(pose[:3, :3]) - 1) > 0.05:
            raise DataError("Invalid estimated rigid camera pose")
        color_name = f"color/{i:03}.png"
        depth_name = f"depth/{i:03}.npy"
        shutil.copyfile(source / original["color"], output / color_name)
        np.save(output / depth_name, z.astype(np.float32), allow_pickle=False)
        frame = {k: v for k, v in original.items()
                 if k not in {"cameraToWorld", "color", "depth"}}
        frame.update({"cameraToWorld": pose.T.reshape(-1).tolist(),
                      "color": color_name, "depth": depth_name,
                      "pose_provenance": provenance[i]["kind"]})
        frames.append(frame)
    candidate = {k: v for k, v in manifest.items() if k != "frames"}
    candidate["frames"] = frames
    candidate["name"] = "TUM room measured RGB-D / sparse known pose anchors / SIFT-PnP"
    candidate["notes"] = ("Real measured sensor depth for all frames; metric RGB-D PnP "
                          "for nonanchor poses; no prior Mesh; GT used only at declared anchors")
    (output / "recording.json").write_text(
        json.dumps(candidate, indent=2, allow_nan=False) + "\n")
    return len(frames)


def run(source, room96, output, work, archive_override=None, levels=(13, 5, 1), max_gap=6):
    source, room96, output, work = map(Path, (source, room96, output, work))
    frozen, replay, _ = checked_replay(source)
    source96 = room96 / "replay/predicted-96"
    manifest = json.loads((source96 / "recording.json").read_text())
    if len(manifest["frames"]) != 96:
        raise DataError("Expected exactly 96 frozen real RGB frames")
    if output.exists() or work.exists():
        raise DataError("Refusing to overwrite experiment directories")
    output.mkdir(parents=True)
    work.mkdir(parents=True)
    association = json.loads((replay / "source/association.json").read_text())
    source_ids = [f["source_anchor_index"] for f in manifest["frames"]]
    if len(set(source_ids)) != 96:
        raise DataError("Duplicate real TUM source frames")
    archive = work / "official-room.tgz"
    receipt = {}
    get_archive(archive, receipt, Path(archive_override) if archive_override else None)
    if file_sha256(archive) != FROZEN_ARCHIVE_SHA256:
        raise DataError("Official real TUM archive hash differs from frozen source")
    sensors, hashes = verified_sensor_depths(archive, association, source_ids)
    depths = [sensors[i] for i in source_ids]
    colors = [cv2.cvtColor(cv2.imread(str(source96 / f["color"])),
                           cv2.COLOR_BGR2RGB) for f in manifest["frames"]]
    if any(im is None or im.shape[:2] != (240, 320) for im in colors):
        raise DataError("Missing or unexpected frozen real RGB")
    known = [camera_to_world(f) for f in manifest["frames"]]
    graph, graph_report = build_pose_graph(colors, depths, camera_matrix(manifest["intrinsics"]), max_gap=max_gap)
    report = {"experiment": "TUM fr1_room 96 real measured RGB-D / sparse pose anchors",
              "date_utc": "2026-10-29",
              "retained": ["96 real RGB", "96 measured sensor depth maps",
                           "calibrated intrinsics", "declared metric pose anchors"],
              "removed": ["all other ground-truth poses", "prior Mesh"],
              "source_archive_sha256": FROZEN_ARCHIVE_SHA256,
              "sensor_depth_pngs_sha256": {str(k): v for k, v in hashes.items()},
              "pose_graph": {**graph_report, "max_temporal_pair_gap": max_gap}, "variants": {},
              "visual_acceptance": "UNREVIEWED; no #74 PASS"}
    old_mesh = room96 / "mesh-sensor-96/mesh.ply"
    for count in levels:
        anchors = anchor_indices(count, 96)
        poses, provenance = estimate_poses(graph, known, anchors)
        candidate = output / f"input-poses-{count}"
        used = prepare_candidate(source96, manifest, depths, poses, provenance, candidate)
        destination = output / f"mesh-poses-{count}"
        fusion = fuse_sparse(candidate / "recording.json", destination,
                             voxel_m=0.03, truncation_m=0.09,
                             max_depth_m=4.0, pixel_stride=2)
        mesh = destination / "mesh.ply"
        # The reused TSDF exporter has a fixed "known poses" PLY comment.
        # Correct that provenance before hashing or handing the GLB to users.
        original_label = b"comment Open3D sparse TSDF; measured RGB-D; known poses; no inferred depth"
        label = (f"comment Open3D sparse TSDF; measured sensor Z; {count}/96 known "
                 "pose anchors; other poses RGB-D SIFT PnP; no prior Mesh").encode()
        raw_mesh = mesh.read_bytes()
        if raw_mesh.count(original_label) != 1:
            raise DataError("Unexpected PLY pose provenance label")
        mesh.write_bytes(raw_mesh.replace(original_label, label, 1))
        fusion["mesh_sha256"] = file_sha256(mesh)
        fusion["geometry_source"] = (f"96 real sensor-depth frames; {count} known "
                                     "metric pose anchors; remaining camera poses estimated from RGB-D")
        (destination / "report.json").write_text(
            json.dumps(fusion, indent=2, allow_nan=False) + "\n")
        mesh_glb(mesh, output / f"measured-rgbd-poses-{count}.glb")
        v, f = read_echo_ply(mesh)
        # Independent diagnostic AFTER fusion: never feeds withheld GT back to estimator.
        errors = [float(np.linalg.norm(p[:3, 3] - known[i][:3, 3]))
                  for i, p in enumerate(poses) if p is not None and i not in anchors]
        board_dir = output / f"visual-poses-{count}"
        board_dir.mkdir()
        views = five_view_comparison(source, replay, old_mesh, mesh, board_dir, frozen)
        report["variants"][str(count)] = {
            "known_pose_anchor_indices": anchors,
            "estimated_pose_frames": used - len(anchors),
            "unresolved_pose_frames": 96 - used,
            "withheld_gt_translation_error_median_m": float(np.median(errors)) if errors else None,
            "withheld_gt_translation_error_max_m": float(np.max(errors)) if errors else None,
            "mesh_sha256": file_sha256(mesh),
            "faces": len(f), "components": face_components(len(v), f),
            "five_views": views, "fusion": fusion}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Frozen #99 original RGB association artifact")
    parser.add_argument("--room96", required=True, help="Frozen #106 96 RGB and measured Mesh artifact")
    parser.add_argument("--output", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--archive", default=None)
    parser.add_argument("--levels", type=int, nargs="+", default=[13, 5, 1])\n    parser.add_argument("--max-gap", type=int, default=6)
    args = parser.parse_args()
    result = run(args.source, args.room96, args.output, args.work, args.archive, args.levels, args.max_gap)
    print(json.dumps({"pose_graph": result["pose_graph"],
                      "variants": {k: {"faces": v["faces"],
                                       "estimated": v["estimated_pose_frames"],
                                       "unresolved": v["unresolved_pose_frames"],
                                       "median_translation_error_m": v["withheld_gt_translation_error_median_m"]}
                                   for k, v in result["variants"].items()}}, indent=2))
