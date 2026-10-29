"""Measured RGB-D one-anchor pose refinement using Open3D dense ICP + pose graph.

Input: the frozen #110 96-frame *estimated* RGB-D recording, with one known
metric pose and 95 estimated poses. No withheld ground-truth pose or prior Mesh
is used for the candidate. RGB-D SIFT-PnP supplies pair hypotheses; measured
depth cloud overlap supplies an independent dense constraint. CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

import tum_room_rgbd_global_loop as loops
import tum_room_rgbd_pose_ladder as base
from sparse_open3d_fusion import fuse_sparse
from tum_room_da2_global_scale_replay import checked_replay, five_view_comparison, mesh_glb
from echo4d.model import DataError, file_sha256
from echo4d.holdout import read_echo_ply
from tum_highres_tsdf import face_components


def cloud_from_measured_depth(depth, intrinsics, stride=3, voxel=0.055):
    """Unproject observed metric depth, not inferred or ground-truth geometry."""
    h, w = depth.shape
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]
    good = np.isfinite(z) & (z > 0.25) & (z < 4.0)
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    x = (xx[good] - intrinsics["cx"]) * z[good] / fx
    y = (yy[good] - intrinsics["cy"]) * z[good] / fy
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.column_stack((x, y, z[good])))
    cloud = cloud.voxel_down_sample(voxel)
    if len(cloud.points) >= 30:
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=25))
    return cloud


def accept_icp(initial, result, min_fitness=0.18, max_rmse=0.065):
    """Reject weak planar/occluded matches before global optimization."""
    if result.fitness < min_fitness or not np.isfinite(result.inlier_rmse):
        return False
    if result.inlier_rmse > max_rmse:
        return False
    delta = result.transformation @ np.linalg.inv(initial)
    shift = float(np.linalg.norm(delta[:3, 3]))
    degrees = float(np.linalg.norm(
        Rotation.from_matrix(delta[:3, :3]).as_rotvec()) * 180 / np.pi)
    return np.isfinite(shift) and shift <= 0.18 and degrees <= 12.0


def refine_poses(graph, initial, clouds):
    """Keep all 96 graph nodes; dense constraints improve only verified edges."""
    n = len(initial)
    if len(graph) != n or len(clouds) != n:
        raise ValueError("Pose graph, cameras and depth clouds must align")
    pg = o3d.pipelines.registration.PoseGraph()
    for pose in initial:
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose.copy()))
    tried, accepted, fallback = 0, 0, 0
    samples = []
    for i in range(n):
        for j, measured, cost in graph[i]:
            if j <= i:
                continue
            tried += 1
            use_dense = False
            transform = measured
            if len(clouds[i].points) >= 30 and len(clouds[j].points) >= 30:
                result = o3d.pipelines.registration.registration_icp(
                    clouds[i], clouds[j], 0.12, measured,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=25))
                if accept_icp(measured, result):
                    transform = result.transformation
                    use_dense = True
                    accepted += 1
                    samples.append([i, j, float(result.fitness),
                                    float(result.inlier_rmse)])
            if not use_dense:
                fallback += 1
            # Normalized information prevents one densely textured view from
            # overwhelming all other observations. Weak PnP edges keep the
            # one-anchor graph connected if ICP has insufficient overlap.
            info = np.eye(6) * (1.0 if use_dense else 0.025)
            pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                i, j, transform, info, uncertain=bool(abs(i - j) > 6)))
    if not pg.edges:
        raise DataError("No observed RGB-D graph edges; cannot optimize")
    before = [node.pose.copy() for node in pg.nodes]
    o3d.pipelines.registration.global_optimization(
        pg,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=0.12,
            edge_prune_threshold=0.25,
            preference_loop_closure=0.15,
            reference_node=0))
    optimized = [node.pose.copy() for node in pg.nodes]
    # Exact declared anchor is immutable; no extra GT pose enters optimization.
    optimized[0] = initial[0].copy()
    corrections = [float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
                   for a, b in zip(optimized, initial)]
    if any(not np.isfinite(p).all() for p in optimized):
        raise DataError("Non-finite optimized camera poses")
    if max(corrections) > 1.0:
        raise DataError("Pose graph diverged >1m; do not publish candidate")
    return optimized, {
        "rgbd_edges_attempted": tried,
        "dense_icp_edges_accepted": accepted,
        "weak_pnp_fallback_edges": fallback,
        "max_camera_correction_m": max(corrections),
        "median_camera_correction_m": float(np.median(corrections)),
        "sample_dense_edges": samples[:12],
        "optimizer": "Open3D 0.19 global pose graph LM, one fixed anchor",
        "all_depths": "96 independently measured sensor depth frames",
    }


def run(old, source, output):
    old, source, output = map(Path, (old, source, output))
    if output.exists():
        raise DataError("Refusing to overwrite a previous real experiment")
    recording = old / "input-poses-1/recording.json"
    if not recording.is_file():
        raise FileNotFoundError(recording)
    manifest = json.loads(recording.read_text())
    frames = manifest["frames"]
    if len(frames) != 96:
        raise DataError("Expected 96 already-estimated real RGB-D frames")
    if sum(f.get("pose_provenance") == "known_metric_anchor" for f in frames) != 1:
        raise DataError("Only one declared known metric pose is allowed")
    source_dir = recording.parent
    colors, depths = [], []
    for frame in frames:
        im = cv2.imread(str(source_dir / frame["color"]))
        if im is None:
            raise FileNotFoundError(source_dir / frame["color"])
        colors.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        depths.append(np.load(source_dir / frame["depth"], allow_pickle=False))
    initial = [base.camera_to_world(f) for f in frames]
    graph, retrieval = loops.retrieval_pose_graph(
        colors, depths, base.camera_matrix(manifest["intrinsics"]), max_gap=6)
    clouds = [cloud_from_measured_depth(d, manifest["intrinsics"]) for d in depths]
    optimized, refinement = refine_poses(graph, initial, clouds)
    output.mkdir(parents=True)
    candidate = output / "input-poses-1"
    provenance = [{"kind": ("known_metric_anchor" if i == 0
                             else "rgbd_dense_icp_pose_graph")}
                  for i in range(96)]
    base.prepare_candidate(source_dir, manifest, depths, optimized, provenance, candidate)
    mesh_dir = output / "mesh-poses-1"
    fusion = fuse_sparse(candidate / "recording.json", mesh_dir,
                         voxel_m=0.03, truncation_m=0.09,
                         max_depth_m=4.0, pixel_stride=2)
    new_mesh = mesh_dir / "mesh.ply"
    raw = new_mesh.read_bytes()
    label = b"comment Open3D sparse TSDF; measured RGB-D; known poses; no inferred depth"
    replacement = (b"comment Open3D TSDF; measured sensor depth; one known pose; "
                   b"95 RGB-D ICP graph poses; no prior Mesh")
    if raw.count(label) != 1:
        raise DataError("Unexpected TSDF provenance comment")
    new_mesh.write_bytes(raw.replace(label, replacement, 1))
    fusion["mesh_sha256"] = file_sha256(new_mesh)
    fusion["geometry_source"] = "96 measured depth maps, one known pose, 95 RGB-D graph estimates"
    (mesh_dir / "report.json").write_text(json.dumps(fusion, indent=2) + "\n")
    mesh_glb(new_mesh, output / "measured-rgbd-poses-1-icp.glb")

    # Only AFTER fusion: withheld GT for diagnostic; previous Mesh for A/B.
    reference = json.loads((old / "input-poses-96/recording.json").read_text())
    known = [base.camera_to_world(f) for f in reference["frames"]]
    def errors(poses):
        a = [np.linalg.norm(p[:3, 3] - k[:3, 3])
             for p, k in zip(poses, known)][1:]
        return {"median_m": float(np.median(a)),
                "max_m": float(np.max(a)),
                "p90_m": float(np.percentile(a, 90))}
    old_mesh = old / "mesh-poses-1/mesh.ply"
    frozen, replay, _ = checked_replay(source)
    visual = output / "visual-old-new-3plus2"
    visual.mkdir()
    views = five_view_comparison(source, replay, old_mesh, new_mesh, visual, frozen)
    loops.relabel_visual_board(visual / "source-old-new-3plus2.png",
                               "OLD: one anchor / RGB-D SIFT global loops",
                               "NEW: one anchor / dense ICP graph")
    v, f = read_echo_ply(new_mesh)
    report = {
        "date_utc": "2026-10-29",
        "experiment": "TUM fr1_room one-anchor dense RGB-D ICP pose refinement",
        "retained": ["96 real RGB", "96 measured metric sensor depths",
                     "calibration", "one known metric camera pose"],
        "removed": ["95 ground-truth camera poses", "prior Mesh reconstruction input"],
        "prior_mesh_used_only_for_visual_comparison": True,
        "ground_truth_only_after_candidate_fusion": True,
        "metric_scale": "measured sensor depth, NOT RGB-only",
        "old_mesh_sha256": file_sha256(old_mesh),
        "new_mesh_sha256": file_sha256(new_mesh),
        "old_camera_error": errors(initial),
        "new_camera_error": errors(optimized),
        "new_faces": len(f),
        "new_components": face_components(len(v), f),
        "retrieval": retrieval,
        "refinement": refinement,
        "fusion": fusion,
        "five_views": views,
        "visual_acceptance": "UNREVIEWED; #74 not PASS",
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, help="Frozen real #110 Mesh and 96 RGB-D frames")
    parser.add_argument("--source", required=True, help="Frozen #99 original real RGB artifact")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.old, args.source, args.output)
    print(json.dumps({"old_camera_error": result["old_camera_error"],
                      "new_camera_error": result["new_camera_error"],
                      "refinement": result["refinement"],
                      "new_faces": result["new_faces"],
                      "visual_acceptance": result["visual_acceptance"]}, indent=2))
