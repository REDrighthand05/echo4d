"""Robust global PnP pose averaging on frozen one-anchor measured RGB-D room.

A distinct alternative to the failed dense ICP experiment: use verified RGB-D
SIFT/PnP pair measurements and robust sparse nonlinear SE(3) least squares.
No withheld camera poses, previous Mesh, or predicted depth enter estimation.
The existing fusion and five-view comparison are reused without reimplementation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

import tum_room_rgbd_dense_icp as previous
from echo4d.model import DataError, file_sha256
import tum_room_rgbd_global_loop as loops


def robust_refine_poses(graph, initial, unused_clouds=None):
    """One fixed metric anchor; robust graph averaging without GT or ICP.

    Graph transforms are camera_i -> camera_j; poses are camera -> world.
    Soft-L1 acts on *each* measured residual, not on posthoc GT pose error.
    The weak zero-centered correction prior keeps unobservable drift bounded.
    """
    n = len(initial)
    if len(graph) != n or n < 2 or any(p is None for p in initial):
        raise DataError("Expected a fully connected, initialized pose graph")
    edges = []
    for i, neighbors in enumerate(graph):
        for j, T, cost in neighbors:
            if j <= i:
                continue
            T = np.asarray(T, dtype=np.float64)
            if T.shape != (4, 4) or not np.isfinite(T).all():
                continue
            if not np.isfinite(cost) or cost <= 0:
                continue
            edges.append((i, j, T[:3, :3].copy(), T[:3, 3].copy(),
                          1.0 / np.sqrt(max(float(cost), 1.0))))
    if len(edges) < n - 1:
        raise DataError("Insufficient observed RGB-D constraints")

    initial = [np.asarray(p, dtype=np.float64) for p in initial]
    rotations = [p[:3, :3].copy() for p in initial]
    translations = [p[:3, 3].copy() for p in initial]
    num_vars = 6 * (n - 1)
    num_residuals = 6 * len(edges) + num_vars

    # A sparse finite-difference Jacobian avoids an impractical dense
    # (6N)x(6E) solve; each pair depends only on its two camera poses.
    sparsity = lil_matrix((num_residuals, num_vars), dtype=np.int8)
    for k, (i, j, _, _, _) in enumerate(edges):
        for index in (i, j):
            if index > 0:
                sparsity[6 * k:6 * k + 6, 6 * (index - 1):6 * index] = 1
    sparsity[6 * len(edges):, :] = lil_matrix(
        np.eye(num_vars, dtype=np.int8))
    sparsity = sparsity.tocsr()

    def unpack(x):
        R = [rotations[0]]
        t = [translations[0]]
        for i in range(1, n):
            d = x[6 * (i - 1):6 * i]
            R.append(Rotation.from_rotvec(d[:3]).as_matrix() @ rotations[i])
            t.append(translations[i] + d[3:])
        return R, t

    def residuals(x):
        R, t = unpack(x)
        residual = np.empty(num_residuals, dtype=np.float64)
        for k, (i, j, Tij_R, Tij_t, weight) in enumerate(edges):
            relative_R = R[j].T @ R[i]
            relative_t = R[j].T @ (t[i] - t[j])
            rotation_error = Rotation.from_matrix(
                Tij_R.T @ relative_R).as_rotvec()
            translation_error = Tij_R.T @ (relative_t - Tij_t)
            residual[6 * k:6 * k + 3] = weight * translation_error
            residual[6 * k + 3:6 * k + 6] = weight * 0.5 * rotation_error
        # Weak soft prior, never an undeclared second ground-truth anchor.
        residual[6 * len(edges):] = 0.06 * x
        return residual

    x0 = np.zeros(num_vars, dtype=np.float64)
    before = float(np.linalg.norm(residuals(x0)[:6 * len(edges)]))
    fit = least_squares(
        residuals, x0, jac="2-point", jac_sparsity=sparsity,
        loss="soft_l1", f_scale=0.08, method="trf", tr_solver="lsmr",
        max_nfev=35, ftol=1e-5, xtol=1e-5, gtol=1e-5)
    R, t = unpack(fit.x)
    poses = []
    for i in range(n):
        pose = np.eye(4)
        pose[:3, :3] = R[i]
        pose[:3, 3] = t[i]
        poses.append(pose)
    poses[0] = initial[0].copy()
    corrections = [float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
                   for a, b in zip(poses, initial)]
    if any(not np.isfinite(p).all() for p in poses):
        raise DataError("Robust graph generated non-finite poses")
    if max(corrections) > 1.0:
        raise DataError("Robust graph diverged over one meter")
    return poses, {
        "solver": "scipy sparse robust soft_l1 SE3 pose averaging; CPU",
        "measured_rgbd_pnp_edges": len(edges),
        "known_metric_pose_anchors": 1,
        "iterations": int(fit.nfev),
        "solver_success": bool(fit.success),
        "solver_message": str(fit.message),
        "initial_observed_edge_residual_l2": before,
        "final_observed_edge_residual_l2":
            float(np.linalg.norm(residuals(fit.x)[:6 * len(edges)])),
        "median_camera_correction_m": float(np.median(corrections)),
        "max_camera_correction_m": max(corrections),
        "no_dense_icp": True,
        "no_gt_pose_or_prior_mesh_in_solver": True,
    }


def run(old, source, output):
    """Reuse the proven actual Open3D TSDF, GLB and 3+2 renderer."""
    output = Path(output)
    original = previous.refine_poses
    previous.refine_poses = robust_refine_poses
    try:
        report = previous.run(old, source, output)
    finally:
        previous.refine_poses = original

    # The shared executor was written for ICP; correct every provenance
    # field and visual caption, without changing any geometry.
    mesh = output / "mesh-poses-1/mesh.ply"
    raw = mesh.read_bytes()
    old_label = b"95 RGB-D ICP graph poses"
    if raw.count(old_label) != 1:
        raise DataError("Cannot locate inherited PLY provenance label")
    mesh.write_bytes(raw.replace(old_label, b"95 robust RGB-D PnP graph poses", 1))
    candidate = output / "input-poses-1/recording.json"
    manifest = json.loads(candidate.read_text())
    for i, frame in enumerate(manifest["frames"]):
        if i:
            assert frame["pose_provenance"] == "rgbd_dense_icp_pose_graph"
            frame["pose_provenance"] = "rgbd_robust_pnp_pose_graph"
    manifest["name"] = "TUM room one-anchor robust RGB-D PnP pose graph"
    candidate.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    old_glb = output / "measured-rgbd-poses-1-icp.glb"
    new_glb = output / "measured-rgbd-poses-1-robust.glb"
    old_glb.rename(new_glb)
    loops.relabel_visual_board(
        output / "visual-old-new-3plus2/source-old-new-3plus2.png",
        "OLD: one anchor / RGB-D SIFT global loops",
        "NEW: one anchor / robust PnP graph averaging")
    report["experiment"] = "TUM fr1_room robust RGB-D PnP global pose averaging"
    report["new_mesh_sha256"] = file_sha256(mesh)
    report["fusion"]["mesh_sha256"] = file_sha256(mesh)
    report["solver_input_boundary"] = (
        "96 real RGB, 96 measured metric depth, one known pose; "
        "95 poses estimated from RGB-D PnP, no prior Mesh or withheld GT")
    report["visual_acceptance"] = "UNREVIEWED; #74 not PASS"
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n")
    fusion_report = output / "mesh-poses-1/report.json"
    fusion = json.loads(fusion_report.read_text())
    fusion["mesh_sha256"] = file_sha256(mesh)
    fusion["geometry_source"] = report["solver_input_boundary"]
    fusion_report.write_text(json.dumps(fusion, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = run(a.old, a.source, a.output)
    print(json.dumps({
        "old_camera_error": result["old_camera_error"],
        "new_camera_error": result["new_camera_error"],
        "refinement": result["refinement"],
        "new_faces": result["new_faces"],
        "visual_acceptance": result["visual_acceptance"],
    }, indent=2))
