"""Occlusion-aware cross-view rejection of known-pose RGB-predicted depth.

A disagreement counts against a candidate only if another predicted camera
observes a *farther* surface where the candidate should have been visible.
A closer surface can occlude it and is not contradictory evidence.
No sensor depth, prior Mesh, or model inference is accessed.
"""
from __future__ import annotations

import numpy as np


def _sample_bilinear_valid(depth: np.ndarray, u: np.ndarray, v: np.ndarray):
    """Interpolate only with four valid pixels; never bleed across holes."""
    h, w = depth.shape
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    good = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < w) & (y0 + 1 < h)
    out = np.zeros(u.shape, dtype=np.float32)
    if not np.any(good):
        return out, good
    indices = np.flatnonzero(good)
    x, y = x0[indices], y0[indices]
    z00, z10 = depth[y, x], depth[y, x+1]
    z01, z11 = depth[y+1, x], depth[y+1, x+1]
    valid = ((z00 > 0) & (z10 > 0) & (z01 > 0) & (z11 > 0)
             & np.isfinite(z00) & np.isfinite(z10)
             & np.isfinite(z01) & np.isfinite(z11))
    good[indices] = valid
    dx, dy = u[indices] - x, v[indices] - y
    out[indices] = ((1-dx)*(1-dy)*z00 + dx*(1-dy)*z10
                    + (1-dx)*dy*z01 + dx*dy*z11)
    return out, good


def filter_reference(depth: np.ndarray, reference_pose: np.ndarray,
                     neighbors: list[tuple[np.ndarray, np.ndarray]], K: np.ndarray,
                     *, absolute_tolerance_m: float = .07,
                     relative_tolerance: float = .045,
                     min_front_conflicts: int = 2) -> tuple[np.ndarray, dict]:
    """Remove candidates contradicted by >=2 independent visible farther depths.

    Camera poses are camera-to-world, depths are metric camera Z.
    Neighbor selection must enforce distinct camera translations.
    """
    z = np.asarray(depth, dtype=np.float32)
    pose = np.asarray(reference_pose, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    if (z.ndim != 2 or not np.isfinite(z).all() or (z < 0).any()
            or pose.shape != (4, 4) or not np.isfinite(pose).all()
            or K.shape != (3, 3) or not np.isfinite(K).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0 or min_front_conflicts < 2):
        raise ValueError("Expected finite metric Z, valid camera poses/intrinsics")
    ys, xs = np.nonzero(z > 0)
    n = len(xs)
    result = z.copy()
    if n == 0:
        return result, {"valid_before": 0, "valid_after": 0, "removed": 0,
                        "supported_pixels": 0, "front_conflicted_pixels": 0,
                        "occluded_observations": 0, "comparable_observations": 0}
    xyz = np.stack(((xs - K[0, 2]) / K[0, 0] * z[ys, xs],
                    (ys - K[1, 2]) / K[1, 1] * z[ys, xs], z[ys, xs]), axis=0)
    support = np.zeros(n, dtype=np.uint8)
    front = np.zeros(n, dtype=np.uint8)
    occluded = 0
    comparable = 0
    for neighbor_z, neighbor_pose in neighbors:
        nz = np.asarray(neighbor_z, dtype=np.float32)
        np_pose = np.asarray(neighbor_pose, dtype=np.float64)
        if (nz.shape != z.shape or not np.isfinite(nz).all() or (nz < 0).any()
                or np_pose.shape != (4, 4) or not np.isfinite(np_pose).all()):
            raise ValueError("Neighbor depth or camera pose invalid")
        transform = np.linalg.solve(np_pose, pose)
        p = transform[:3, :3] @ xyz + transform[:3, 3:4]
        positive = p[2] > .2
        safe_z = np.maximum(p[2], 1e-6)
        u = K[0, 0] * p[0] / safe_z + K[0, 2]
        v = K[1, 1] * p[1] / safe_z + K[1, 2]
        sampled, valid = _sample_bilinear_valid(nz, u, v)
        valid &= positive
        tolerance = absolute_tolerance_m + relative_tolerance * np.maximum(safe_z, sampled)
        agreeing = valid & (np.abs(sampled - p[2]) <= tolerance)
        contradicted = valid & (sampled > p[2] + tolerance)
        support += agreeing.astype(np.uint8)
        front += contradicted.astype(np.uint8)
        occluded += int(np.count_nonzero(valid & (sampled < p[2] - tolerance)))
        comparable += int(np.count_nonzero(valid))
    reject = (support == 0) & (front >= min_front_conflicts)
    result[ys[reject], xs[reject]] = 0
    return result, {
        "valid_before": int(n), "valid_after": int(n - np.count_nonzero(reject)),
        "removed": int(np.count_nonzero(reject)),
        "supported_pixels": int(np.count_nonzero(support)),
        "front_conflicted_pixels": int(np.count_nonzero(front)),
        "occluded_observations": occluded,
        "comparable_observations": comparable,
        "neighbors_considered": len(neighbors),
        "rule": "zero agreements and >=2 farther visible neighbor depths; closer depth treated as occlusion",
    }


def filter_all(depths: list[np.ndarray], poses: list[np.ndarray], K: np.ndarray,
               neighbor_selector, *, max_neighbors: int = 6):
    """Compare against immutable original maps; no cascade of filtered depths."""
    if len(depths) != len(poses):
        raise ValueError("Depth and camera cohort mismatch")
    outputs, receipts = [], []
    for i, depth in enumerate(depths):
        ids = neighbor_selector(poses, i, limit=max_neighbors)
        ids = [j for j in ids if j != i and np.any(depths[j] > 0)]
        filtered, receipt = filter_reference(
            depth, poses[i], [(depths[j], poses[j]) for j in ids], K)
        outputs.append(filtered)
        receipts.append({"position": i, "neighbors": ids, **receipt})
    return outputs, receipts
