"""Known-pose RGB-only depth agreement for Stanford Lounge.

All depths passed here must be predicted from RGB, never withheld sensor depths.
A target RGB prediction is accepted only where a *different camera center*
independently predicts the same world surface with sufficient parallax.
This rejects one-view hallucinated sheets without claiming unseen space exists.
"""
from __future__ import annotations

import numpy as np


def corroborate_rgb_depths(depth_by_id, poses, camera, *, min_baseline_m=.16,
                           min_parallax_deg=3., abs_tolerance_m=.12,
                           rel_tolerance=.07, min_other_views=1):
    """Return masks and per-frame receipts; target + min_other_views agree.

    Pixel centers are reprojected through known camera-to-world poses. Other
    RGB-predicted depths are sampled at nearest pixels. A source only counts
    if its camera *center* is separated and its viewing ray subtends the
    required parallax at that surface. Missing/occluded predictions abstain.
    The comparison uses neither a prior Mesh nor target sensor depth.
    """
    if min_other_views < 1 or min_baseline_m <= 0 or min_parallax_deg <= 0:
        raise ValueError("Require independent camera centers and other RGB support")
    h, w = camera.height, camera.width
    fx, fy, cx, cy = (float(camera.fx), float(camera.fy),
                      float(camera.cx), float(camera.cy))
    if min(fx, fy) <= 0:
        raise ValueError("Invalid camera intrinsics")
    ids = sorted(depth_by_id)
    for i in ids:
        d = np.asarray(depth_by_id[i])
        if d.shape != (h, w):
            raise ValueError(f"RGB prediction {i} dimensions mismatch")
        if np.asarray(poses[i]).shape != (4, 4):
            raise ValueError("Invalid camera-to-world pose")
    yy, xx = np.indices((h, w))
    rays = np.stack(((xx-cx)/fx, (yy-cy)/fy, np.ones((h, w))), axis=-1)
    masks, receipts = {}, {}
    max_cos = np.cos(np.deg2rad(min_parallax_deg))
    for i in ids:
        depth = np.asarray(depth_by_id[i], np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        xyz = (rays * depth[..., None]).reshape(-1, 3)
        pose = np.asarray(poses[i], np.float64)
        world = xyz @ pose[:3, :3].T + pose[:3, 3]
        origin = pose[:3, 3]
        target_ray = world - origin
        target_norm = np.linalg.norm(target_ray, axis=1)
        votes = np.zeros(h*w, np.uint8)
        eligible_pairs = 0
        for j in ids:
            if j == i:
                continue
            other = np.asarray(poses[j], np.float64)
            other_origin = other[:3, 3]
            if np.linalg.norm(origin-other_origin) < min_baseline_m:
                continue
            eligible_pairs += 1
            # R^T(world - t): world -> other camera coordinates.
            xyz_other = (world-other_origin) @ other[:3, :3]
            z = xyz_other[:, 2]
            in_front = z > 0
            safe_z = np.where(in_front, z, 1.)
            u = np.rint(xyz_other[:, 0]*fx/safe_z + cx).astype(np.int32)
            v = np.rint(xyz_other[:, 1]*fy/safe_z + cy).astype(np.int32)
            inside = (in_front & (u >= 0) & (u < w) &
                      (v >= 0) & (v < h) & valid.ravel())
            idx = np.flatnonzero(inside)
            if not len(idx):
                continue
            other_depth = np.asarray(depth_by_id[j], np.float32)[v[idx], u[idx]]
            other_ray = world[idx] - other_origin
            other_norm = np.linalg.norm(other_ray, axis=1)
            cos_angle = np.sum(target_ray[idx]*other_ray, axis=1) / np.maximum(
                target_norm[idx]*other_norm, 1e-8)
            independent = cos_angle <= max_cos
            agreed = (independent & np.isfinite(other_depth) & (other_depth > 0)
                      & (np.abs(other_depth-z[idx]) <= np.maximum(
                          abs_tolerance_m, rel_tolerance*z[idx])))
            # Saturate votes to avoid overflow on future larger camera cohorts.
            votes[idx[agreed]] = np.minimum(
                votes[idx[agreed]].astype(np.uint16)+1, 255).astype(np.uint8)
        accepted = valid.ravel() & (votes >= min_other_views)
        masks[i] = accepted.reshape(h, w)
        receipts[i] = {
            "predicted_pixels": int(valid.sum()),
            "crossview_confirmed_pixels": int(accepted.sum()),
            "crossview_confirmed_fraction": float(accepted.sum()/max(valid.sum(), 1)),
            "eligible_independent_cameras": eligible_pairs,
            "minimum_other_rgb_views": min_other_views,
            "minimum_baseline_m": min_baseline_m,
            "minimum_parallax_deg": min_parallax_deg,
        }
    return masks, receipts
