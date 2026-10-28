"""RGB-only cross-view surface consensus for known-pose DA2 depth maps.

Only DA2-predicted, RGB-calibrated depths and metric camera poses enter here.
Sensor depth and the measured high-information mesh are not read.
No pixels are removed: add independently supported holes and gently reconcile
overlapping predictions; never claim recovered occluded surfaces as ground truth.
"""
from __future__ import annotations

import numpy as np


def reproject_z_color(depth, color, source_c2w, target_c2w, K):
    """Forward splat source pixels with a nearest-Z visibility buffer.

    Returns camera-Z (not Euclidean range) and the source RGB of each winning
    pixel. Unobserved pixels are zero. Pose convention is camera-to-world.
    """
    depth = np.asarray(depth, dtype=np.float32)
    color = np.asarray(color, dtype=np.uint8)
    K = np.asarray(K, dtype=np.float64)
    src = np.asarray(source_c2w, dtype=np.float64)
    dst = np.asarray(target_c2w, dtype=np.float64)
    if (depth.ndim != 2 or color.shape != (*depth.shape, 3)
            or K.shape != (3, 3) or src.shape != (4, 4)
            or dst.shape != (4, 4) or not np.isfinite(depth).all()
            or (depth < 0).any() or not np.isfinite(K).all()
            or not np.isfinite(src).all() or not np.isfinite(dst).all()
            or abs(np.linalg.det(K)) < 1e-10):
        raise ValueError("Expected nonnegative finite camera-Z, RGB, K and camera-to-world poses")
    h, w = depth.shape
    yy, xx = np.nonzero((depth > .2) & (depth < 6.))
    z = depth[yy, xx].astype(np.float64)
    out = np.zeros((h, w), np.float32)
    out_rgb = np.zeros((h, w, 3), np.uint8)
    if not len(z):
        return out, out_rgb
    rays = np.linalg.inv(K) @ np.stack((xx, yy, np.ones_like(xx)), axis=0)
    xyz = rays * z
    transform = np.linalg.inv(dst) @ src
    target_xyz = transform[:3, :3] @ xyz + transform[:3, 3:4]
    tz = target_xyz[2]
    good = tz > .2
    projected = K @ target_xyz
    u = np.zeros_like(tz)
    v = np.zeros_like(tz)
    np.divide(projected[0], projected[2], out=u, where=good)
    np.divide(projected[1], projected[2], out=v, where=good)
    good &= np.isfinite(u) & np.isfinite(v) & (u >= -.5) & (u < w-.5) & (v >= -.5) & (v < h-.5)
    if not good.any():
        return out, out_rgb
    u = np.floor(u[good] + .5).astype(np.int64)
    v = np.floor(v[good] + .5).astype(np.int64)
    z = tz[good].astype(np.float32)
    rgb = color[yy[good], xx[good]]
    flat = v * w + u
    # Sort by pixel, then ascending camera-Z; first is nearest visible source.
    order = np.lexsort((z, flat))
    _, first = np.unique(flat[order], return_index=True)
    chosen = order[first]
    out.flat[flat[chosen]] = z[chosen]
    out_rgb.reshape(-1, 3)[flat[chosen]] = rgb[chosen]
    return out, out_rgb


def neighbors_for(poses, index, limit=6):
    """Select nearby distinct camera centres with overlapping optical axes."""
    here = poses[index]
    options = []
    for j, pose in enumerate(poses):
        if j == index:
            continue
        baseline = float(np.linalg.norm(pose[:3, 3] - here[:3, 3]))
        facing = float(np.dot(pose[:3, 2], here[:3, 2]))
        if .025 < baseline < .8 and facing > .82:
            options.append((baseline + .2 * (1-facing), j))
    options.sort()
    return [j for _, j in options[:limit]]


def reconcile_depths(depths, rgbs, poses, K, *, tolerance_m=.10, color_l1=30.):
    """Consensus of >=2 independent source-camera predictions.

    Two source reprojected Z values must agree within tolerance_m + 4% of
    range and their projected RGB must resemble the target RGB. A missing
    target depth can then be filled; a measured-in-RGB target depth is only
    blended when already consistent. Disagreements stay unchanged.
    """
    n = len(depths)
    if not (n == len(rgbs) == len(poses)):
        raise ValueError("Depth/RGB/pose frame counts differ")
    if not n:
        raise ValueError("No depth frames")
    shape = np.asarray(depths[0]).shape
    if any(np.asarray(z).shape != shape for z in depths):
        raise ValueError("Depth maps must have the same shape")
    results, receipts = [], []
    for i in range(n):
        target = np.asarray(depths[i], np.float32)
        colors = np.asarray(rgbs[i], np.uint8)
        peers = neighbors_for(poses, i)
        reprojections = []
        for j in peers:
            z, rgb = reproject_z_color(depths[j], rgbs[j], poses[j], poses[i], K)
            photometric = np.abs(rgb.astype(np.int16) - colors.astype(np.int16)).mean(axis=2)
            z[(z <= 0) | (photometric > color_l1)] = 0
            reprojections.append(z)
        result = target.copy()
        filled = blended = 0
        if len(reprojections) >= 2:
            stack = np.stack(reprojections)
            valid = stack > 0
            support = valid.sum(axis=0)
            # A source pair is a witness only if their projected depths agree.
            zmin = np.where(valid, stack, np.inf).min(axis=0)
            zmax = np.where(valid, stack, -np.inf).max(axis=0)
            consistent = ((support >= 2) & np.isfinite(zmin)
                          & (zmax - zmin <= tolerance_m + .04 * zmin))
            median = np.median(np.where(valid, stack, np.nan), axis=0) if False else None
            # Median of available source estimates; zero means missing.
            ordered = np.sort(np.where(valid, stack, np.inf), axis=0)
            # First two closest surfaces are sufficient; do not let an
            # unrelated third farther surface fill a background hole.
            agreement = ((support >= 2) & np.isfinite(ordered[1])
                         & (ordered[1] - ordered[0] <= tolerance_m + .04 * ordered[0]))
            candidate = .5 * (ordered[0] + ordered[1])
            # For >=3 witnesses, require majority to avoid one accidental pair.
            if len(reprojections) >= 3:
                third = ordered[2]
                agreement &= (support == 2) | ((third - ordered[0]) <= tolerance_m + .04 * ordered[0])
            agreement &= consistent
            holes = agreement & (target == 0) & np.isfinite(candidate)
            compatible = (agreement & (target > 0)
                          & (np.abs(candidate - target) <= tolerance_m + .04 * target))
            result[holes] = candidate[holes]
            result[compatible] = (.75 * target[compatible] + .25 * candidate[compatible])
            filled = int(holes.sum())
            blended = int(compatible.sum())
        results.append(result)
        receipts.append({"frame": i, "source_neighbors": peers,
                         "filled_rgb_supported_pixels": filled,
                         "blended_consistent_pixels": blended,
                         "original_pixels": int(np.count_nonzero(target)),
                         "output_pixels": int(np.count_nonzero(result))})
    return results, receipts
