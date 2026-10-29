"""Synchronize per-frame DA2 metric depth scale from known-pose training RGB.

No sensor depth or prior Mesh is read. The initial metric scale comes from the
existing known-pose RGB ZNCC calibration, not from the relative-depth network.
"""
from __future__ import annotations
import numpy as np
import cv2


def _pairs(poses, limit=5):
    centers = np.asarray([p[:3, 3] for p in poses])
    axes = np.asarray([p[:3, 2] for p in poses])
    out = set()
    for i in range(len(poses)):
        options = []
        for j in range(len(poses)):
            if i == j:
                continue
            baseline = float(np.linalg.norm(centers[i] - centers[j]))
            facing = float(axes[i] @ axes[j])
            if .035 <= baseline <= .75 and facing >= .80:
                options.append((baseline + .15 * (1-facing), j))
        for _, j in sorted(options)[:limit]:
            out.add((min(i, j), max(i, j)))
    return sorted(out)


def _sample_pair(i, j, depths, rgbs, poses, K, stride=4, max_points=1800):
    """Return target Z, rotated source Z and metric translation Z for RGB matches."""
    a, b = np.asarray(depths[i]), np.asarray(depths[j])
    ca, cb = np.asarray(rgbs[i]), np.asarray(rgbs[j])
    h, w = a.shape
    if b.shape != (h, w) or ca.shape != (h, w, 3) or cb.shape != (h, w, 3):
        raise ValueError("RGB/depth dimensions differ")
    ys, xs = np.mgrid[stride//2:h:stride, stride//2:w:stride]
    yy, xx = ys.ravel(), xs.ravel()
    z = a[yy, xx].astype(np.float64)
    valid = np.isfinite(z) & (z >= .45) & (z <= 4.)
    if not valid.any():
        return np.empty((0, 3))
    xx, yy, z = xx[valid], yy[valid], z[valid]
    ray = np.stack(((xx-K[0, 2])/K[0, 0],
                    (yy-K[1, 2])/K[1, 1], np.ones(len(z))))
    rel = np.linalg.inv(poses[j]) @ poses[i]
    rotated = rel[:3, :3] @ (ray * z)
    points = rotated + rel[:3, 3:4]
    tz = points[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (K[0, 0]*points[0]/tz + K[0, 2]).astype(np.float32)
        v = (K[1, 1]*points[1]/tz + K[1, 2]).astype(np.float32)
    inside = (tz > .45) & (tz < 4.) & (u >= 2) & (u < w-3) & (v >= 2) & (v < h-3)
    if not inside.any():
        return np.empty((0, 3))
    u, v, tz, rz, xx, yy = (u[inside], v[inside], tz,
                            rotated[2, inside], xx[inside], yy[inside])
    mx, my = u.reshape(-1, 1), v.reshape(-1, 1)
    zb = cv2.remap(b.astype(np.float32), mx, my, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT).ravel()
    min_z = cv2.erode(b.astype(np.float32), np.ones((3, 3), np.uint8))
    max_z = cv2.dilate(b.astype(np.float32), np.ones((3, 3), np.uint8))
    spread = (cv2.remap(max_z, mx, my, cv2.INTER_NEAREST).ravel() -
              cv2.remap(min_z, mx, my, cv2.INTER_NEAREST).ravel())
    rgbb = cv2.remap(cb, mx, my, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT).reshape(-1, 3)
    color_diff = np.abs(ca[yy, xx].astype(np.int16) -
                        rgbb.astype(np.int16)).mean(axis=1)
    good = (np.isfinite(zb) & (zb > .45) & (zb < 4.) &
            (spread < .14) & (color_diff < 28.) &
            (np.abs(zb-tz) < np.maximum(.38, .24*tz)))
    if good.sum() < 50:
        return np.empty((0, 3))
    rows = np.column_stack((zb[good], rz[good], (tz-rz)[good]))
    if len(rows) > max_points:
        rows = rows[np.linspace(0, len(rows)-1, max_points, dtype=int)]
    return rows


def synchronize_scales(depths, rgbs, poses, K, *, prior_strength=120.,
                       max_adjust=.18, min_pair_points=140):
    """IRLS global camera-Z scale synchronization with a held-out RGB guard.

    Camera-to-camera Z equation is s_j*Z_j - s_i*(R_ji*X_i).z = t_ji.z.
    Hold out every fifth correspondence before fitting; revert all changes if
    this same-scene but unseen-pixel check fails. No hole filling or sensor Z.
    """
    n = len(depths)
    if not (n == len(rgbs) == len(poses)) or n == 0:
        raise ValueError("Depth/RGB/pose count mismatch")
    if not (0 < max_adjust <= .3 and prior_strength > 0):
        raise ValueError("Invalid synchronization bounds")
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid K")
    for d, p in zip(depths, poses):
        if (np.asarray(p).shape != (4, 4) or not np.isfinite(p).all()
                or not np.isfinite(d).all() or (np.asarray(d) < 0).any()):
            raise ValueError("Nonfinite or negative input")
    pairs = []
    for i, j in _pairs(poses):
        rows = _sample_pair(i, j, depths, rgbs, poses, K)
        if len(rows) >= min_pair_points:
            pairs.append((i, j, rows))
    if not pairs:
        return [np.asarray(d, dtype=np.float32).copy() for d in depths], {
            "status": "no_supported_RGB_pairs", "pairs": 0,
            "scales": [1.]*n, "observations": 0}
    lhs, rhs, weights, validation = [], [], [], []
    for i, j, rows in pairs:
        z_j, rotated_z, translation_z = rows.T
        A = np.zeros((len(rows), n), dtype=np.float64)
        A[:, j] = z_j
        A[:, i] = -rotated_z
        residual = translation_z-z_j+rotated_z
        train = np.arange(len(rows)) % 5 != 0
        validation.append((A[~train], residual[~train]))
        lhs.append(A[train])
        rhs.append(residual[train])
        weights.append(np.full(int(train.sum()), 1/np.sqrt(int(train.sum()))))
    A, y, pair_w = np.concatenate(lhs), np.concatenate(rhs), np.concatenate(weights)
    delta = np.zeros(n)
    for _ in range(6):
        residual = A@delta-y
        robust = np.minimum(1., .10/np.maximum(np.abs(residual), 1e-8))
        w = pair_w*np.sqrt(robust)
        M = np.vstack((A*w[:, None], np.eye(n)*np.sqrt(prior_strength/n),
                       np.ones((1, n))*np.sqrt(prior_strength)))
        v = np.concatenate((y*w, np.zeros(n+1)))
        delta = np.linalg.lstsq(M, v, rcond=None)[0]
    raw = 1+delta
    scales = np.clip(raw, 1-max_adjust, 1+max_adjust)
    Av = np.concatenate([a for a, _ in validation])
    yv = np.concatenate([b for _, b in validation])
    heldout_before = float(np.sqrt(np.mean(yv*yv)))
    heldout_after = float(np.sqrt(np.mean((Av@(scales-1)-yv)**2)))
    applied = heldout_after < heldout_before
    if not applied:
        scales = np.ones(n)
    output = []
    for d, s in zip(depths, scales):
        z = np.asarray(d, dtype=np.float32).copy()
        valid = z > 0
        z[valid] *= float(s)
        z[(z > 4.) | (z < .45)] = 0
        output.append(z)
    return output, {
        "status": "RGB_pose_scale_synchronized" if applied else "heldout_no_improvement_reverted",
        "pairs": len(pairs), "observations": len(y),
        "scales": [float(s) for s in scales],
        "clamped_frames": int(np.count_nonzero(raw != np.clip(raw, 1-max_adjust, 1+max_adjust))),
        "median_pair_residual_before_m": float(np.median(np.abs(y))),
        "median_pair_residual_after_m": float(np.median(np.abs(A@(scales-1)-y))),
        "rmse_pair_before_m": float(np.sqrt(np.mean(y*y))),
        "rmse_pair_after_m": float(np.sqrt(np.mean((A@(scales-1)-y)**2))),
        "heldout_pair_rmse_before_m": heldout_before,
        "heldout_pair_rmse_after_m": heldout_after,
        "applied": applied,
        "pair_counts": [{"i": i, "j": j, "pixels": len(rows)} for i, j, rows in pairs],
        "source": "RGB inferred depth and known metric poses only; no measured Z"}
