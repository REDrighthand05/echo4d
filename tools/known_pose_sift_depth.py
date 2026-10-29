"""Known-pose RGB triangulation anchors for DA2 depth. No sensor Z or old Mesh.

SIFT, RANSAC and OpenCV triangulation supply sparse RGB-only metric anchors.
A bounded smooth 2D grid corrects local depth warp without filling holes.
"""
from __future__ import annotations
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt


def triangulated_anchors(rgbs, poses, K, *, min_matches=12):
    """Per-camera [u,v,metric_z] from RGB and known metric poses."""
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("OpenCV SIFT required; cannot fabricate geometry")
    n = len(rgbs)
    if len(poses) != n or n == 0:
        raise ValueError("RGB/pose length mismatch")
    K = np.asarray(K, np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid camera calibration")
    poses = [np.asarray(p, np.float64) for p in poses]
    if any(p.shape != (4, 4) or not np.isfinite(p).all() for p in poses):
        raise ValueError("Invalid camera poses")
    sift = cv2.SIFT_create(nfeatures=1200, contrastThreshold=.015)
    feats = [sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), None)
             for im in rgbs]
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    w2cs = [np.linalg.inv(p) for p in poses]
    anchors = [[] for _ in range(n)]
    accepted_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            baseline = float(np.linalg.norm(poses[i][:3, 3] - poses[j][:3, 3]))
            facing = float(poses[i][:3, 2] @ poses[j][:3, 2])
            if not (.07 < baseline < .8 and facing > .77):
                continue
            key_i, desc_i = feats[i]
            key_j, desc_j = feats[j]
            if desc_i is None or desc_j is None or len(desc_j) < 2:
                continue
            matches = matcher.knnMatch(desc_i, desc_j, k=2)
            matched = [a for a, b in matches if a.distance < .72 * b.distance]
            if len(matched) < min_matches:
                continue
            xy_i = np.asarray([key_i[m.queryIdx].pt for m in matched], np.float64)
            xy_j = np.asarray([key_j[m.trainIdx].pt for m in matched], np.float64)
            F, inliers = cv2.findFundamentalMat(xy_i, xy_j, cv2.FM_RANSAC, 1., .995)
            if F is None or inliers is None:
                continue
            xy_i = xy_i[inliers.ravel().astype(bool)]
            xy_j = xy_j[inliers.ravel().astype(bool)]
            if len(xy_i) < min_matches:
                continue
            P_i, P_j = K @ w2cs[i][:3, :], K @ w2cs[j][:3, :]
            homogeneous = cv2.triangulatePoints(P_i, P_j, xy_i.T, xy_j.T)
            good_w = np.abs(homogeneous[3]) > 1e-8
            if good_w.sum() < min_matches:
                continue
            X = (homogeneous[:3, good_w] / homogeneous[3:4, good_w]).T
            xy_i, xy_j = xy_i[good_w], xy_j[good_w]
            camera_i = (w2cs[i][:3, :3] @ X.T + w2cs[i][:3, 3:4]).T
            camera_j = (w2cs[j][:3, :3] @ X.T + w2cs[j][:3, 3:4]).T
            q_i, q_j = (K @ camera_i.T).T, (K @ camera_j.T).T
            with np.errstate(divide="ignore", invalid="ignore"):
                uv_i, uv_j = q_i[:, :2]/q_i[:, 2:3], q_j[:, :2]/q_j[:, 2:3]
                ray_i, ray_j = X - poses[i][:3, 3], X - poses[j][:3, 3]
                cosine = (ray_i*ray_j).sum(axis=1) / (
                    np.linalg.norm(ray_i, axis=1)*np.linalg.norm(ray_j, axis=1))
            valid = ((camera_i[:, 2] > .45) & (camera_i[:, 2] < 4.) &
                     (camera_j[:, 2] > .45) & (camera_j[:, 2] < 4.) &
                     (np.linalg.norm(uv_i-xy_i, axis=1) < 1.2) &
                     (np.linalg.norm(uv_j-xy_j, axis=1) < 1.2) &
                     (cosine < np.cos(np.deg2rad(1.2))) & np.isfinite(cosine))
            if valid.sum() < 8:
                continue
            accepted_pairs += 1
            anchors[i].extend(np.column_stack((xy_i[valid], camera_i[valid, 2])).tolist())
            anchors[j].extend(np.column_stack((xy_j[valid], camera_j[valid, 2])).tolist())
    return [np.asarray(a, np.float64).reshape(-1, 3) for a in anchors], accepted_pairs


def _design(x, y, w, h, nx=5, ny=4):
    x = np.clip(np.asarray(x)/(w-1)*(nx-1), 0, nx-1-1e-6)
    y = np.clip(np.asarray(y)/(h-1)*(ny-1), 0, ny-1-1e-6)
    i, j = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x-i, y-j
    indices = np.stack((j*nx+i, j*nx+i+1, (j+1)*nx+i, (j+1)*nx+i+1), axis=1)
    values = np.stack(((1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy), axis=1)
    D = np.zeros((len(x), nx*ny), np.float64)
    for k in range(4):
        D[np.arange(len(x)), indices[:, k]] += values[:, k]
    return D


def correct_frame(depth, anchors, *, min_anchors=30, max_correction_m=.22):
    """Correct only if disjoint unique-pixel RGB anchor holdout improves."""
    depth = np.asarray(depth, np.float32)
    if depth.ndim != 2 or not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError("Invalid predicted depth")
    h, w = depth.shape
    anchors = np.asarray(anchors, np.float64).reshape(-1, 3)
    unchanged = depth.copy()
    failure = {"applied": False, "anchors": 0, "heldout": 0}
    if len(anchors) < min_anchors:
        return unchanged, failure
    xy = np.rint(anchors[:, :2]).astype(int)
    inside = ((xy[:, 0] >= 0) & (xy[:, 0] < w) &
              (xy[:, 1] >= 0) & (xy[:, 1] < h) & np.isfinite(anchors[:, 2]))
    xy, truth = xy[inside], anchors[inside, 2]
    predicted = depth[xy[:, 1], xy[:, 0]]
    valid = ((predicted > .45) & (predicted < 4.) &
             (truth > .45) & (truth < 4.) & (np.abs(predicted-truth) < .7))
    xy, residual = xy[valid], truth[valid]-predicted[valid]
    if len(xy) < min_anchors:
        return unchanged, failure
    unique, inverse = np.unique(xy, axis=0, return_inverse=True)
    if len(unique) < min_anchors:
        return unchanged, failure
    obs = np.asarray([np.median(residual[inverse == k]) for k in range(len(unique))])
    # Unique image coordinates, not raw pair rows: duplicate tracks cannot leak.
    train = ((unique[:, 0]*73 + unique[:, 1]*101) % 7) > 1
    if train.sum() < 20 or (~train).sum() < 7:
        return unchanged, failure
    D = _design(unique[:, 0], unique[:, 1], w, h)
    nx, ny = 5, 4
    smooth = []
    for y in range(ny):
        for x in range(nx):
            for a, b in ((y+1, x), (y, x+1)):
                if a >= ny or b >= nx:
                    continue
                row = np.zeros(nx*ny)
                row[y*nx+x], row[a*nx+b] = 1, -1
                smooth.append(row)
    L = np.asarray(smooth)
    delta = np.zeros(nx*ny)
    for _ in range(5):
        error = D[train] @ delta - obs[train]
        robust = np.minimum(1., .10/np.maximum(np.abs(error), 1e-7))
        M = D[train].T @ (robust[:, None]*D[train]) + 2*L.T@L + .6*np.eye(nx*ny)
        rhs = D[train].T @ (robust*obs[train])
        delta = np.linalg.solve(M, rhs)
    yy, xx = np.mgrid[:h, :w]
    correction = (_design(xx.ravel(), yy.ravel(), w, h) @ delta).reshape(h, w)
    supported = np.zeros((h, w), bool)
    supported[unique[train, 1], unique[train, 0]] = True
    distance = distance_transform_edt(~supported)
    correction *= np.clip((90-distance)/60, 0, 1)
    correction = np.clip(correction, -max_correction_m, max_correction_m)
    before = np.abs(obs[~train])
    after = np.abs(obs[~train]-correction[unique[~train, 1], unique[~train, 0]])
    accepted = bool(after.mean() < .98*before.mean())
    report = {"applied": accepted, "anchors": int(len(unique)),
              "heldout": int(len(before)), "mae_before_m": float(before.mean()),
              "mae_after_m": float(after.mean()),
              "max_abs_correction_m": float(np.max(np.abs(correction)))}
    if not accepted:
        return unchanged, report
    out = unchanged.copy()
    valid = out > 0
    out[valid] = np.clip(out[valid]+correction[valid], .45, 4.)
    return out, report


def correct_sequence(depths, rgbs, poses, K):
    if not (len(depths) == len(rgbs) == len(poses)):
        raise ValueError("Input sequence length mismatch")
    anchors, pairs = triangulated_anchors(rgbs, poses, K)
    corrected, reports = [], []
    for d, a in zip(depths, anchors):
        z, report = correct_frame(d, a)
        corrected.append(z)
        reports.append(report)
    return corrected, {"triangulated_rgb_pairs": pairs,
                       "frames_corrected": sum(r["applied"] for r in reports),
                       "frame_reports": reports,
                       "source": "SIFT RGB triangulation + known metric poses; no sensor depth"}
