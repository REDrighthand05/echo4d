"""Train-only cross-view reprojection of RGB-inferred camera-Z maps.

Unknown pixels are transferred only where TWO geometrically distinct
source views independently support a matching surface and observed RGB.
No measured depth, prior Mesh, learned weights, or hole inpainting.
"""
from __future__ import annotations
import numpy as np


def _supported_local_surface(depth: np.ndarray) -> np.ndarray:
    """Two adjacent agreeing inferred-Z samples, never a measured-depth prior."""
    support = np.zeros(depth.shape, dtype=np.uint8)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        other = np.zeros_like(depth)
        if dy == -1:
            other[1:] = depth[:-1]
        elif dy == 1:
            other[:-1] = depth[1:]
        elif dx == -1:
            other[:, 1:] = depth[:, :-1]
        else:
            other[:, :-1] = depth[:, 1:]
        agreement = ((depth > 0) & (other > 0) &
                     (np.abs(depth - other) <= np.maximum(.12, .05 * depth)))
        support += agreement.astype(np.uint8)
    return (depth > 0) & (support >= 2)


def fuse_reprojected_depths(depths: list[np.ndarray], poses: list[np.ndarray],
                            rgbs: list[np.ndarray], camera, *,
                            neighbors: int = 6, tolerance_m: float = 0.12,
                            color_threshold: float = 45.0,\n                            local_surface_support: bool = False) -> tuple[list[np.ndarray], list[dict]]:
    if (not 3 <= len(depths) == len(poses) == len(rgbs) <= 64 or
            type(neighbors) is not int or not 2 <= neighbors <= 8 or
            not np.isfinite(tolerance_m) or not 0.02 <= tolerance_m <= 0.25 or
            not np.isfinite(color_threshold) or not 0 <= color_threshold <= 100):
        raise ValueError('Invalid cross-view fusion inputs/budget')
    h, w = int(camera.height), int(camera.width)
    for d, p, rgb in zip(depths, poses, rgbs):
        if (not isinstance(d, np.ndarray) or d.shape != (h, w) or d.dtype != np.float32
                or not np.isfinite(d).all() or np.any(d < 0)
                or not isinstance(rgb, np.ndarray) or rgb.shape != (h, w, 3)
                or rgb.dtype != np.uint8
                or not isinstance(p, np.ndarray) or p.shape != (4, 4)
                or not np.isfinite(p).all() or not np.allclose(p[3], [0, 0, 0, 1], atol=1e-6)
                or not np.allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), atol=1e-3)):
            raise ValueError('Expected finite inferred float32 Z, uint8 RGB and rigid camera-to-world poses')
    fx, fy, cx, cy = float(camera.fx), float(camera.fy), float(camera.cx), float(camera.cy)
    if min(fx, fy) <= 0 or not np.isfinite([fx, fy, cx, cy]).all():
        raise ValueError('Invalid camera calibration')
    yy, xx = np.indices((h, w))
    rays = np.stack(((xx - cx) / fx, (yy - cy) / fy, np.ones((h, w))),
                    axis=-1).reshape(-1, 3)
    matrices = np.stack(poses)
    centers = matrices[:, :3, 3]
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    source_support = ([_supported_local_surface(d).reshape(-1) for d in depths]\n                      if local_surface_support else None)\n    fused, reports = [], []
    for i, anchor in enumerate(depths):
        # Pose-only selection; this function receives training maps only.
        chosen = [int(j) for j in np.argsort(distances[i], kind='stable')
                  if j != i and 0.05 <= distances[i, j] <= 0.28][:neighbors]
        if len(chosen) < 2:
            fused.append(anchor.copy())
            reports.append({'anchor': i, 'sources': chosen, 'added_pixels': 0,
                            'adjusted_pixels': 0,
                            'reason': 'insufficient distinct nearby inferred-depth views'})
            continue
        projected = []
        for j in chosen:
            src = depths[j].reshape(-1)
            valid = (src >= 0.4) & (src <= 4.0)\n            if source_support is not None:\n                valid &= source_support[j]
            if not np.any(valid):
                projected.append(np.zeros((h, w), np.float32))
                continue
            r = matrices[i, :3, :3].T @ matrices[j, :3, :3]
            t = matrices[i, :3, :3].T @ (centers[j] - centers[i])
            pts = (rays[valid] * src[valid, None]) @ r.T + t
            z = pts[:, 2]
            positive = (z >= 0.4) & (z <= 4.0)
            u = np.rint(fx * pts[positive, 0] / z[positive] + cx).astype(np.int32)
            v = np.rint(fy * pts[positive, 1] / z[positive] + cy).astype(np.int32)
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            uv = v[inside] * w + u[inside]
            zz = z[positive][inside]
            # Check observed RGB at source and projected target pixels.
            source_color = rgbs[j].reshape(-1, 3)[valid][positive][inside]
            target_color = rgbs[i].reshape(-1, 3)[uv]
            color_gap = np.mean(np.abs(source_color.astype(np.int16) -
                                       target_color.astype(np.int16)), axis=1)
            # Visibility is geometric: a rejected foreground color must not
            # expose a farther surface at the same projected anchor pixel.
            nearest = np.full(h * w, np.inf, dtype=np.float32)
            np.minimum.at(nearest, uv, zz.astype(np.float32))
            visible = np.abs(zz - nearest[uv]) <= 1e-5
            keep = (color_gap <= color_threshold) & visible
            zbuffer = np.full(h * w, np.inf, dtype=np.float32)
            np.minimum.at(zbuffer, uv[keep], zz[keep].astype(np.float32))
            zbuffer[~np.isfinite(zbuffer)] = 0
            projected.append(zbuffer.reshape(h, w))
        best = np.zeros((h, w), dtype=np.float32)
        best_cost = np.full((h, w), np.inf, dtype=np.float32)
        for a in range(len(chosen)):
            for b in range(a + 1, len(chosen)):
                if distances[chosen[a], chosen[b]] < 0.035:
                    continue
                da, db = projected[a], projected[b]
                delta = np.abs(da - db)
                agree = ((da > 0) & (db > 0) &
                         (delta <= np.maximum(tolerance_m, .04 * np.minimum(da, db))))
                candidate = .5 * (da + db)
                cost = delta + np.where(anchor > 0, .5 * np.abs(candidate - anchor), 0)
                better = agree & (cost < best_cost)
                best[better] = candidate[better]
                best_cost[better] = cost[better]
        out = anchor.copy()
        holes = (anchor == 0) & (best > 0)
        out[holes] = best[holes]
        # Contradictory existing estimates remain unchanged, not overwritten.
        matching = (anchor > 0) & (best > 0) & (np.abs(anchor - best) <= .08)
        out[matching] = .6 * anchor[matching] + .4 * best[matching]
        fused.append(out)
        reports.append({'anchor': i, 'sources': chosen,
                        'added_pixels': int(np.count_nonzero(holes)),
                        'adjusted_pixels': int(np.count_nonzero(
                            matching & (np.abs(out - anchor) > 1e-6))),
                        'original_valid_pixels': int(np.count_nonzero(anchor)),
                        'fused_valid_pixels': int(np.count_nonzero(out)),
                        'depth_source': 'two RGB-consistent, pose-reprojected inferred-depth views',\n                        'local_surface_support': local_surface_support})
    return fused, reports
