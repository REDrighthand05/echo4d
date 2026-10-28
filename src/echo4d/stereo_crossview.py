"""Conservative train-only geometric contradictions for RGB-inferred camera-Z.

Zero is unknown; a closer neighbor can occlude a point, not contradict it.
No sensor depth, holdout image, or generated surface enters this check.
"""
from __future__ import annotations
import numpy as np


def reject_crossview_foreground(depths, poses, camera, *, neighbor_radius=2,
                                tolerance_m=0.12):
    """Reject source foreground pixels where another training view sees farther.

    A corroborating view overrides contradictions. Missing observations and
    occluded points are retained; no previously unknown pixels are filled.
    """
    if not 2 <= len(depths) == len(poses) <= 48:
        raise ValueError('Cross-view check requires 2..48 aligned training views')
    if not 1 <= neighbor_radius <= 4 or not 0.02 <= tolerance_m <= 0.5:
        raise ValueError('Cross-view resource or tolerance bound invalid')
    shape = (camera.height, camera.width)
    if any(not isinstance(d, np.ndarray) or d.shape != shape or
           not np.issubdtype(d.dtype, np.floating) or
           not np.isfinite(d).all() or (d < 0).any() for d in depths):
        raise ValueError('Depth maps must be finite nonnegative float camera-Z')
    matrices = [np.asarray(p, dtype=np.float64) for p in poses]
    if any(p.shape != (4, 4) or not np.isfinite(p).all() or
           not np.allclose(p[3], [0, 0, 0, 1], atol=1e-6) or
           not np.allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), atol=1e-4) or
           np.linalg.det(p[:3, :3]) < 0.999 for p in matrices):
        raise ValueError('Camera-to-world poses must be rigid and finite')
    yy, xx = np.indices(shape)
    rays = np.stack(((xx-camera.cx)/camera.fx,
                     (yy-camera.cy)/camera.fy, np.ones(shape)), axis=-1)
    outputs, diagnostics = [], []
    for i, depth in enumerate(depths):
        support = np.zeros(shape, np.uint8)
        contradict = np.zeros(shape, np.uint8)
        valid = depth > 0
        if not valid.any():
            outputs.append(depth.copy())
            diagnostics.append({'source_view': i, 'input_pixels': 0,
                                'supported_pixels': 0, 'contradicted_pixels': 0,
                                'rejected_pixels': 0, 'output_pixels': 0})
            continue
        xyz = rays[valid] * depth[valid, None]
        for j in range(max(0, i-neighbor_radius), min(len(depths), i+neighbor_radius+1)):
            if i == j:
                continue
            rotation = matrices[j][:3, :3].T @ matrices[i][:3, :3]
            translation = matrices[j][:3, :3].T @ (matrices[i][:3, 3] - matrices[j][:3, 3])
            projected = xyz @ rotation.T + translation
            z = projected[:, 2]
            u = np.rint(camera.fx * projected[:, 0] / np.maximum(z, 1e-6) + camera.cx).astype(int)
            v = np.rint(camera.fy * projected[:, 1] / np.maximum(z, 1e-6) + camera.cy).astype(int)
            inside = ((z > .1) & (u >= 0) & (u < camera.width)
                      & (v >= 0) & (v < camera.height))
            observed = np.zeros(len(z), dtype=np.float32)
            observed[inside] = depths[j][v[inside], u[inside]]
            comparable = inside & (observed > 0)
            delta = observed - z
            tolerance = np.maximum(tolerance_m, .05 * z)
            yes = comparable & (np.abs(delta) <= tolerance)
            no = comparable & (delta > tolerance)  # farther, not an occluder
            support[valid] += yes.astype(np.uint8)
            contradict[valid] += no.astype(np.uint8)
        rejected = valid & (contradict > 0) & (support == 0)
        filtered = depth.copy()
        filtered[rejected] = 0
        outputs.append(filtered)
        diagnostics.append({'source_view': i, 'input_pixels': int(valid.sum()),
                            'supported_pixels': int(np.count_nonzero(support)),
                            'contradicted_pixels': int(np.count_nonzero(contradict)),
                            'rejected_pixels': int(rejected.sum()),
                            'output_pixels': int(np.count_nonzero(filtered))})
    return outputs, diagnostics
