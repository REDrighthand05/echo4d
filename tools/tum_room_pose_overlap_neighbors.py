"""Independent known-pose RGB witness selector using projected frustum overlap.

Only selects source RGB cameras. No sensor Z or prior Mesh is read.
The existing ZNCC plane sweep remains responsible for rejecting false depths.
"""
from __future__ import annotations

import math
import numpy as np


def projected_overlap(reference: np.ndarray, source: np.ndarray,
                      *, width: int = 320, height: int = 240,
                      fx: float = 262.5, fy: float = 262.5,
                      cx: float = 159.5, cy: float = 119.5) -> float:
    """Fraction of 25 image rays in source frustum at five hypothetical Z planes.

    These planes screen camera overlap, NOT scene depth or surface evidence.
    """
    K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
    x = np.linspace(.08 * width, .92 * width, 5)
    y = np.linspace(.10 * height, .90 * height, 5)
    u, v = np.meshgrid(x, y)
    pixels = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], axis=0)
    rays = np.linalg.solve(K, pixels)
    transform = np.linalg.inv(source) @ reference
    total = 0.
    for z in (.65, 1., 1.5, 2.2, 3.2):
        pts = transform[:3, :3] @ (rays * z) + transform[:3, 3:4]
        proj = K @ pts
        uv = proj[:2] / np.maximum(proj[2], 1e-8)
        inside = ((pts[2] > .15) & (uv[0] >= 8.) & (uv[0] < width - 8.)
                  & (uv[1] >= 8.) & (uv[1] < height - 8.))
        total += float(np.count_nonzero(inside)) / u.size
    return total / 5.


def choose_overlap_neighbors(poses: list[np.ndarray], anchor: int,
                             limit: int = 4) -> list[int]:
    """Choose distinct translated RGB witnesses; keep unsupported frames unknown."""
    if not (0 <= anchor < len(poses)) or limit < 2:
        raise ValueError("Invalid anchor or neighbor limit")
    poses = [np.asarray(p, dtype=np.float64) for p in poses]
    if any(p.shape != (4, 4) or not np.isfinite(p).all() for p in poses):
        raise ValueError("Expected finite 4x4 camera-to-world poses")
    a = poses[anchor]
    candidates = []
    for j, b in enumerate(poses):
        if j == anchor:
            continue
        delta = b[:3, 3] - a[:3, 3]
        baseline = float(np.linalg.norm(delta))
        lateral = float(np.linalg.norm(delta - np.dot(delta, a[:3, 2]) * a[:3, 2]))
        facing = float(np.dot(a[:3, 2], b[:3, 2]))
        if not (.025 <= baseline <= .85 and lateral >= .018 and
                facing >= math.cos(math.radians(48))):
            continue
        overlap = projected_overlap(a, b)
        if overlap < .25:
            continue
        score = (-overlap + .5 * abs(baseline - .14)
                 + .1 * (1. - facing) + .0001 * abs(j - anchor))
        candidates.append((score, j))
    candidates.sort()
    result = []
    for _, j in candidates:
        if any(np.linalg.norm(poses[j][:3, 3] - poses[k][:3, 3]) <= 1e-6
               for k in result):
            continue
        result.append(j)
        if len(result) == limit:
            break
    return result
