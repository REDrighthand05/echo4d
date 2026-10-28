"""RGB-guided, observation-preserving consensus for known-pose inferred Z.

Only modifies or removes already inferred positive depth pixels; it never
fills unknown image locations or uses measured depth / prior Mesh.
"""
from __future__ import annotations
import numpy as np


def regularize_observed_depth(depth: np.ndarray, rgb: np.ndarray, *,
                              iterations: int = 2) -> tuple[np.ndarray, dict]:
    if (not isinstance(depth, np.ndarray) or depth.ndim != 2 or
            depth.dtype != np.float32 or not np.isfinite(depth).all() or
            np.any(depth < 0) or not isinstance(rgb, np.ndarray) or
            rgb.dtype != np.uint8 or rgb.shape != depth.shape + (3,) or
            type(iterations) is not int or not 1 <= iterations <= 3):
        raise ValueError('Expected finite nonnegative float32 Z, aligned uint8 RGB, 1-3 iterations')
    original = depth.copy()
    current = depth.copy()
    color = rgb.astype(np.float32)
    h, w = depth.shape
    for _ in range(iterations):
        numerator = current.astype(np.float64).copy()
        denominator = (current > 0).astype(np.float64)
        support = np.zeros((h, w), np.uint8)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dx or dy):
                    continue
                y0, y1 = max(0, -dy), min(h, h - dy)
                x0, x1 = max(0, -dx), min(w, w - dx)
                cy, cx = slice(y0, y1), slice(x0, x1)
                ny, nx = slice(y0 + dy, y1 + dy), slice(x0 + dx, x1 + dx)
                center = current[cy, cx]
                neighbor = current[ny, nx]
                color_gap = np.mean(np.abs(color[cy, cx] - color[ny, nx]), axis=2)
                tolerance = np.maximum(.08, .045 * center)
                agree = ((center > 0) & (neighbor > 0) &
                         (np.abs(neighbor - center) <= tolerance) &
                         (color_gap <= 22.0))
                weight = np.exp(-.5 * (color_gap / 18.0) ** 2).astype(np.float64)
                weight *= agree
                numerator[cy, cx] += weight * neighbor
                denominator[cy, cx] += weight
                support[cy, cx] += agree.astype(np.uint8)
        eligible = (current > 0) & (support >= 3)
        smooth = (numerator / np.maximum(denominator, 1e-8)).astype(np.float32)
        # Original inferred Z stays the anchor. No crossing depth edges.
        step = np.clip(smooth - current, -.045, .045)
        current = np.where(eligible, current + .7 * step, current)
        # Remove unsupported single-pixel depths, never fill holes.
        current[(current > 0) & (support == 0)] = 0
    original_valid = original > 0
    result_valid = current > 0
    return current.astype(np.float32), {
        'method': 'RGB-guided local Z consensus, no hole filling',
        'measured_depth_used': False,
        'original_valid_pixels': int(np.count_nonzero(original_valid)),
        'remaining_valid_pixels': int(np.count_nonzero(result_valid)),
        'removed_isolated_pixels': int(np.count_nonzero(original_valid & ~result_valid)),
        'adjusted_pixels': int(np.count_nonzero(result_valid & (np.abs(current - original) > 1e-6))),
        'newly_filled_pixels': int(np.count_nonzero(~original_valid & result_valid)),
        'iterations': iterations,
    }
