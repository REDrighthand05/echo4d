"""Procedural pattern and noise generators for micro-surface synthesis.

Deterministic, seed-reproducible multi-scale noise functions including
fBm, directional anisotropic noise, Worley cellular noise, twill weave, and crack networks.
"""

from __future__ import annotations

import math
from typing import Tuple
import numpy as np


def _value_noise_2d(h: int, w: int, freq: float, seed: int = 0) -> np.ndarray:
    """Basic smooth bilinear 2D value noise."""
    rng = np.random.default_rng(seed)
    grid_h = max(2, int(h * freq))
    grid_w = max(2, int(w * freq))
    rand_grid = rng.uniform(0.0, 1.0, (grid_h, grid_w)).astype(np.float32)

    # Coordinates
    y = np.linspace(0, grid_h - 1, h, endpoint=False)
    x = np.linspace(0, grid_w - 1, w, endpoint=False)
    xx, yy = np.meshgrid(x, y)

    x0 = np.floor(xx).astype(np.int32) % grid_w
    x1 = (x0 + 1) % grid_w
    y0 = np.floor(yy).astype(np.int32) % grid_h
    y1 = (y0 + 1) % grid_h

    fx = xx - np.floor(xx)
    fy = yy - np.floor(yy)

    # Smoothstep interpolation
    fx = fx * fx * (3.0 - 2.0 * fx)
    fy = fy * fy * (3.0 - 2.0 * fy)

    top = (1.0 - fx) * rand_grid[y0, x0] + fx * rand_grid[y0, x1]
    bottom = (1.0 - fx) * rand_grid[y1, x0] + fx * rand_grid[y1, x1]
    return (1.0 - fy) * top + fy * bottom


def fbm_noise(
    shape: Tuple[int, int],
    octaves: int = 4,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Fractal Brownian Motion combining multiple octaves of smooth noise."""
    h, w = shape
    accum = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    freq = 0.02
    max_amp = 0.0

    for i in range(octaves):
        accum += amp * _value_noise_2d(h, w, freq, seed=seed + i * 31)
        max_amp += amp
        freq *= lacunarity
        amp *= gain

    return accum / max(max_amp, 1e-6)


def directional_noise(
    shape: Tuple[int, int],
    angle_rad: float = 0.0,
    freq_along: float = 0.005,
    freq_across: float = 0.25,
    seed: int = 42,
) -> np.ndarray:
    """High-frequency anisotropic directional noise (brushed metal streaks)."""
    h, w = shape
    y = np.linspace(0, 1, h, dtype=np.float32)
    x = np.linspace(0, 1, w, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    u = xx * cos_a + yy * sin_a
    v = -xx * sin_a + yy * cos_a

    rng = np.random.default_rng(seed)
    lines = rng.uniform(0.0, 1.0, max(4, int(h * freq_across))).astype(np.float32)
    v_idx = np.clip((v * (len(lines) - 1)).astype(np.int32) % len(lines), 0, len(lines) - 1)
    streak_val = lines[v_idx]

    # Add gentle along-streak variation
    along_var = np.sin(u * 2.0 * math.pi * 5.0) * 0.15 + 0.85
    return np.clip(streak_val * along_var, 0.0, 1.0)


def cellular_worley_noise(
    shape: Tuple[int, int],
    num_cells: int = 64,
    seed: int = 42,
) -> np.ndarray:
    """Cellular Worley distance noise (cast metal granularity / crystal facets)."""
    h, w = shape
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 1.0, (num_cells, 2)).astype(np.float32)

    y = np.linspace(0, 1, h, dtype=np.float32)
    x = np.linspace(0, 1, w, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])

    from scipy.spatial import KDTree
    tree = KDTree(pts)
    dists, _ = tree.query(grid_pts)
    res = dists.reshape(h, w).astype(np.float32)
    res = (res - res.min()) / max(res.max() - res.min(), 1e-6)
    return res


def twill_weave_pattern(
    shape: Tuple[int, int],
    yarn_count: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """2x2 twill weave pattern for carbon fiber composites.

    Returns:
        height: yarn surface cross-section elevation [0.0, 1.0]
        direction: angle field indicating fiber orientation
    """
    h, w = shape
    y = np.linspace(0, yarn_count, h, dtype=np.float32)
    x = np.linspace(0, yarn_count, w, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    ix = np.floor(xx).astype(np.int32)
    iy = np.floor(yy).astype(np.int32)

    # 2x2 twill condition: (x - y) mod 4 in {0, 1} -> warp over weft
    twill_cond = ((ix - iy) % 4) < 2

    # Yarn curvature (sinusoidal cross-section)
    fx = np.abs((xx - ix) - 0.5) * 2.0  # 0 at center, 1 at edge
    fy = np.abs((yy - iy) - 0.5) * 2.0

    # Height elevation
    h_warp = np.cos(fx * (math.pi / 2.0))
    h_weft = np.cos(fy * (math.pi / 2.0)) * 0.85
    height = np.where(twill_cond, h_warp, h_weft).astype(np.float32)

    # Fiber direction (0 rad for warp, pi/2 for weft)
    direction = np.where(twill_cond, 0.0, math.pi / 2.0).astype(np.float32)
    return height, direction


def hexagonal_honeycomb_pattern(
    shape: Tuple[int, int],
    cell_freq: float = 12.0,
    wall_ratio: float = 0.12,
) -> np.ndarray:
    """Hexagonal regular acoustic liner lattice."""
    h, w = shape
    y = np.linspace(0, cell_freq * math.sqrt(3), h, dtype=np.float32)
    x = np.linspace(0, cell_freq, w, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    # Hexagonal grid distance approximation
    q = (2.0 / 3.0 * xx)
    r = (-1.0 / 3.0 * xx + math.sqrt(3) / 3.0 * yy)
    rx = np.round(q)
    ry = np.round(r)
    rz = np.round(-q - r)

    dx = np.abs(q - rx)
    dy = np.abs(r - ry)
    dz = np.abs(-q - r - rz)

    dist = np.maximum(dx, np.maximum(dy, dz))
    mask = np.where(dist > (0.5 - wall_ratio), 1.0, 0.0).astype(np.float32)
    return mask


def crack_network_pattern(
    shape: Tuple[int, int],
    num_seeds: int = 36,
    seed: int = 101,
) -> np.ndarray:
    """Non-uniform thermal barrier crack network."""
    h, w = shape
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 1.0, (num_seeds, 2)).astype(np.float32)

    y = np.linspace(0, 1, h, dtype=np.float32)
    x = np.linspace(0, 1, w, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])

    from scipy.spatial import KDTree
    tree = KDTree(pts)
    dists, _ = tree.query(grid_pts, k=2)

    # Crack width based on difference between 1st and 2nd closest Voronoi seeds
    diff = np.abs(dists[:, 1] - dists[:, 0]).reshape(h, w)
    cracks = np.clip(1.0 - diff * 25.0, 0.0, 1.0).astype(np.float32)
    return cracks
