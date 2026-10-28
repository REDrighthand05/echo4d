"""Train-only RGB stereo surface extension. Unknown stays zero; no inpainting."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import convolve, distance_transform_edt, label


def extend_contiguous_stereo_depth(depths: list[np.ndarray], rgb: np.ndarray,
                                   *, radius_px: int = 3,
                                   tolerance_m: float = 0.10,
                                   color_tolerance: float = 32.0) -> tuple[np.ndarray, dict]:
    """Add actual backup stereo Z near an agreeing best-connected surface.

    Input depth maps share the same calibrated original left camera. Only
    training RGB matches contribute; no sensor depth or held-out input.
    """
    if not 2 <= len(depths) <= 3 or type(radius_px) is not int or not 1 <= radius_px <= 5:
        raise ValueError('Expected 2..3 stereo backups and radius 1..5')
    if not 0.02 <= tolerance_m <= 0.30 or not 5 <= color_tolerance <= 64:
        raise ValueError('Stereo extension tolerance out of bounds')
    if not isinstance(depths[0], np.ndarray) or depths[0].ndim != 2:
        raise ValueError('Expected 2D depth arrays')
    shape = depths[0].shape
    if (not isinstance(rgb, np.ndarray) or rgb.shape != (*shape, 3)
            or rgb.dtype != np.uint8 or any(
                not isinstance(d, np.ndarray) or d.shape != shape
                or not np.issubdtype(d.dtype, np.floating)
                or not np.isfinite(d).all() or (d < 0).any() for d in depths)):
        raise ValueError('Expected aligned finite nonnegative depths and uint8 RGB')
    scores = []
    for depth in depths:
        mask = depth > 0
        components, count = label(mask, structure=np.ones((3, 3), bool))
        sizes = np.bincount(components[mask], minlength=count + 1)
        scores.append((int(sizes.max()) if count else 0, int(mask.sum())))
    best = max(range(len(depths)), key=lambda i: (*scores[i], -i))
    base = depths[best].astype(np.float32, copy=True)
    known = base > 0
    if not known.any():
        return base, {'selected_candidate_index': best, 'added_pixels': 0,
                      'conflict_pixels': 0, 'base_pixels': 0,
                      'output_pixels': 0, 'candidate_scores': [list(x) for x in scores]}
    distance, nearest = distance_transform_edt(~known, return_indices=True)
    nearest_depth = base[tuple(nearest)]
    near_color = rgb[tuple(nearest)].astype(np.float32)
    color_delta = np.abs(rgb.astype(np.float32) - near_color).mean(axis=2)
    allowed = (~known) & (distance <= radius_px) & (color_delta <= color_tolerance)
    limit = np.maximum(tolerance_m, 0.05 * nearest_depth)
    proposed = []
    for i, depth in enumerate(depths):
        if i == best:
            continue
        mask = depth > 0
        local_support = convolve(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                                 mode='constant', cval=0)
        good = allowed & mask & (local_support >= 3) & (np.abs(depth - nearest_depth) <= limit)
        proposed.append(np.where(good, depth, np.nan))
    stack = np.stack(proposed)
    count = np.isfinite(stack).sum(axis=0)
    low = np.min(np.where(np.isfinite(stack), stack, np.inf), axis=0)
    high = np.max(np.where(np.isfinite(stack), stack, -np.inf), axis=0)
    conflict = (count > 1) & ((high - low) > limit)
    accept = (count > 0) & ~conflict
    total = np.where(np.isfinite(stack), stack, 0).sum(axis=0)
    base[accept] = total[accept] / count[accept]
    return base, {'selected_candidate_index': best, 'added_pixels': int(accept.sum()),
                  'conflict_pixels': int(conflict.sum()), 'base_pixels': int(known.sum()),
                  'output_pixels': int(np.count_nonzero(base)),
                  'candidate_scores': [list(x) for x in scores],
                  'rule': 'RGB stereo backup Z near matching colour/depth surface; no inpainting'}
