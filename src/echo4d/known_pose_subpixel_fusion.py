"""Conservative subpixel projection adapter for already RGB-inferred camera-Z.

Upsampling refines which measured image ray a TSDF voxel samples. It does
NOT add new observations at unknown pixels, extrapolate a surface, use sensor
Z, or create metric scale. Even-grid samples are byte-exact old estimates.
"""
from __future__ import annotations

import numpy as np
from .model import Camera, DataError


def upsampled_camera(camera: Camera) -> Camera:
    """Keep original pixel rays exactly at even coordinates (u'=2u, v'=2v)."""
    return Camera(2 * camera.width - 1, 2 * camera.height - 1,
                  2 * camera.fx, 2 * camera.fy, 2 * camera.cx, 2 * camera.cy)


def upsample_observed_depth(depth: np.ndarray, *, tolerance_m: float = .08,\n                            fallback: str = 'zero') -> tuple[np.ndarray, dict]:
    """2x-1 bilinear sampling ONLY inside mutually consistent inferred-Z cells.

    Odd horizontal/vertical sites require two real inferred neighbors; odd/odd
    sites require all four. Unknown or depth discontinuities remain unknown.
    Interpolated samples are not independent depth measurements.
    """
    if (not isinstance(depth, np.ndarray) or depth.ndim != 2 or
            depth.dtype != np.float32 or min(depth.shape) < 2 or
            not np.isfinite(depth).all() or np.any(depth < 0) or
            not np.isfinite(tolerance_m) or not .01 <= tolerance_m <= .20 or\n            fallback not in ('zero', 'nearest')):
        raise DataError('Expected finite nonnegative 2D float32 inferred Z and bounded tolerance')
    h, w = depth.shape
    out = np.zeros((2 * h - 1, 2 * w - 1), dtype=np.float32)
    out[::2, ::2] = depth

    def agree(values: tuple[np.ndarray, ...]) -> np.ndarray:
        low = np.minimum.reduce(values)
        high = np.maximum.reduce(values)
        return (low > 0) & (high - low <= np.maximum(tolerance_m, .04 * low))

    a, b = depth[:, :-1], depth[:, 1:]
    good = agree((a, b))
    out[::2, 1::2] = np.where(good, .5 * (a + b),\n                                depth[:, 1:] if fallback == 'nearest' else 0)
    horizontal = int(np.count_nonzero(good))

    a, b = depth[:-1, :], depth[1:, :]
    good = agree((a, b))
    out[1::2, ::2] = np.where(good, .5 * (a + b),\n                                depth[1:, :] if fallback == 'nearest' else 0)
    vertical = int(np.count_nonzero(good))

    values = (depth[:-1, :-1], depth[:-1, 1:],
              depth[1:, :-1], depth[1:, 1:])
    good = agree(values)
    out[1::2, 1::2] = np.where(good, .25 * sum(values),\n                                  depth[1:, 1:] if fallback == 'nearest' else 0)
    center = int(np.count_nonzero(good))
    return out, {
        'original_valid_pixels': int(np.count_nonzero(depth)),
        'interpolated_horizontal': horizontal,
        'interpolated_vertical': vertical,
        'interpolated_four_corner': center,
        'interpolated_pixels_not_independent_measurements': horizontal + vertical + center,
        'tolerance_m': tolerance_m,
        'unknown_remains_zero': fallback == 'zero',\n        'fallback': fallback,\n        'nearest_fallback_samples_not_new_measurements': (\n            int(np.count_nonzero(out)) - int(np.count_nonzero(depth)) - horizontal - vertical - center),
    }


def upsample_observed_rgb(rgb: np.ndarray) -> np.ndarray:
    """Align RGB interpolation to the same exact even-grid original rays."""
    if (not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or
            rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2 or rgb.dtype != np.uint8):
        raise DataError('Expected uint8 RGB image')
    h, w, _ = rgb.shape
    out = np.empty((2 * h - 1, 2 * w - 1, 3), dtype=np.uint8)
    source = rgb.astype(np.uint16)
    out[::2, ::2] = rgb
    out[::2, 1::2] = ((source[:, :-1] + source[:, 1:] + 1) // 2).astype(np.uint8)
    out[1::2, ::2] = ((source[:-1] + source[1:] + 1) // 2).astype(np.uint8)
    out[1::2, 1::2] = ((source[:-1, :-1] + source[:-1, 1:] +
                         source[1:, :-1] + source[1:, 1:] + 2) // 4).astype(np.uint8)
    return out
