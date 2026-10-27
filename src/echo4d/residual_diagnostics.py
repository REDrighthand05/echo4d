"""Explain the distribution, NOT the cause, of held-out RGB-D depth errors.

The diagnostic strata use held-out depth only after the mesh was built. They
must never be fed into TSDF optimisation or used to tune its hyperparameters.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import binary_dilation, maximum_filter, minimum_filter


LABELS = ('depth_step_band', 'missing_depth_band', 'image_border', 'interior')


def _validate(predicted: np.ndarray, measured: np.ndarray, excluded: np.ndarray | None,
              min_depth: float, max_depth: float, step_m: float, band_pixels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = np.asarray(predicted)
    obs = np.asarray(measured)
    if (pred.ndim != 2 or obs.shape != pred.shape or pred.dtype.kind not in 'fiu' or
            obs.dtype.kind not in 'fiu' or not np.isfinite(pred).all() or
            not np.isfinite(obs).all() or np.any(pred < 0) or np.any(obs < 0)):
        raise ValueError('Require equal-shape, finite, nonnegative numeric depth images')
    if (not all(math.isfinite(x) for x in (min_depth, max_depth, step_m)) or
            not 0 < min_depth < max_depth <= 100 or step_m <= 0 or
            type(band_pixels) is not int or not 0 <= band_pixels <= 8):
        raise ValueError('Invalid depth/edge diagnostic parameters')
    valid = (obs >= min_depth) & (obs <= max_depth)
    if excluded is not None:
        if excluded.shape != obs.shape or excluded.dtype != bool:
            raise ValueError('Exclusion mask must be boolean and match image shape')
        valid &= ~excluded
    overlap = valid & (pred >= min_depth) & (pred <= max_depth)
    if not np.any(valid):
        raise ValueError('No held-out valid sensor depths')
    if not np.any(overlap):
        raise ValueError('No overlap between held-out sensor and mesh')
    return pred.astype(np.float64), obs.astype(np.float64), valid


def _exclusive_regions(measured: np.ndarray, valid: np.ndarray,
                       step_m: float, band_pixels: int) -> dict[str, np.ndarray]:
    """Disjoint masks. A 3x3 step detector sees only valid neighbours."""
    hi = maximum_filter(np.where(valid, measured, -np.inf), size=3, mode='constant', cval=-np.inf)
    lo = minimum_filter(np.where(valid, measured, np.inf), size=3, mode='constant', cval=np.inf)
    # A step requires two valid neighbours and an observed depth jump.
    with np.errstate(invalid='ignore'):
        step = valid & np.isfinite(hi) & np.isfinite(lo) & ((hi - lo) > step_m)
    if band_pixels:
        step = binary_dilation(step, iterations=band_pixels) & valid
    missing_band = valid & binary_dilation(~valid, iterations=max(1, band_pixels)) & ~step
    border = np.zeros(valid.shape, dtype=bool)
    b = max(1, band_pixels)
    border[:b, :] = True
    border[-b:, :] = True
    border[:, :b] = True
    border[:, -b:] = True
    border &= valid & ~(step | missing_band)
    interior = valid & ~(step | missing_band | border)
    result = dict(zip(LABELS, (step, missing_band, border, interior)))
    assert np.array_equal(sum((m.astype(np.uint8) for m in result.values())), valid.astype(np.uint8))
    return result


def _stats(residual: np.ndarray, total_squared_error: float | None = None) -> dict:
    if not len(residual):
        return {'pixels': 0, 'mae_m': None, 'rmse_m': None, 'p95_abs_m': None,
                'bias_m': None, 'squared_error_share': None}
    abs_err = np.abs(residual)
    squared_sum = float(np.dot(residual, residual))
    return {
        'pixels': int(len(residual)),
        'mae_m': float(abs_err.mean()),
        'rmse_m': float(np.sqrt(np.mean(residual ** 2))),
        'p95_abs_m': float(np.percentile(abs_err, 95)),
        'bias_m': float(np.mean(residual)),
        'squared_error_share': (squared_sum / total_squared_error if total_squared_error and total_squared_error > 0 else None),
    }


def residual_map(predicted: np.ndarray, measured: np.ndarray, excluded: np.ndarray | None = None,
                 min_depth: float = .15, max_depth: float = 4.0,
                 step_m: float = .10, band_pixels: int = 2) -> dict:
    """Return all masks/residuals for a *never-fused* observed depth image.

    A sensor step band is a correlation/stratification tool, not an inferred
    object label or proven root cause. High errors can also be pose drift,
    dynamic objects, missing geometry, sensor calibration, and occlusion.
    """
    pred, obs, valid = _validate(predicted, measured, excluded, min_depth,
                                 max_depth, step_m, band_pixels)
    overlap = valid & (pred >= min_depth) & (pred <= max_depth)
    error = np.zeros(valid.shape, dtype=np.float64)
    error[overlap] = pred[overlap] - obs[overlap]
    masks = _exclusive_regions(obs, valid, step_m, band_pixels)
    return {'valid': valid, 'overlap': overlap, 'residual': error, 'regions': masks}


def diagnose_frames(observations: list[dict], *, step_m: float = .10, band_pixels: int = 2) -> dict:
    """Aggregate frames with pixel weights, not a misleading mean of frame RMSE.

    Each element: {'index': int, 'prediction': HxW array,
                   'measured': HxW array, 'excluded': optional bool HxW}.
    """
    if not observations or len(observations) > 1000:
        raise ValueError('Expected between 1 and 1000 held-out depth frames')
    frames = []
    all_err = []
    region_err = {name: [] for name in LABELS}
    region_valid = {name: 0 for name in LABELS}
    total_valid = 0
    missing = 0
    for entry in observations:
        d = residual_map(entry['prediction'], entry['measured'], entry.get('excluded'),
                         step_m=step_m, band_pixels=band_pixels)
        r = d['residual']
        overlap = d['overlap']
        valid = d['valid']
        errs = r[overlap]
        all_err.append(errs)
        total_valid += int(valid.sum())
        missing += int((valid & ~overlap).sum())
        region_metrics = {}
        for name, mask in d['regions'].items():
            grouped = r[overlap & mask]
            region_err[name].append(grouped)
            region_valid[name] += int(mask.sum())
            region_metrics[name] = {'valid_sensor_pixels': int(mask.sum()),
                                    'overlap_pixels': int((overlap & mask).sum()),
                                    **_stats(grouped)}
        frames.append({'index': int(entry['index']), 'valid_sensor_pixels': int(valid.sum()),
                       'overlap_pixels': int(overlap.sum()),
                       'missing_surface_pixels': int((valid & ~overlap).sum()),
                       'overlap': _stats(errs), 'regions': region_metrics})
    joined = np.concatenate(all_err)
    squared_total = float(np.dot(joined, joined))
    all_abs = np.abs(joined)
    thresholds = (.03, .05, .1, .3, .5, 1.0)
    tails = {}
    for t in thresholds:
        sel = all_abs > t
        tails[f'over_{int(t*100)}cm'] = {
            'pixels': int(sel.sum()),
            'fraction_of_overlap': float(sel.mean()),
            'squared_error_share': (float(np.dot(joined[sel], joined[sel])) / squared_total
                                    if squared_total else None),
            'sign_positive_fraction': (float(np.mean(joined[sel] > 0)) if sel.any() else None),
        }
    regions = {}
    for name in LABELS:
        group = np.concatenate(region_err[name])
        regions[name] = {'valid_sensor_pixels': region_valid[name],
                         'overlap_pixels': len(group),
                         'within_region_coverage': len(group)/region_valid[name] if region_valid[name] else None,
                         **_stats(group, squared_total)}
    overlap_count = len(joined)
    if sum(region_valid.values()) != total_valid or sum(v['overlap_pixels'] for v in regions.values()) != overlap_count:
        raise AssertionError('Sensor regions are not disjoint')
    return {
        'method': 'post-hoc depth-discontinuity stratification; no heldout pixels used to fit mesh',
        'parameters': {'step_m': step_m, 'band_pixels': band_pixels,
                       'depth_valid_range_m': [.15, 4.0]},
        'sensor_valid_pixels': total_valid,
        'overlap_pixels': overlap_count,
        'missing_surface_pixels': missing,
        'coverage': overlap_count/total_valid,
        'mae_m': float(np.mean(all_abs)),
        'rmse_m': float(np.sqrt(np.mean(joined**2))),
        'p95_abs_m': float(np.percentile(all_abs, 95)),
        'median_abs_m': float(np.median(all_abs)),
        'bias_m': float(joined.mean()),
        'tails': tails,
        'regions': regions,
        'per_frame': frames,
        'limitations': [
            'Depth edges here are detected from the held-out ground truth ONLY for analysis, never training.',
            'Depth-step proximity is correlated with edges, but cannot prove the cause of a residual.',
            'Pixel errors are not statistically independent (same sequence and local surfaces).',
            'Measured missing surfaces are counted by coverage, not imputed.',
            'Unmeasured depth cannot establish spurious predicted geometry.',
            'This is not a laser-scanned 3D accuracy benchmark.',
        ],
    }