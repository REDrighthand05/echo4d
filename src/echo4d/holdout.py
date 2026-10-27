"""Held-out RGB-D *view consistency* against a previously fused mesh (reused from PR #8).

Renders the actual triangles, not only vertices, into never-fused camera views.
This does NOT provide laser-scanned surface ground truth or independent poses.
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np

from .model import Camera, DataError, Frame, rigid_matrix


def read_echo_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read only ECHO's precise binary little-endian triangle PLY schema."""
    path = Path(path)
    with path.open('rb') as stream:
        lines = []
        while True:
            if sum(len(x) for x in lines) > 4096 or len(lines) > 24:
                raise DataError('PLY header exceeds strict ECHO format bounds')
            line = stream.readline()
            if not line:
                raise DataError('Truncated PLY header')
            lines.append(line.decode('ascii'))
            if line == b'end_header\n':
                break
        h = ''.join(lines)
        if not h.startswith('ply\nformat binary_little_endian 1.0\n'):
            raise DataError('Not binary little-endian ECHO PLY')
        import re
        nv = re.search(r'^element vertex (\d+)$', h, re.M)
        nf = re.search(r'^element face (\d+)$', h, re.M)
        if not nv or not nf:
            raise DataError('PLY is missing vertex/face count')
        nverts, nfaces = int(nv.group(1)), int(nf.group(1))
        if not (3 <= nverts <= 300000 and 1 <= nfaces <= 500000):
            raise DataError('PLY counts outside bounded benchmark limits')
        # Never silently parse a schema whose stride differs from echo4d.output.write_ply.
        expected = ('property float x\nproperty float y\nproperty float z\n'
                    'property float nx\nproperty float ny\nproperty float nz\n'
                    'property uchar red\nproperty uchar green\nproperty uchar blue\n'
                    f'element face {nfaces}\n'
                    'property list uchar int vertex_indices\nend_header\n')
        if expected not in h:
            raise DataError('PLY vertex/face fields differ from the ECHO exporter contract')
        vtype = np.dtype([(k, '<f4') for k in ('x', 'y', 'z', 'nx', 'ny', 'nz')]
                         + [(k, 'u1') for k in ('red', 'green', 'blue')])
        ftype = np.dtype([('count', 'u1'), ('indices', '<i4', (3,))])
        v = np.fromfile(stream, dtype=vtype, count=nverts)
        f = np.fromfile(stream, dtype=ftype, count=nfaces)
        if len(v) != nverts or len(f) != nfaces or stream.read(1):
            raise DataError('PLY payload is truncated or contains unexpected trailing bytes')
        if np.any(f['count'] != 3):
            raise DataError('PLY contains non-triangular faces')
        vertices = np.column_stack([v['x'], v['y'], v['z']]).astype(np.float64)
        faces = f['indices'].copy()
        if not np.isfinite(vertices).all() or np.any(faces < 0) or faces.max() >= nverts:
            raise DataError('PLY has invalid vertices or triangle indices')
        return vertices, faces


def rasterize_mesh_depth(vertices: np.ndarray, faces: np.ndarray,
                         camera: Camera, camera_to_world: np.ndarray,
                         near: float = 0.15, far: float = 4.0) -> tuple[np.ndarray, dict]:
    """CPU perspective-correct triangle z-buffer, aligned to ECHO integer pixel centres.

    Triangles that cross the near plane are not clipped; they are skipped and counted.
    This yields a CONSERVATIVE coverage metric near camera, never fake surfaces.
    """
    points = np.asarray(vertices)
    tris = np.asarray(faces)
    pose = np.asarray(camera_to_world)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or len(points) > 500000 or not np.isfinite(points).all()
            or tris.ndim != 2 or tris.shape[1] != 3 or tris.dtype.kind not in 'iu'
            or not len(tris) or len(tris) > 800000 or np.any(tris < 0)
            or np.any(tris >= len(points))):
        raise DataError('Invalid rasterizer mesh')
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise DataError('Invalid rasterizer camera pose')
    # Direct callers must meet the same right-handed SE(3) contract as Recording.
    # A finite scale, shear or reflection would silently distort held-out depth.
    pose = rigid_matrix(pose.flatten(order='F'), 'rasterizer camera pose')
    if not (math.isfinite(near) and math.isfinite(far) and 0 < near < far <= 100):
        raise DataError('Invalid rasterizer clipping distances')
    cam = (points - pose[:3, 3]) @ pose[:3, :3]
    out = np.full((camera.height, camera.width), np.inf, dtype=np.float64)
    stats = {'triangles': len(tris), 'skip_near_crossing': 0,
             'skip_nonprojecting': 0, 'triangles_drawn': 0}
    for inds in tris:
        p = cam[inds]
        if np.any(p[:, 2] <= near):
            stats['skip_near_crossing'] += 1
            continue
        uv = np.column_stack((camera.fx * p[:, 0] / p[:, 2] + camera.cx,
                              camera.fy * p[:, 1] / p[:, 2] + camera.cy))
        x0 = max(0, math.ceil(float(np.min(uv[:, 0])) - 1e-9))
        x1 = min(camera.width - 1, math.floor(float(np.max(uv[:, 0])) + 1e-9))
        y0 = max(0, math.ceil(float(np.min(uv[:, 1])) - 1e-9))
        y1 = min(camera.height - 1, math.floor(float(np.max(uv[:, 1])) + 1e-9))
        if x0 > x1 or y0 > y1:
            stats['skip_nonprojecting'] += 1
            continue
        a, b, c = uv
        denominator = ((b[1] - c[1]) * (a[0] - c[0])
                       + (c[0] - b[0]) * (a[1] - c[1]))
        if abs(denominator) < 1e-11:
            stats['skip_nonprojecting'] += 1
            continue
        xx, yy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        w0 = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / denominator
        w1 = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / denominator
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not np.any(inside):
            continue
        # Screen barycentric weights are NOT world barycentric weights.
        # Correct Z is reciprocal of the interpolated reciprocal depth.
        inv_z = w0 / p[0, 2] + w1 / p[1, 2] + w2 / p[2, 2]
        valid = inside & (inv_z > 0) & (inv_z >= 1.0 / far)
        if not np.any(valid):
            continue
        region = out[y0:y1 + 1, x0:x1 + 1]
        np.minimum(region, np.where(valid, 1.0 / np.maximum(inv_z, 1e-100), np.inf), out=region)
        stats['triangles_drawn'] += 1
    out[~np.isfinite(out)] = 0.0
    stats['predicted_pixels'] = int(np.count_nonzero(out))
    return out.astype(np.float32), stats


def compare_holdout_depth(prediction: np.ndarray, measured: np.ndarray,
                          excluded: np.ndarray | None = None,
                          min_depth: float = .15, max_depth: float = 4.0) -> dict:
    """No false certainty for pixels with no measured depth; report coverage separately."""
    prediction = np.asarray(prediction)
    measured = np.asarray(measured)
    if (prediction.shape != measured.shape or prediction.ndim != 2 or
            not np.isfinite(prediction).all() or not np.isfinite(measured).all() or
            np.any(prediction < 0) or np.any(measured < 0)):
        raise DataError('Depth comparison requires same-size finite nonnegative images')
    valid = (measured >= min_depth) & (measured <= max_depth)
    if excluded is not None:
        if excluded.shape != measured.shape or excluded.dtype != bool:
            raise DataError('Excluded mask must be boolean same-size')
        valid &= ~excluded
    if not np.any(valid):
        raise DataError('No valid independently measured depth pixels')
    overlap = valid & (prediction >= min_depth) & (prediction <= max_depth)
    count = int(np.count_nonzero(overlap))
    if not count:
        raise DataError('No overlapping predicted and held-out measured depth')
    residual = prediction[overlap].astype(np.float64) - measured[overlap].astype(np.float64)
    absolute = np.abs(residual)
    return {
        'sensor_valid_pixels': int(np.count_nonzero(valid)),
        'overlap_pixels': count,
        'missing_surface_pixels': int(np.count_nonzero(valid)) - count,
        'observed_depth_coverage': count / int(np.count_nonzero(valid)),
        'mae_m': float(np.mean(absolute)),
        'rmse_m': float(np.sqrt(np.mean(residual ** 2))),
        'median_abs_m': float(np.median(absolute)),
        'p95_abs_m': float(np.percentile(absolute, 95)),
        'bias_m': float(np.mean(residual)),
        'within_3cm': float(np.mean(absolute <= .03)),
        'within_5cm': float(np.mean(absolute <= .05)),
        'within_10cm': float(np.mean(absolute <= .10)),
        'prediction_valid_pixels': int(np.count_nonzero((prediction >= min_depth) &
                                                        (prediction <= max_depth))),
        'metric_scope': 'Only held-out valid measured pixels WITH a predicted surface; missed pixels reported separately.',
    }


def evaluate_frames(vertices: np.ndarray, faces: np.ndarray,
                    recording, heldout_indices: list[int], max_depth: float = 4.0) -> dict:
    if not heldout_indices or len(set(heldout_indices)) != len(heldout_indices):
        raise DataError('Holdout indices must be nonempty and unique')
    if any(type(i) is not int or i < 0 or i >= len(recording) for i in heldout_indices):
        raise DataError('Holdout index is outside recording')
    frames = []
    all_abs = []
    all_signed = []
    total_valid = total_overlap = 0
    target = set(heldout_indices)
    for i, frame, _ in recording.iter_frames():
        if i not in target:
            continue
        predicted, renderer = rasterize_mesh_depth(vertices, faces, recording.camera,
                                                    frame.camera_to_world, far=max_depth)
        score = compare_holdout_depth(predicted, frame.depth, frame.excluded, max_depth=max_depth)
        valid = (frame.depth >= .15) & (frame.depth <= max_depth)
        if frame.excluded is not None:
            valid &= ~frame.excluded
        overlap = valid & (predicted >= .15) & (predicted <= max_depth)
        residual = predicted[overlap].astype(np.float64) - frame.depth[overlap].astype(np.float64)
        all_abs.append(np.abs(residual))
        all_signed.append(residual)
        total_valid += score['sensor_valid_pixels']
        total_overlap += score['overlap_pixels']
        frames.append({'index': i, 't_s': frame.t, 'score': score, 'raster': renderer})
    if len(frames) != len(target):
        raise DataError('Did not evaluate every selected heldout index')
    errors = np.concatenate(all_signed)
    absolute = np.concatenate(all_abs)
    return {'heldout_frames': len(frames), 'heldout_indices': heldout_indices,
            'measured_valid_pixels': total_valid, 'overlap_pixels': total_overlap,
            'observed_depth_coverage': total_overlap / total_valid,
            'mae_m': float(np.mean(absolute)), 'rmse_m': float(np.sqrt(np.mean(errors ** 2))),
            'bias_m': float(np.mean(errors)), 'median_abs_m': float(np.median(absolute)),
            'p95_abs_m': float(np.percentile(absolute, 95)),
            'within_3cm': float(np.mean(absolute <= .03)),
            'within_5cm': float(np.mean(absolute <= .05)),
            'within_10cm': float(np.mean(absolute <= .1)),
            'per_frame': frames,
            'limitations': [
                'Held-out TUM RGB-D frames were NOT fused; their depths supply external observations.',
                'TUM camera poses and calibration are shared and not estimated independently.',
                'Temporal neighbours may show the same surfaces; this is NOT an independent laser scan.',
                'Metrics ignore measured pixels with no mesh surface; separate observed_depth_coverage measures this.',
                'Unknown depth pixels cannot establish false-positive geometry.',
                'Triangles crossing the near plane are skipped, not clipped; coverage is conservative there.',
                'Depth accuracy is conditioned on the visible overlap, and may be biased by dynamic objects.',
            ]}
