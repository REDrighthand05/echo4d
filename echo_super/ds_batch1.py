# DS Flash batch 1 (2026-10-31): written by DeepSeek Flash from
# echo_super/workorders/WO-*.md specs; pending Muse verification below.

# ---- estimate_scale ----
import numpy as np


def estimate_scale(object_points, real_size_m):
    points = np.asarray(object_points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("object_points must be an (N, 3) array")

    n = points.shape[0]
    if n < 2:
        raise ValueError("object_points must contain at least two points")

    centered = points - np.mean(points, axis=0, keepdims=True)
    covariance = centered.T @ centered

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_axis = eigenvectors[:, int(np.argmax(eigenvalues))]

    projections = centered @ principal_axis
    p1, p99 = np.percentile(projections, [1.0, 99.0])
    measured_size = float(p99 - p1)

    if measured_size <= 0.0:
        raise ValueError("measured size must be positive")

    scale = float(real_size_m / measured_size)
    return scale, measured_size

# ---- build_replay_path ----
import numpy as np


def _interp_catmull_rom(points, samples_per_segment):
    n = points.shape[0]
    s = samples_per_segment
    t = np.arange(s, dtype=np.float64) / s
    t2 = t * t
    t3 = t2 * t
    segments = []
    for i in range(n - 1):
        p0 = points[max(i - 1, 0)]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[min(i + 2, n - 1)]
        a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3
        b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3
        c = -0.5 * p0 + 0.5 * p2
        d = p1
        vals = (
            a[None, :] * t3[:, None]
            + b[None, :] * t2[:, None]
            + c[None, :] * t[:, None]
            + d[None, :]
        )
        segments.append(vals)
    segments.append(points[-1][None, :])
    return np.vstack(segments)


def _interp_linear(points, samples_per_segment):
    n = points.shape[0]
    s = samples_per_segment
    t = np.arange(s, dtype=np.float64) / s
    segments = []
    for i in range(n - 1):
        p1 = points[i]
        p2 = points[i + 1]
        vals = p1[None, :] + (p2 - p1)[None, :] * t[:, None]
        segments.append(vals)
    segments.append(points[-1][None, :])
    return np.vstack(segments)


def build_replay_path(centers, look_targets, samples_per_segment=10):
    centers = np.asarray(centers, dtype=np.float64)
    look_targets = np.asarray(look_targets, dtype=np.float64)
    n = centers.shape[0]
    if n < 2:
        raise ValueError("centers must contain at least 2 points")

    s = int(samples_per_segment)
    if s < 1:
        s = 1

    if n >= 4:
        path = _interp_catmull_rom(centers, s)
        look_interp = _interp_catmull_rom(look_targets, s)
    else:
        path = _interp_linear(centers, s)
        look_interp = _interp_linear(look_targets, s)

    diff = look_interp - path
    norms = np.linalg.norm(diff, axis=1)
    look_dirs = np.zeros_like(diff)
    mask = norms > 0
    look_dirs[mask] = diff[mask] / norms[mask, None]
    return path, look_dirs

# ---- surface_voxel_indices ----
import numpy as np


def surface_voxel_indices(points, cube):
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.empty((0,), dtype=np.int64)

    q = np.rint(points / cube).astype(np.int64)
    coords = [tuple(row) for row in q.tolist()]
    occupied = set(coords)

    result = []
    for i, (x, y, z) in enumerate(coords):
        if ((x + 1, y, z) not in occupied or
                (x - 1, y, z) not in occupied or
                (x, y + 1, z) not in occupied or
                (x, y - 1, z) not in occupied or
                (x, y, z + 1) not in occupied or
                (x, y, z - 1) not in occupied):
            result.append(i)

    return np.asarray(result, dtype=np.int64)
