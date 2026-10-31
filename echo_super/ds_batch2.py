# DS Flash batch 2 (2026-10-31): written by DeepSeek Flash from
# echo_super/workorders/WO-*.md specs; verified by Muse the same day
# (echo_super/test_ds_batch2.py — 19 bench tests, all green first try,
# zero retries; registered via adapters.register_ds_batch2).

# ---- resample_path_even ----
import numpy as np


def resample_path_even(points, spacing):
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n < 2:
        raise ValueError("points must contain at least two vertices")
    if spacing <= 0:
        raise ValueError("spacing must be positive")

    diffs = np.diff(pts, axis=0)
    all_lens = np.linalg.norm(diffs, axis=1)
    mask = all_lens > 0.0

    if np.any(mask):
        seg_start = pts[:-1][mask]
        seg_vec = diffs[mask]
        seg_len = all_lens[mask]
        cum = np.cumsum(seg_len)
        total = float(cum[-1])
    else:
        seg_start = np.empty((0, 3), dtype=np.float64)
        seg_vec = np.empty((0, 3), dtype=np.float64)
        seg_len = np.empty((0,), dtype=np.float64)
        cum = np.empty((0,), dtype=np.float64)
        total = 0.0

    samples = [pts[0].copy()]
    eps = 1e-12
    last_pos = 0.0
    k = 1

    while True:
        pos = k * spacing
        if pos > total:
            break

        if pos >= total:
            sample = pts[-1].copy()
        else:
            idx = int(np.searchsorted(cum, pos, side="right"))
            prev = cum[idx - 1] if idx > 0 else 0.0
            t = (pos - prev) / seg_len[idx]
            sample = seg_start[idx] + t * seg_vec[idx]

        samples.append(sample)
        last_pos = pos
        k += 1

    if abs(last_pos - total) > eps:
        samples.append(pts[-1].copy())

    return np.asarray(samples, dtype=np.float64)

# ---- catmull_rom_path ----
import numpy as np


def catmull_rom_path(knots, samples_per_segment):
    knots = np.asarray(knots, dtype=np.float64)
    if knots.ndim != 2 or knots.shape[1] != 3:
        raise ValueError("knots must have shape (N, 3)")
    n = knots.shape[0]
    if n < 2:
        raise ValueError("knots must contain at least two points")
    if not isinstance(samples_per_segment, (int, np.integer)) or samples_per_segment < 1:
        raise ValueError("samples_per_segment must be an integer >= 1")
    s = int(samples_per_segment)

    diffs = knots[1:] - knots[:-1]
    dists = np.linalg.norm(diffs, axis=1)
    if np.any(dists < 1e-12):
        raise ValueError("consecutive knots must be separated by at least 1e-12")

    def _lerp(t, ta, tb, pa, pb):
        den = tb - ta
        if den == 0.0:
            if pa.ndim == 1:
                return np.broadcast_to(pa, t.shape + (3,)).copy()
            return pa.copy()
        w = (t - ta) / den
        if pa.ndim == 1:
            w = w[:, None]
            return pa[None, :] + w * (pb - pa)[None, :]
        w = w[:, None]
        return pa + w * (pb - pa)

    ext = np.empty((n + 2, 3), dtype=np.float64)
    ext[0] = knots[0]
    ext[1:n + 1] = knots
    ext[n + 1] = knots[-1]

    m = (n - 1) * s + 1
    out = np.empty((m, 3), dtype=np.float64)
    u = np.arange(s, dtype=np.float64) / float(s)

    for i in range(n - 1):
        q0 = ext[i]
        q1 = ext[i + 1]
        q2 = ext[i + 2]
        q3 = ext[i + 3]

        d01 = np.sqrt(np.linalg.norm(q1 - q0))
        d12 = np.sqrt(np.linalg.norm(q2 - q1))
        d23 = np.sqrt(np.linalg.norm(q3 - q2))

        t0 = 0.0
        t1 = t0 + d01
        t2 = t1 + d12
        t3 = t2 + d23

        t = t1 + u * (t2 - t1)

        a1 = _lerp(t, t0, t1, q0, q1)
        a2 = _lerp(t, t1, t2, q1, q2)
        a3 = _lerp(t, t2, t3, q2, q3)
        b1 = _lerp(t, t0, t2, a1, a2)
        b2 = _lerp(t, t1, t3, a2, a3)
        c = _lerp(t, t1, t2, b1, b2)

        start = i * s
        out[start:start + s] = c
        out[start] = knots[i]

    out[-1] = knots[-1]
    return out

# ---- frustum_select_points ----
import numpy as np


def frustum_select_points(points, cam_position, cam_rotation, fx, fy, cx, cy, image_width, image_height, near, far):
    pts = np.asarray(points, dtype=np.float64)
    cam = np.asarray(cam_position, dtype=np.float64)
    rot = np.asarray(cam_rotation, dtype=np.float64)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if cam.shape != (3,):
        raise ValueError("cam_position must have shape (3,)")
    if rot.shape != (3, 3):
        raise ValueError("cam_rotation must have shape (3, 3)")

    p_cam = (pts - cam) @ rot.T

    x = p_cam[:, 0]
    y = p_cam[:, 1]
    z = p_cam[:, 2]

    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * x / z + cx
        v = fy * y / z + cy

    mask = (
        (z >= near)
        & (z <= far)
        & (u >= 0.0)
        & (u < image_width)
        & (v >= 0.0)
        & (v < image_height)
    )

    return np.flatnonzero(mask).astype(np.int64)

# ---- plane_fit_lsq ----
import numpy as np


def plane_fit_lsq(points):
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
        raise ValueError("points must be an (N,3) array with N >= 3")

    centroid = pts.mean(axis=0)
    centered = pts - centroid

    covariance = centered.T @ centered
    _, eigenvectors = np.linalg.eigh(covariance)

    normal = eigenvectors[:, 0]
    normal = normal / np.linalg.norm(normal)

    idx = int(np.argmax(np.abs(normal)))
    if normal[idx] < 0.0:
        normal = -normal

    offset = -float(normal @ centroid)
    distances = pts @ normal + offset

    return normal, offset, distances

# ---- voxel_select_representative ----
import numpy as np


def voxel_select_representative(points, cell_size):
    if cell_size <= 0:
        raise ValueError("cell_size must be > 0")

    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be an (N,3) array")

    n = pts.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)

    cell_idx = np.floor(pts / cell_size).astype(np.int64)
    centers = (cell_idx + 0.5) * cell_size
    diff = pts - centers
    dist2 = np.sum(diff * diff, axis=1)

    orig_idx = np.arange(n, dtype=np.int64)

    order = np.lexsort((
        orig_idx,
        dist2,
        cell_idx[:, 2],
        cell_idx[:, 1],
        cell_idx[:, 0],
    ))

    sorted_cells = cell_idx[order]
    same_as_prev = np.all(sorted_cells[1:] == sorted_cells[:-1], axis=1)
    first_in_group = np.empty(n, dtype=bool)
    first_in_group[0] = True
    first_in_group[1:] = ~same_as_prev

    selected = order[first_in_group].astype(np.int64, copy=False)
    selected.sort()
    return selected
