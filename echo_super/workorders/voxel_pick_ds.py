import numpy as np

def voxel_pick(points, colors, voxel_size) -> tuple:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    n = points.shape[0]

    colors = np.asarray(colors)
    if colors.shape != (n, 3):
        raise ValueError("colors must have shape (N, 3) matching points")

    if n == 0:
        raise ValueError("N must be greater than 0")

    try:
        vs = float(voxel_size)
    except (TypeError, ValueError):
        raise ValueError("voxel_size must be positive and finite")

    if not np.isfinite(vs) or vs <= 0.0:
        raise ValueError("voxel_size must be positive and finite")

    if not np.isfinite(points).all():
        raise ValueError("points contain non-finite coordinates")

    keys = np.floor(points / vs)
    idx = np.arange(n)

    order = np.lexsort((idx, keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_points = points[order]
    sorted_keys = keys[order]
    sorted_idx = idx[order]

    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    if n > 1:
        new_group[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)

    starts = np.flatnonzero(new_group)
    counts = np.diff(np.r_[starts, n])

    centroids = np.add.reduceat(sorted_points, starts, axis=0) / counts[:, None]
    centroid_per_point = np.repeat(centroids, counts, axis=0)

    diff = sorted_points - centroid_per_point
    dist2 = np.einsum("ij,ij->i", diff, diff)

    min_dist2 = np.minimum.reduceat(dist2, starts)
    candidate_mask = dist2 == np.repeat(min_dist2, counts)
    candidate_positions = np.flatnonzero(candidate_mask)

    group_ids = np.repeat(np.arange(starts.size), counts)
    candidate_groups = group_ids[candidate_positions]

    first_candidate = np.empty(candidate_positions.size, dtype=bool)
    first_candidate[0] = True
    if candidate_positions.size > 1:
        first_candidate[1:] = candidate_groups[1:] != candidate_groups[:-1]

    selected_positions = candidate_positions[first_candidate]
    selected_indices = sorted_idx[selected_positions]
    selected_indices.sort()

    return points[selected_indices], colors[selected_indices]
