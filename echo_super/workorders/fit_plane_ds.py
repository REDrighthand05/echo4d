import numpy as np

def fit_plane(points) -> tuple:
    try:
        arr = np.asarray(points, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("points must be a numeric (N, 3) array") from exc

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("points must be an (N, 3) array")
    if arr.shape[0] < 3:
        raise ValueError("points must contain at least 3 points")
    if not np.all(np.isfinite(arr)):
        raise ValueError("points contain non-finite values")

    centroid = arr.mean(axis=0)
    centered = arr - centroid
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)

    if singular_values[1] <= 1e-9 * singular_values[0]:
        raise ValueError("points are collinear or nearly collinear; no unique plane exists")

    normal = vh[-1].copy()
    normal /= np.linalg.norm(normal)

    idx = int(np.argmax(np.abs(normal)))
    if normal[idx] < 0:
        normal = -normal

    d = float(-np.dot(normal, centroid))
    signed_distances = arr @ normal + d
    rms = float(np.sqrt(np.mean(signed_distances * signed_distances)))

    return normal, d, rms
