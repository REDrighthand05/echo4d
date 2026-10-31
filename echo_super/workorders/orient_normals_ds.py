import numpy as np


def orient_normals(points, normals, cam_centers) -> np.ndarray:
    pts = np.asarray(points)
    nrm = np.asarray(normals)
    cams = np.asarray(cam_centers)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be a 2D array with shape (N, 3)")
    if nrm.ndim != 2 or nrm.shape[1] != 3:
        raise ValueError("normals must be a 2D array with shape (N, 3)")
    if cams.ndim != 2 or cams.shape[1] != 3:
        raise ValueError("cam_centers must be a 2D array with shape (N, 3)")

    if pts.shape != nrm.shape or pts.shape != cams.shape:
        raise ValueError("points, normals, and cam_centers must have the same shape")

    if pts.shape[0] == 0:
        raise ValueError("inputs must contain at least one point (N > 0)")

    if not np.isfinite(pts).all():
        raise ValueError("points contain non-finite values")
    if not np.isfinite(nrm).all():
        raise ValueError("normals contain non-finite values")
    if not np.isfinite(cams).all():
        raise ValueError("cam_centers contain non-finite values")

    view = cams - pts
    dots = np.einsum("ij,ij->i", nrm, view)
    return np.where((dots < 0)[:, None], -nrm, nrm)
