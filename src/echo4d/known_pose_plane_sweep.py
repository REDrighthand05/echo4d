"""CPU calibrated plane-sweep MVS: RGB + known metric poses, never sensor Z.

A hypothesis must agree photometrically with TWO distinct training views. Low
texture, inconsistent and unobserved pixels remain zero (unknown). No inpaint.
"""
from __future__ import annotations
import numpy as np
from echo4d.model import Camera, DataError, rigid_matrix


def infer_plane_sweep(reference: np.ndarray, sources: list[tuple[np.ndarray, np.ndarray]],
                      camera: Camera, pose_ref: np.ndarray, *,
                      near: float = 0.4, far: float = 4.0,
                      planes: int = 64) -> tuple[np.ndarray, dict]:
    """Return metric camera-Z using 2+ known-pose RGB views, not sensor depth.

    Inverse-depth planes are fronto-parallel in the reference camera. ZNCC
    uses 5x5 windows, requires two agreeing source views and a nonadjacent
    second-best margin. Occluded/textureless pixels are deliberately unknown.
    """
    import cv2
    if (not isinstance(reference, np.ndarray) or reference.dtype != np.uint8
            or reference.shape != (camera.height, camera.width, 3)):
        raise DataError('Plane sweep requires calibrated uint8 RGB reference')
    if not 2 <= len(sources) <= 4 or type(planes) is not int or not 16 <= planes <= 128:
        raise DataError('Plane sweep requires 2-4 sources and 16-128 planes')
    if not (np.isfinite(near) and np.isfinite(far) and 0.3 <= near < far <= 6.0):
        raise DataError('Invalid metric plane sweep depth range')
    a = rigid_matrix(np.asarray(pose_ref).flatten(order='F'))
    k = np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]], np.float64)
    inv_k = np.linalg.inv(k)
    relative = []
    for image, pose in sources:
        if (not isinstance(image, np.ndarray) or image.shape != reference.shape
                or image.dtype != np.uint8):
            raise DataError('Plane sweep source RGB mismatch')
        b = rigid_matrix(np.asarray(pose).flatten(order='F'))
        r = b[:3, :3].T @ a[:3, :3]
        t = b[:3, :3].T @ (a[:3, 3] - b[:3, 3])
        baseline = float(np.linalg.norm(t))
        if not 0.035 <= baseline <= 0.30:
            raise DataError('Plane sweep source baseline outside bounded range')
        relative.append((cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32), r, t))
    gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float32)
    win = (5, 5)
    def smooth(x):
        return cv2.boxFilter(x, cv2.CV_32F, win, normalize=True, borderType=cv2.BORDER_REFLECT)
    mref = smooth(gray)
    vref = np.maximum(smooth(gray * gray) - mref * mref, 0)
    texture = vref >= 64.0
    yy, xx = np.indices(gray.shape, dtype=np.float32)
    xy = np.stack((xx, yy), axis=-1)
    z_planes = 1.0 / np.linspace(1.0 / far, 1.0 / near, planes)
    volume = np.full((planes, camera.height, camera.width), np.inf, np.float32)
    kernel = np.ones((5, 5), np.uint8)
    for pi, z in enumerate(z_planes):
        candidates = []
        for src, r, t in relative:
            h = k @ (r + np.outer(t, [0., 0., 1.]) / z) @ inv_k
            uv = cv2.perspectiveTransform(xy.reshape(-1, 1, 2), h).reshape(gray.shape + (2,))
            u, v = uv[..., 0], uv[..., 1]
            inside = ((u >= 2) & (u < camera.width - 3) &
                      (v >= 2) & (v < camera.height - 3))
            warped = cv2.remap(src, u, v, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            ms = smooth(warped)
            vs = np.maximum(smooth(warped * warped) - ms * ms, 0)
            cov = smooth(gray * warped) - mref * ms
            corr = cov / np.sqrt(np.maximum(vref * vs, 1e-8))
            cost = np.clip(1.0 - corr, 0, 2).astype(np.float32)
            valid = (cv2.erode(inside.astype(np.uint8), kernel) > 0) & (vs >= 64)
            cost[~valid] = np.inf
            candidates.append(cost)
        stack = np.stack(candidates)
        two = np.partition(stack, 1, axis=0)[:2]
        score = (two[0] + two[1]) * .5
        volume[pi] = np.where(texture, score, np.inf)
    best_idx = np.argmin(volume, axis=0)
    best = np.take_along_axis(volume, best_idx[None, ...], axis=0)[0]
    indices = np.arange(planes)[:, None, None]
    alternatives = np.where(np.abs(indices - best_idx[None, ...]) >= 3,
                            volume, np.inf)
    second = alternatives.min(axis=0)
    valid = np.isfinite(best) & np.isfinite(second) & (best <= 0.35)
    valid &= (np.where(np.isfinite(second), second, 3.0) -
              np.where(np.isfinite(best), best, 3.0)) >= 0.025
    depth = np.where(valid, z_planes[best_idx], 0).astype(np.float32)
    return depth, {
        'method': 'known-pose CPU plane-sweep ZNCC, two-source support',
        'source': 'RGB-inferred camera-Z with metric scale from supplied poses',
        'measured_depth_used': False,
        'source_view_count': len(sources), 'planes': planes,
        'near_m': near, 'far_m': far,
        'valid_depth_pixels': int(np.count_nonzero(valid)),
        'total_pixels': int(depth.size),
        'valid_fraction': float(np.mean(valid)),
        'unknown_pixels': int(depth.size - np.count_nonzero(valid)),
    }
