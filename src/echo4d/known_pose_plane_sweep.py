"""CPU calibrated plane-sweep MVS: RGB + known metric poses, never sensor Z.

A hypothesis must agree photometrically with TWO distinct training views. Low
texture, inconsistent and unobserved pixels remain zero (unknown). No inpaint.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np
from echo4d.model import Camera, DataError, rigid_matrix


def infer_plane_sweep(reference: np.ndarray, sources: list[tuple[np.ndarray, np.ndarray]],
                      camera: Camera, pose_ref: np.ndarray, *,
                      near: float = 0.4, far: float = 4.0,
                      planes: int = 64, pairing: str = 'top2',
                      source_depths: list[np.ndarray] | None = None) -> tuple[np.ndarray, dict]:
    """Return metric camera-Z using 2+ known-pose RGB views, not sensor depth.

    Inverse-depth planes are fronto-parallel in the reference camera. ZNCC
    uses 5x5 windows, requires two agreeing source views and a nonadjacent
    second-best margin. Optional diverse pairing requires source matches with
    >=60-degree lateral parallax separation when available. Optional inferred
    source-view Z rejects stable foreground occlusion (not sensor Z or GT).
    Occluded/textureless pixels remain unknown.
    """
    import cv2
    if (not isinstance(reference, np.ndarray) or reference.dtype != np.uint8
            or reference.shape != (camera.height, camera.width, 3)):
        raise DataError('Plane sweep requires calibrated uint8 RGB reference')
    if not 2 <= len(sources) <= 4 or type(planes) is not int or not 16 <= planes <= 128:
        raise DataError('Plane sweep requires 2-4 sources and 16-128 planes')
    if pairing not in ('top2', 'diverse'):
        raise DataError('Unknown plane-sweep source pairing')
    if not (np.isfinite(near) and np.isfinite(far) and 0.3 <= near < far <= 6.0):
        raise DataError('Invalid metric plane sweep depth range')
    a = rigid_matrix(np.asarray(pose_ref).flatten(order='F'))
    k = np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]], np.float64)
    inv_k = np.linalg.inv(k)
    # Optional source-view Z is RGB-inferred in a separate TRAIN-only pass.
    # It is not sensor depth or a prior Mesh, and may itself be wrong.
    if source_depths is not None:
        if (len(source_depths) != len(sources) or any(
                not isinstance(d, np.ndarray) or d.shape != reference.shape[:2]
                or d.dtype != np.float32 or not np.isfinite(d).all()
                or np.any(d < 0) for d in source_depths)):
            raise DataError('Source visibility requires aligned finite inferred Z')
    relative = []
    directions = []
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
        # Use translational parallax in the REFERENCE camera. Pure camera
        # rotation must not masquerade as a second baseline direction.
        displacement = a[:3, :3].T @ (b[:3, 3] - a[:3, 3])
        lateral = displacement[:2]
        length = float(np.linalg.norm(lateral))
        directions.append(lateral / length if length >= 0.015 else None)
    all_pairs = list(combinations(range(len(relative)), 2))
    diverse_pairs = [(i, j) for i, j in all_pairs
                     if directions[i] is not None and directions[j] is not None
                     and float(np.dot(directions[i], directions[j])) <= 0.5]
    # When only one baseline direction is observable, preserve old top-two
    # behavior rather than pretending that independent occlusion evidence exists.
    pairs = diverse_pairs if pairing == 'diverse' and diverse_pairs else all_pairs
    stable_source_depth = []
    if source_depths is not None:
        support_kernel = np.ones((3, 3), np.uint8)
        for d in source_depths:
            stable = cv2.erode((d > 0).astype(np.uint8), support_kernel) > 0
            spread = cv2.dilate(d, support_kernel) - cv2.erode(d, support_kernel)
            stable &= spread <= np.maximum(.12, .08 * d)
            stable_source_depth.append(stable.astype(np.uint8))
    occluded_samples = 0
    gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float32)
    win = (5, 5)
    def smooth(x):
        return cv2.boxFilter(x, cv2.CV_32F, win, normalize=True, borderType=cv2.BORDER_REFLECT)
    mref = smooth(gray)
    vref = np.maximum(smooth(gray * gray) - mref * mref, 0)
    texture = vref >= 64.0
    yy, xx = np.indices(gray.shape, dtype=np.float32)
    xy = np.stack((xx, yy), axis=-1)
    source_ray_z = [r[2, 0] * (xx - camera.cx) / camera.fx
                    + r[2, 1] * (yy - camera.cy) / camera.fy + r[2, 2]
                    for _, r, _ in relative] if source_depths is not None else []
    z_planes = 1.0 / np.linspace(1.0 / far, 1.0 / near, planes)
    volume = np.full((planes, camera.height, camera.width), np.inf, np.float32)
    kernel = np.ones((5, 5), np.uint8)
    for pi, z in enumerate(z_planes):
        candidates = []
        for si, (src, r, t) in enumerate(relative):
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
            if source_depths is not None:
                # Reject a plane behind a stable, nearer RGB-inferred surface
                # in this source view. Unknown or discontinuous source Z is
                # never treated as an occluder; no pixels are inpainted.
                sampled = cv2.remap(source_depths[si], u, v, cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                stable = cv2.remap(stable_source_depth[si], u, v, cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
                projected_z = z * source_ray_z[si] + t[2]
                hidden = stable & (projected_z > sampled + np.maximum(.12, .08 * sampled))
                occluded_samples += int(np.count_nonzero(valid & hidden))
                valid &= ~hidden
            cost[~valid] = np.inf
            candidates.append(cost)
        stack = np.stack(candidates)
        if pairing == 'diverse' and diverse_pairs:
            # Two photometric matches from the same side can reinforce one
            # occlusion error. Require a pair with >=60 deg parallax separation
            # when geometry offers such a pair; leave unsupported Z unknown.
            score = np.minimum.reduce([.5 * (stack[i] + stack[j]) for i, j in pairs])
        else:
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
        'source_pairing': pairing,
        'source_visibility': source_depths is not None,
        'occluded_photo_samples': occluded_samples,
        'diverse_pair_count': len(diverse_pairs),
        'candidate_pair_count': len(pairs),
        'used_diverse_pairs': bool(pairing == 'diverse' and diverse_pairs),
        'near_m': near, 'far_m': far,
        'valid_depth_pixels': int(np.count_nonzero(valid)),
        'total_pixels': int(depth.size),
        'valid_fraction': float(np.mean(valid)),
        'unknown_pixels': int(depth.size - np.count_nonzero(valid)),
    }
