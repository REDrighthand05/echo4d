"""Calibrated two-view RGB stereo with *known* metric poses, no measured depth.

Optional OpenCV CPU provider. Depth is inferred from RGB correspondences, NOT a
sensor measurement. No unknown surfaces are filled and failed matches remain zero.
"""
from __future__ import annotations
import math
import numpy as np
from echo4d.model import Camera, DataError, rigid_matrix


def _intrinsics(camera: Camera) -> np.ndarray:
    return np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]], dtype=np.float64)


def rectify_geometry(camera: Camera, pose_left: np.ndarray, pose_right: np.ndarray):
    """Return OpenCV rectification for two camera-to-world transforms in metres."""
    import cv2
    a = rigid_matrix(np.asarray(pose_left).flatten(order='F'))
    b = rigid_matrix(np.asarray(pose_right).flatten(order='F'))
    r = b[:3, :3].T @ a[:3, :3]
    t = b[:3, :3].T @ (a[:3, 3] - b[:3, 3])
    baseline = float(np.linalg.norm(t))
    angle = float(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1)))
    if not 0.035 <= baseline <= 0.30 or angle > math.radians(35):
        raise DataError('Stereo pair baseline/rotation outside frozen range')
    k = _intrinsics(camera)
    zeros = np.zeros(5, np.float64)
    rect = cv2.stereoRectify(k, zeros, k, zeros, (camera.width, camera.height),
                             r, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    r1, r2, p1, p2, q = rect[:5]
    if abs(p2[1, 3]) > 1e-7 or p2[0, 3] >= -1e-6:
        raise DataError('Stereo pair is vertical or has negative expected disparity')
    return r1, r2, p1, p2, q, baseline, angle


def infer_stereo(left: np.ndarray, right: np.ndarray, camera: Camera,
                 pose_left: np.ndarray, pose_right: np.ndarray,
                 *, num_disparities: int = 64) -> tuple[np.ndarray, np.ndarray, Camera, np.ndarray, dict]:
    """Produce rectified left RGB, inferred camera-Z, rectified camera and pose.

    Positive disparities and left-right consistency are required. Does not read
    depth files, use a pretrained model, or infer poses/scale.
    """
    import cv2
    if (left.shape != (camera.height, camera.width, 3) or right.shape != left.shape
            or left.dtype != np.uint8 or right.dtype != np.uint8):
        raise DataError('Stereo requires two calibrated same-size uint8 RGB images')
    if (type(num_disparities) is not int or num_disparities % 16
            or not 16 <= num_disparities <= 128 or num_disparities >= camera.width // 2):
        raise DataError('Invalid bounded disparity search')
    r1, r2, p1, p2, q, baseline, angle = rectify_geometry(camera, pose_left, pose_right)
    k = _intrinsics(camera)
    d = np.zeros(5, dtype=np.float64)
    size = (camera.width, camera.height)
    map_l = cv2.initUndistortRectifyMap(k, d, r1, p1, size, cv2.CV_32FC1)
    map_r = cv2.initUndistortRectifyMap(k, d, r2, p2, size, cv2.CV_32FC1)
    rgb_l = cv2.remap(left, *map_l, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rgb_r = cv2.remap(right, *map_r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    gray_l = cv2.cvtColor(rgb_l, cv2.COLOR_RGB2GRAY)
    gray_r = cv2.cvtColor(rgb_r, cv2.COLOR_RGB2GRAY)
    opts = dict(numDisparities=num_disparities, blockSize=5,
                P1=8 * 5 * 5, P2=32 * 5 * 5,
                disp12MaxDiff=1, uniquenessRatio=10,
                speckleWindowSize=80, speckleRange=2)
    dl = cv2.StereoSGBM_create(minDisparity=0, **opts).compute(gray_l, gray_r).astype(np.float32) / 16
    dr = cv2.StereoSGBM_create(minDisparity=-num_disparities, **opts).compute(gray_r, gray_l).astype(np.float32) / 16
    yy, xx = np.indices(dl.shape)
    xr = np.rint(xx - dl).astype(np.int32)
    in_bounds = (xr >= 0) & (xr < camera.width)
    xr = np.clip(xr, 0, camera.width - 1)
    consistent = in_bounds & (np.abs(dl + dr[yy, xr]) <= 1.5)
    pts = cv2.reprojectImageTo3D(dl, q)
    z = pts[..., 2]
    source_l = cv2.remap(np.full(dl.shape, 255, np.uint8), *map_l,
                         cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
    source_r = cv2.remap(np.full(dl.shape, 255, np.uint8), *map_r,
                         cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
    valid = ((dl > 1) & (dl < num_disparities - 1) & consistent
             & source_l & source_r[yy, xr] & np.isfinite(z) & (z >= 0.3) & (z <= 4.0))
    depth = np.where(valid, z, 0).astype(np.float32)
    pose = rigid_matrix(np.asarray(pose_left).flatten(order='F')) @ np.block([
        [r1.T, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])
    new_cam = Camera(camera.width, camera.height, float(p1[0, 0]),
                     float(p1[1, 1]), float(p1[0, 2]), float(p1[1, 2]))
    return rgb_l, depth, new_cam, pose, {
        'baseline_m': baseline, 'relative_rotation_deg': math.degrees(angle),
        'valid_depth_pixels': int(valid.sum()), 'total_pixels': int(valid.size),
        'valid_fraction': float(valid.mean()),
        'source': 'RGB stereo matching, inferred camera-Z metres using known metric poses',
        'measured_depth_used': False, 'left_right_consistency_px': 1.5}
