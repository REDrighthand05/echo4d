"""Conservative temporal refinement of measured camera-Z for Open3D TSDF.

Uses ONLY source and independently supplied training RGB-D camera poses. All
unknown/invalid pixels remain unknown. No mesh prior or evaluation holdout.
"""
import math
import numpy as np
from echo4d.model import Camera, DataError


def _rigid(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all() or
            not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6, rtol=0) or
            not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=2e-3, rtol=0) or
            abs(np.linalg.det(pose[:3, :3]) - 1.) > 2e-3):
        raise DataError("Temporal Z requires a rigid camera-to-world transform")
    return pose


def refine_observed_depth(depth: np.ndarray, camera_to_world: np.ndarray,
                          witnesses: list[tuple[np.ndarray, np.ndarray]],
                          camera: Camera, tolerance_m: float = .04
                          ) -> tuple[np.ndarray, int]:
    """Median of source Z and >=2 consistent, edge-stable training Z views.

    Each candidate must survive a forward projection, observed 5-neighbor
    depth consistency, witness Z agreement, and reverse source reprojection.
    A missing or contradictory view does not vote; no depths are synthesized.
    """
    if (not isinstance(camera, Camera) or not isinstance(depth, np.ndarray) or
            depth.shape != (camera.height, camera.width) or
            not np.issubdtype(depth.dtype, np.floating)):
        raise DataError("Temporal Z requires floating depth and matched camera")
    if (type(tolerance_m) not in (float, int) or not math.isfinite(tolerance_m) or
            not 0 < tolerance_m <= .20):
        raise DataError("Temporal Z tolerance outside (0,20cm]")
    if not isinstance(witnesses, list) or len(witnesses) > 8:
        raise DataError("Temporal Z accepts at most eight training witnesses")
    pose = _rigid(camera_to_world)
    result = depth.copy()
    if len(witnesses) < 2:
        return result, 0
    ys, xs = np.nonzero(np.isfinite(depth) & (depth > .15))
    if not len(xs):
        return result, 0
    z = depth[ys, xs].astype(np.float64)
    p = np.stack(((xs - camera.cx)*z/camera.fx, (ys - camera.cy)*z/camera.fy, z))
    votes = []
    source_inverse = np.linalg.inv(pose)
    for image, witness_pose in witnesses:
        if (not isinstance(image, np.ndarray) or image.shape != depth.shape or
                not np.issubdtype(image.dtype, np.floating)):
            raise DataError("Temporal Z witness resolution/type mismatch")
        witness_pose = _rigid(witness_pose)
        to_witness = np.linalg.inv(witness_pose) @ pose
        q = to_witness[:3, :3] @ p + to_witness[:3, 3, None]
        safe_z = np.where(q[2] > .15, q[2], 1.)
        u = camera.fx*q[0]/safe_z + camera.cx
        v = camera.fy*q[1]/safe_z + camera.cy
        valid = ((q[2] > .15) & np.isfinite(u) & np.isfinite(v) &
                 (u >= 1) & (v >= 1) & (u < camera.width-2) & (v < camera.height-2))
        ind = np.flatnonzero(valid)
        vote = np.full(len(z), np.nan)
        if len(ind):
            xx = np.rint(u[ind]).astype(np.int32)
            yy = np.rint(v[ind]).astype(np.int32)
            samples = np.stack((image[yy, xx], image[yy-1, xx], image[yy+1, xx],
                                image[yy, xx-1], image[yy, xx+1])).astype(np.float64)
            measured = samples[0]
            stable = ((np.isfinite(samples) & (samples > .15)).all(axis=0) &
                      (np.ptp(samples, axis=0) <= tolerance_m) &
                      (np.abs(measured - q[2, ind]) <= tolerance_m))
            ids = ind[stable]
            if len(ids):
                rz = measured[stable]
                measured_q = np.stack(((xx[stable]-camera.cx)*rz/camera.fx,
                                       (yy[stable]-camera.cy)*rz/camera.fy, rz))
                back = source_inverse @ witness_pose
                s = back[:3, :3] @ measured_q + back[:3, 3, None]
                safe_s = np.where(s[2] > .15, s[2], 1.)
                sx = camera.fx*s[0]/safe_s + camera.cx
                sy = camera.fy*s[1]/safe_s + camera.cy
                agreed = ((s[2] > .15) & (np.abs(s[2]-z[ids]) <= tolerance_m) &
                          ((sx-xs[ids])**2 + (sy-ys[ids])**2 <= 2.25))
                vote[ids[agreed]] = s[2, agreed]
        votes.append(vote)
    good = np.count_nonzero(np.isfinite(np.stack(votes)), axis=0) >= 2
    if not good.any():
        return result, 0
    corrected = np.nanmedian(np.vstack([z]+votes)[:, good], axis=0).astype(depth.dtype)
    original = depth[ys[good], xs[good]]
    result[ys[good], xs[good]] = corrected
    return result, int(np.count_nonzero(original != corrected))
