"""Post-hoc TUM trajectory scoring for monocular SfM.

Ground truth is used ONLY after the SfM process finishes. Sim(3) fixes a gauge
for evaluation; the fitted scale is not a measured deployable metric scale.
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .model import DataError


def read_tum_groundtruth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times, poses = [], []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 8:
            raise DataError("TUM ground-truth row must contain timestamp, xyz, xyzw quaternion")
        try:
            stamp, *values = [float(v) for v in fields]
        except ValueError as exc:
            raise DataError("Non-numeric TUM ground-truth row") from exc
        if not np.isfinite([stamp, *values]).all():
            raise DataError("Non-finite TUM ground-truth pose")
        q = np.asarray(values[3:], dtype=float)
        norm = np.linalg.norm(q)
        if not 0.95 <= norm <= 1.05:
            raise DataError("Invalid TUM ground-truth quaternion")
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(q / norm).as_matrix()
        pose[:3, 3] = values[:3]
        times.append(stamp)
        poses.append(pose)
    if len(times) < 3 or any(b <= a for a, b in zip(times, times[1:])):
        raise DataError("TUM ground-truth timestamps require >=3 strictly increasing entries")
    return np.asarray(times), np.stack(poses)


def interpolate_gt(t: float, times: np.ndarray, poses: np.ndarray,
                   max_gap: float = 0.10) -> np.ndarray | None:
    """Interpolate only within bracketed, close GT samples; never extrapolate."""
    i = int(np.searchsorted(times, t, side="left"))
    if i < len(times) and abs(times[i] - t) < 1e-8:
        return poses[i].copy()
    if i == 0 or i == len(times):
        return None
    lo, hi = i - 1, i
    if (t - times[lo] > max_gap or times[hi] - t > max_gap or
            times[hi] <= times[lo]):
        return None
    alpha = (t - times[lo]) / (times[hi] - times[lo])
    pose = np.eye(4)
    pose[:3, 3] = (1 - alpha) * poses[lo, :3, 3] + alpha * poses[hi, :3, 3]
    pose[:3, :3] = Slerp(times[[lo, hi]],
                         Rotation.from_matrix(poses[[lo, hi], :3, :3]))([t])[0].as_matrix()
    return pose


def _valid_pose(matrix: object) -> np.ndarray:
    pose = np.asarray(matrix, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise DataError("Non-finite or incorrectly shaped 4x4 pose")
    r = pose[:3, :3]
    if (not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-7) or
            not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or
            not math.isclose(np.linalg.det(r), 1.0, abs_tol=1e-5)):
        raise DataError("Pose must be a right-handed rigid camera-to-world transform")
    return pose


def sim3_align(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Proper-rotation least-squares similarity map source -> target."""
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or x.shape != y.shape or x.shape[1] != 3 or len(x) < 3:
        raise DataError("Sim(3) requires at least 3 paired 3D camera centres")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise DataError("Non-finite trajectory positions")
    mx, my = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - mx, y - my
    variance = float((xc * xc).sum() / len(x))
    if (variance < 1e-12 or np.linalg.matrix_rank(xc, tol=1e-8) < 2 or
            np.linalg.matrix_rank(yc, tol=1e-8) < 2):
        raise DataError("Degenerate collinear or stationary trajectory; cannot assess Sim(3)")
    u, singular, vt = np.linalg.svd(xc.T @ yc / len(x))
    d = np.eye(3)
    d[-1, -1] = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ d @ u.T
    scale = float(np.sum(singular * np.diag(d)) / variance)
    if not 0 < scale < 1e12 or not np.isfinite(scale):
        raise DataError("Invalid similarity alignment scale")
    return scale, r, my - scale * (r @ mx)


def score_pairs(estimated: list, reference: list) -> dict:
    if len(estimated) != len(reference):
        raise DataError("Estimated and reference pose counts differ")
    est = np.stack([_valid_pose(p) for p in estimated])
    gt = np.stack([_valid_pose(p) for p in reference])
    scale, r, t = sim3_align(est[:, :3, 3], gt[:, :3, 3])
    aligned = scale * (r @ est[:, :3, 3].T).T + t
    residual = np.linalg.norm(aligned - gt[:, :3, 3], axis=1)
    angles = []
    for estimate, truth in zip(est, gt):
        relative = truth[:3, :3].T @ r @ estimate[:3, :3]
        angles.append(float(np.degrees(Rotation.from_matrix(relative).magnitude())))
    angles = np.asarray(angles)
    return {
        "pairs": int(len(est)),
        "ate_rmse_m_after_sim3": float(np.sqrt(np.mean(residual ** 2))),
        "ate_median_m_after_sim3": float(np.median(residual)),
        "rotation_mean_deg_after_sim3": float(angles.mean()),
        "rotation_median_deg_after_sim3": float(np.median(angles)),
        "rotation_p95_deg_after_sim3": float(np.percentile(angles, 95)),
        "alignment": {"scale_for_evaluation_only": scale,
                      "rotation_row_major": r.tolist(),
                      "translation_m": t.tolist()},
        "metric_scale_estimated_by_sfm": False,
        "warning": "Sim(3) uses held-out GT only for scoring; scale is not deployable."
    }


def evaluate_tum(trajectory: dict, sampling: dict, source_frames: list[dict],
                 gt_times: np.ndarray, gt_poses: np.ndarray) -> dict:
    if trajectory.get("metric_scale_known") is not False:
        raise DataError("Evaluation expects an explicitly unknown-scale trajectory")
    samples = sampling.get("samples", [])
    lookup = {}
    for sample in samples:
        name = sample["name"]
        index = sample["decoded_frame_index"]
        if name in lookup or not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(source_frames):
            raise DataError("Duplicate or out-of-bounds decoded-frame mapping")
        lookup[name] = source_frames[index]["t"]
    entries = trajectory["poses"]
    if len({entry["name"] for entry in entries}) != len(entries):
        raise DataError("Duplicate registered frame name")
    est, truth = [], []
    for entry in entries:
        if entry["name"] not in lookup:
            raise DataError("Registered frame absent from sampled RGB manifest")
        gt = interpolate_gt(lookup[entry["name"]], gt_times, gt_poses)
        if gt is not None:
            est.append(entry["camera_to_world_row_major"])
            truth.append(gt)
    if len(est) < 3:
        raise DataError("Insufficient independent TUM ground-truth pairs")
    report = score_pairs(est, truth)
    report["registered_images"] = len(entries)
    report["sampled_images"] = len(samples)
    report["registration_fraction"] = len(entries) / len(samples)
    report["paired_groundtruth_fraction_of_registered"] = len(est) / len(entries)
    report["groundtruth_stage"] = "post-hoc only, TUM camera poses never provided to COLMAP"
    return report
