"""Known-pose real-room RGB depth: patch ZNCC + non-neighboring depth uniqueness.

Replaces the winner-take-all grayscale SAD cost in #94, without sensor depth,
a prior Mesh, weights or GPU. This is a distinct depth producer, not another
TSDF post-filter. The original room importer, fusion and 3+2 render are reused.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

import tum_room_knownpose_multiview as base
from echo4d.model import DataError, file_sha256

PATCH = 7
MIN_VARIANCE = 16.0
MAX_ZNCC_COST = .52
MIN_UNIQUENESS_MARGIN = .08
LOG_DEPTH_EXCLUSION = .16


def sweep_zncc(reference: np.ndarray, sources: list[np.ndarray], K: np.ndarray,
               reference_pose: np.ndarray, source_poses: list[np.ndarray],
               depths: np.ndarray = base.DEPTHS_M) -> tuple[np.ndarray, dict]:
    """Use independent RGB witnesses and distinguish far-apart depth hypotheses.

    All candidates are metric camera-Z because the known TUM camera baselines
    are in metres. The two best *distinct camera* patch scores must agree.
    A depth hypothesis is rejected when a distant hypothesis matches almost
    as well; adjacent depth bins are not counted as ambiguity.
    """
    h, w = reference.shape[:2]
    depths = np.asarray(depths, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    poses = [np.asarray(p, dtype=np.float64) for p in [reference_pose, *source_poses]]
    if (reference.shape != (h, w, 3) or reference.dtype != np.uint8
            or len(sources) != len(source_poses) or len(sources) < 2
            or any(s.shape != reference.shape or s.dtype != np.uint8 for s in sources)
            or K.shape != (3, 3) or not np.isfinite(K).all()
            or abs(np.linalg.det(K)) < 1e-8
            or depths.ndim != 1 or len(depths) < 3
            or not np.isfinite(depths).all() or np.any(depths <= 0)
            or np.any(np.diff(depths) <= 0)
            or any(p.shape != (4, 4) or not np.isfinite(p).all() for p in poses)):
        raise DataError("ZNCC sweep requires calibrated RGB, >=2 sources, finite poses and ordered metric depths")
    # Same camera center, even rotated, is not a second independent witness.
    unique = []
    for i, pose in enumerate(poses[1:]):
        if np.linalg.norm(pose[:3, 3] - poses[0][:3, 3]) <= 1e-6:
            continue
        if any(np.linalg.norm(pose[:3, 3] - poses[j+1][:3, 3]) <= 1e-6
               for j in unique):
            continue
        unique.append(i)
    if len(unique) < 2:
        raise DataError("ZNCC sweep requires >=2 sources at distinct camera positions")
    sources = [sources[i] for i in unique]
    source_poses = [poses[i+1] for i in unique]

    ref = cv2.GaussianBlur(cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY),
                           (3, 3), 0).astype(np.float32)
    grays = [cv2.GaussianBlur(cv2.cvtColor(s, cv2.COLOR_RGB2GRAY),
                              (3, 3), 0).astype(np.float32) for s in sources]
    box = lambda x: cv2.boxFilter(x, -1, (PATCH, PATCH), normalize=True)
    mean_ref = box(ref)
    var_ref = np.maximum(box(ref * ref) - mean_ref * mean_ref, 0)
    texture = box(np.abs(cv2.Sobel(ref, cv2.CV_32F, 1, 0, ksize=3)) +
                  np.abs(cv2.Sobel(ref, cv2.CV_32F, 0, 1, ksize=3)))
    inv_K = np.linalg.inv(K)
    transforms = []
    for pose in source_poses:
        R = pose[:3, :3].T @ poses[0][:3, :3]
        t = pose[:3, :3].T @ (poses[0][:3, 3] - pose[:3, 3])
        transforms.append((R, t))
    source_mask = np.full((h, w), 255, dtype=np.uint8)
    erode_kernel = np.ones((PATCH, PATCH), dtype=np.uint8)
    scores = np.empty((len(depths), h, w), dtype=np.float32)
    for zi, z in enumerate(depths):
        costs = []
        for gray, (R, t) in zip(grays, transforms):
            M = R.copy()
            M[:, 2] += t / float(z)
            H = K @ M @ inv_K
            if abs(np.linalg.det(H)) < 1e-8:
                costs.append(np.full((h, w), np.inf, dtype=np.float32))
                continue
            flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
            warped = cv2.warpPerspective(gray, H, (w, h), flags=flags)
            valid = cv2.warpPerspective(source_mask, H, (w, h),
                                        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
            # A partially out-of-frame patch is not a complete observation.
            valid = cv2.erode(valid.astype(np.uint8), erode_kernel) > 0
            mu = box(warped)
            var = np.maximum(box(warped * warped) - mu * mu, 0)
            cov = box(ref * warped) - mean_ref * mu
            zncc = cov / np.sqrt(np.maximum(var_ref * var, 1e-5))
            cost = 1 - np.clip(zncc, -1, 1)
            costs.append(np.where(valid & (var > MIN_VARIANCE),
                                  cost, np.inf))
        stack = np.stack(costs)
        scores[zi] = .5 * (np.min(stack, axis=0) +
                          np.partition(stack, 1, axis=0)[1])
    winner = np.argmin(scores, axis=0)
    best = np.take_along_axis(scores, winner[None], axis=0)[0]
    depth_values = depths.astype(np.float32)
    far_hypothesis = np.abs(np.log(depth_values[:, None, None] /
                                   depth_values[winner][None])) > LOG_DEPTH_EXCLUSION
    competitor = np.min(np.where(far_hypothesis, scores, np.inf), axis=0)
    margin = np.zeros_like(best)
    np.subtract(competitor, best, out=margin, where=np.isfinite(best))
    matched = (best < MAX_ZNCC_COST) & (texture > 9) & (var_ref > MIN_VARIANCE)
    valid = matched & (margin > MIN_UNIQUENESS_MARGIN) & np.isfinite(best)
    result = depth_values[winner].copy()
    result[~valid] = 0
    return result, {
        "valid_pixels": int(np.count_nonzero(valid)),
        "ambiguous_matched_pixels": int(np.count_nonzero(matched & ~valid)),
        "median_zncc_cost": float(np.median(best[valid])) if valid.any() else None,
        "depth_source": "two independent known-pose RGB ZNCC patch witnesses; no sensor Z",
        "depth_planes_m": len(depths),
        "source_cameras": len(sources),
        "uniqueness_margin": MIN_UNIQUENESS_MARGIN,
        "depth_exclusion_log": LOG_DEPTH_EXCLUSION,
    }


_original_run = base.run


def run_with_control(args, report):
    report["algorithm_variant"] = "7x7 patch ZNCC + separated-depth uniqueness before TSDF"
    _original_run(args, report)
    previous_dir = os.environ.get("ECHO_PREVIOUS_COHERENCE")
    if previous_dir:
        previous = Path(previous_dir)
        prior_report = base.json.loads((previous / "report.json").read_text())
        prior_mesh = previous / "multiview-mesh" / "mesh.ply"
        if (file_sha256(prior_mesh) != prior_report["new_mesh"]["sha256"]
                or prior_report["source_archive_sha256"] != report["source_archive_sha256"]
                or prior_report["selection"]["expanded_indices"]
                   != report["selection"]["expanded_indices"]):
            raise DataError("Previous RGB-coherence control does not match room, split or Mesh SHA")
        high = base.json.loads(
            (args.baseline / "room-sparse-quality-report.json").read_text())
        new_mesh = args.report_dir / "multiview-mesh" / "mesh.ply"
        report["visual_old_vs_zncc"] = base.compare_multiview_five(
            args.work / "room-import", prior_mesh, new_mesh, high,
            args.report_dir / "visual-old-vs-zncc",
            "previous RGB multiview spatial coherence / no prior Mesh")
        import trimesh
        from echo4d.holdout import read_echo_ply
        from tum_graph_visual import read_vertex_rgb
        vv, ff = read_echo_ply(prior_mesh)
        cc = read_vertex_rgb(prior_mesh, len(vv))
        old_glb = args.report_dir / "old-coherence-rgb.glb"
        trimesh.Trimesh(vertices=vv, faces=ff, vertex_colors=cc,
                        process=False).export(old_glb)
        report["old_coherence_control"] = {
            "mesh_sha256": file_sha256(prior_mesh),
            "glb_sha256": file_sha256(old_glb),
            "role": "posthoc A/B only; never candidate input"}
    report["issue_74"] = "FAIL / NOT ACCEPTED"
    report["status"] = "real_room_ZNCC_mesh_generated_visual_FAIL_pending_review"


def main():
    base.sweep_depth = sweep_zncc
    base.run = run_with_control
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
