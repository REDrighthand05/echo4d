"""Filter ONLY newly extracted TSDF faces by independent RGB-inferred Z views.

This is an opt-in geometry experiment on the existing observed-edge extractor.
The baseline triangles are preserved, even if the inferred depths disagree.
Sensor depth and prior Mesh are NOT used to decide where new geometry belongs.
The baseline Mesh is used solely to label which already-extracted faces are old.
"""
from __future__ import annotations

import numpy as np

from .known_pose_edge_extraction import _attached_faces


def _face_keys(vertices: np.ndarray, faces: np.ndarray) -> list[tuple]:
    # Matching is by geometric triangle, not PLY vertex indices/order.
    quantized = np.rint(np.asarray(vertices, dtype=np.float64) * 100000).astype(np.int64)
    return [tuple(sorted(tuple(int(v) for v in quantized[index])
                         for index in face)) for face in faces]


def filter_crossview_faces(
    original_vertices: np.ndarray, original_faces: np.ndarray,
    candidate_vertices: np.ndarray, candidate_faces: np.ndarray,
    inferred_depths: list[np.ndarray], camera_to_world: list[np.ndarray],
    camera, *, tolerance_m: float = .10, distinct_baseline_m: float = .04,
) -> tuple[np.ndarray, dict]:
    """Preserve original faces; require 2 distinct inferred-Z supports for extensions.

    Support means a face centroid projects to an *observed inferred* depth
    consistent in camera Z. A second vote from an effectively identical camera
    center does not count. Occluded or unknown pixels are not evidence.
    Re-check connectivity after filtering to avoid detached new islands.
    """
    base_v = np.asarray(original_vertices)
    base_f = np.asarray(original_faces)
    new_v = np.asarray(candidate_vertices)
    new_f = np.asarray(candidate_faces)
    if (base_v.ndim != 2 or new_v.ndim != 2 or base_v.shape[1:] != (3,)
            or new_v.shape[1:] != (3,) or base_f.ndim != 2 or new_f.ndim != 2
            or base_f.shape[1:] != (3,) or new_f.shape[1:] != (3,)
            or not len(base_f) or not len(new_f)
            or not np.isfinite(base_v).all() or not np.isfinite(new_v).all()
            or np.any(base_f < 0) or np.any(new_f < 0)
            or base_f.max() >= len(base_v) or new_f.max() >= len(new_v)):
        raise ValueError("Invalid original or candidate triangle geometry")
    if (not 2 <= len(inferred_depths) == len(camera_to_world) <= 64
            or not np.isfinite(tolerance_m) or not .02 <= tolerance_m <= .30
            or not np.isfinite(distinct_baseline_m)
            or not .01 <= distinct_baseline_m <= .5):
        raise ValueError("Invalid frozen view budget or geometry tolerance")
    h, w = camera.height, camera.width
    for depth, pose in zip(inferred_depths, camera_to_world):
        if (depth.shape != (h, w) or depth.dtype != np.float32
                or not np.isfinite(depth).all() or np.any(depth < 0)
                or pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3],
                                   np.eye(3), atol=1e-3)):
            raise ValueError("Expected frozen finite inferred Z and rigid metric poses")

    old_keys = set(_face_keys(base_v, base_f))
    candidate_keys = _face_keys(new_v, new_f)
    original = np.array([key in old_keys for key in candidate_keys], dtype=bool)
    if set(candidate_keys[i] for i in np.flatnonzero(original)) != old_keys:
        raise ValueError("Observed-edge candidate lost original baseline triangles")
    extension = np.flatnonzero(~original)
    keep = original.copy()
    independent = np.zeros(len(extension), dtype=np.int16)
    agreements = np.zeros(len(extension), dtype=np.int16)
    if len(extension):
        centers = new_v[new_f[extension]].mean(axis=1)
        first_center = np.full(len(extension), -1, dtype=np.int16)
        for j, (depth, pose) in enumerate(zip(inferred_depths, camera_to_world)):
            xyz = (centers - pose[:3, 3]) @ pose[:3, :3]
            z = xyz[:, 2]
            front = (z >= .15) & (z <= 4.0)
            safe_z = np.maximum(z, 1e-8)
            u = np.rint(camera.fx * xyz[:, 0] / safe_z + camera.cx).astype(np.int64)
            v = np.rint(camera.fy * xyz[:, 1] / safe_z + camera.cy).astype(np.int64)
            inside = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            sampled = np.zeros(len(extension), dtype=np.float32)
            sampled[inside] = depth[v[inside], u[inside]]
            consistent = (inside & (sampled > 0) &
                          (np.abs(sampled - z) <= np.maximum(tolerance_m, .04 * z)))
            agreements += consistent.astype(np.int16)
            first = consistent & (first_center < 0)
            first_center[first] = j
            other = consistent & (first_center >= 0) & (first_center != j)
            if np.any(other):
                ids = np.flatnonzero(other)
                ref = np.array([camera_to_world[int(first_center[i])][:3, 3]
                                for i in ids])
                separated = np.linalg.norm(ref - pose[:3, 3], axis=1) >= distinct_baseline_m
                independent[ids[separated]] = 2
        supported = independent >= 2
        keep[extension[supported]] = True
    # New geometry can only extend an original edge-connected surface.
    attached = _attached_faces(new_f[keep], original[keep])
    keep[np.flatnonzero(keep)[~attached]] = False
    if not np.all(keep[original]):
        raise ValueError("Baseline preservation invariant failed")
    return keep, {
        "baseline_faces_preserved": int(original.sum()),
        "candidate_extra_faces": int(len(extension)),
        "extra_faces_with_two_independent_views": int(np.count_nonzero(independent >= 2)),
        "extra_faces_retained_after_connectivity": int(np.count_nonzero(keep & ~original)),
        "extra_faces_rejected": int(np.count_nonzero(~keep & ~original)),
        "mean_raw_agreeing_views_per_extra_face":
            float(agreements.mean()) if len(extension) else 0.,
        "tolerance_m": tolerance_m,
        "distinct_camera_baseline_m": distinct_baseline_m,
        "sensor_depth_used": False,
        "original_mesh_used_as_reconstruction_prior": False,
        "baseline_mesh_used_for_old_face_label_only": True,
    }
