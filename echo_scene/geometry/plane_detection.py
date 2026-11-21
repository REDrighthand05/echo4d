"""Structural plane detection via adaptive RANSAC and Huber-loss refinement.

Extracts planar candidates with scale-adaptive thresholds and classifies them
into floor, wall, and ceiling components based on orientation and spatial context.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
import numpy as np

from .provenance import PlaneModel, StructuralCategory


def _huber_weights(residuals: np.ndarray, delta: float) -> np.ndarray:
    """Calculate M-estimator weights for Huber loss."""
    abs_r = np.abs(residuals)
    weights = np.ones_like(abs_r)
    mask = abs_r > delta
    weights[mask] = delta / np.maximum(abs_r[mask], 1e-7)
    return weights


def fit_plane_ransac_huber(
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    distance_threshold: float = 0.04,
    normal_angle_threshold_deg: float = 12.0,
    max_iterations: int = 400,
    huber_delta: float = 0.03,
) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
    """Fit a dominant plane using RANSAC followed by Huber IRLS refinement."""
    n_pts = len(points)
    if n_pts < 3:
        return None

    best_inliers: Optional[np.ndarray] = None
    best_normal: Optional[np.ndarray] = None
    best_d: float = 0.0
    cos_angle_thr = math.cos(math.radians(normal_angle_threshold_deg))

    rng = np.random.default_rng(seed=42)

    for _ in range(max_iterations):
        sample_idx = rng.choice(n_pts, size=3, replace=False)
        p0, p1, p2 = points[sample_idx]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            continue
        normal /= norm_len
        d = -float(np.dot(normal, p0))

        residuals = np.abs(np.dot(points, normal) + d)
        inlier_mask = residuals <= distance_threshold

        if normals is not None and np.any(inlier_mask):
            cand_normals = normals[inlier_mask]
            dot_normals = np.abs(np.dot(cand_normals, normal))
            inlier_indices = np.where(inlier_mask)[0]
            valid_normal_mask = dot_normals >= cos_angle_thr
            inlier_mask = np.zeros(n_pts, dtype=bool)
            inlier_mask[inlier_indices[valid_normal_mask]] = True

        inlier_count = int(np.sum(inlier_mask))
        if best_inliers is None or inlier_count > len(best_inliers):
            best_inliers = np.where(inlier_mask)[0]
            best_normal = normal
            best_d = d

    if best_inliers is None or len(best_inliers) < 10:
        return None

    # Huber IRLS refinement on inliers
    curr_normal = best_normal.copy()
    curr_d = best_d
    inlier_pts = points[best_inliers]

    for _ in range(5):
        residuals = np.dot(inlier_pts, curr_normal) + curr_d
        weights = _huber_weights(residuals, huber_delta)

        # Weighted centroid
        w_sum = np.sum(weights)
        if w_sum < 1e-6:
            break
        center = np.sum(inlier_pts * weights[:, None], axis=0) / w_sum

        # Weighted covariance
        centered = inlier_pts - center
        cov = np.dot((centered * weights[:, None]).T, centered)
        eigvals, eigvecs = np.linalg.eigh(cov)
        refined_normal = eigvecs[:, 0]
        if np.dot(refined_normal, curr_normal) < 0:
            refined_normal = -refined_normal
        refined_normal /= np.linalg.norm(refined_normal)
        refined_d = -float(np.dot(refined_normal, center))

        curr_normal = refined_normal
        curr_d = refined_d

    # Final inlier check
    final_residuals = np.abs(np.dot(points, curr_normal) + curr_d)
    final_mask = final_residuals <= distance_threshold
    if normals is not None:
        cand_normals = normals[final_mask]
        dot_n = np.abs(np.dot(cand_normals, curr_normal))
        valid_n = dot_n >= cos_angle_thr
        inlier_idx = np.where(final_mask)[0][valid_n]
    else:
        inlier_idx = np.where(final_mask)[0]

    return curr_normal, curr_d, inlier_idx


def classify_plane_category(
    normal: np.ndarray,
    distance: float,
    inlier_points: np.ndarray,
    up_vector: np.ndarray = np.array([0.0, 1.0, 0.0]),
) -> Tuple[StructuralCategory, float]:
    """Classify plane into FLOOR, WALL, CEILING, or OBJECT based on orientation and height."""
    norm_up = up_vector / (np.linalg.norm(up_vector) + 1e-9)
    cos_sim = float(np.dot(normal, norm_up))
    abs_cos = abs(cos_sim)

    # Height along up direction
    heights = np.dot(inlier_points, norm_up)
    mean_height = float(np.mean(heights))

    # Horizontal plane (|cos| >= cos(20 deg) = 0.939)
    if abs_cos >= 0.92:
        confidence = min(0.95, 0.6 + 0.35 * abs_cos)
        if cos_sim > 0:  # pointing generally up
            return StructuralCategory.FLOOR, confidence
        else:
            return StructuralCategory.CEILING, confidence

    # Vertical plane (|cos| <= sin(20 deg) = 0.342)
    if abs_cos <= 0.35:
        confidence = min(0.95, 0.6 + 0.35 * (1.0 - abs_cos))
        return StructuralCategory.WALL, confidence

    return StructuralCategory.OBJECT, 0.4


def detect_structural_planes(
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    min_support_points: int = 150,
    max_planes: int = 8,
    up_vector: Optional[np.ndarray] = None,
) -> List[PlaneModel]:
    """Extract structural planar segments iteratively with scale adaptation."""
    if len(points) < min_support_points:
        return []

    if up_vector is None:
        up_vector = np.array([0.0, 1.0, 0.0])

    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    adaptive_dist = max(0.015, min(0.06, extent * 0.012))
    huber_delta = adaptive_dist * 0.75

    remaining_mask = np.ones(len(points), dtype=bool)
    detected_planes: List[PlaneModel] = []

    for _ in range(max_planes):
        rem_indices = np.where(remaining_mask)[0]
        if len(rem_indices) < min_support_points:
            break

        cand_pts = points[rem_indices]
        cand_normals = normals[rem_indices] if normals is not None else None

        result = fit_plane_ransac_huber(
            cand_pts,
            normals=cand_normals,
            distance_threshold=adaptive_dist,
            huber_delta=huber_delta,
        )
        if result is None:
            break

        normal, d, local_inliers = result
        if len(local_inliers) < min_support_points:
            break

        global_inliers = rem_indices[local_inliers]
        inlier_pts = points[global_inliers]

        cat, conf = classify_plane_category(
            normal, d, inlier_pts, up_vector=up_vector
        )

        plane = PlaneModel(
            normal=normal,
            distance=d,
            inlier_indices=global_inliers,
            category=cat,
            confidence_heuristic=conf,
            bounds_min=inlier_pts.min(axis=0),
            bounds_max=inlier_pts.max(axis=0),
        )
        detected_planes.append(plane)
        remaining_mask[global_inliers] = False

    return detected_planes
