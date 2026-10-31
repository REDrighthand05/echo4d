"""Choose RGB-only Lounge views from holes in the RETAINED measured Mesh.

All scores use only the 72 sensor-depth anchors, their TSDF Mesh, and supplied
camera poses. No target depth or supplied scan PRIOR is read during selection.
This is a scene-specific camera adapter, not a generic experiment planner.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from tum_xyz_open3d_320 import camera_matrix


def score_gap_views(o3d, mesh, camera, poses, candidate_ids):
    """Raycast low-resolution sparse Mesh; prioritize holes near visible surfaces.

    Completely unobserved directions have no depth calibration, so the score
    requires some actual measured-Mesh visibility. A gap is a ray miss within
    six 80x60 pixels of a hit, not evidence of a true physical surface.
    """
    probe = camera.scaled(4)  # existing 320x240 fusion -> 80x60 probe
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    intrinsic = o3d.core.Tensor(camera_matrix(probe).astype(np.float32))
    rows = []
    for i in candidate_ids:
        extrinsic = o3d.core.Tensor(np.linalg.inv(poses[i]).astype(np.float32))
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic, extrinsic, probe.width, probe.height)
        hits = scene.cast_rays(rays)["t_hit"].numpy().reshape(
            probe.height, probe.width)
        visible = np.isfinite(hits) & (hits > .1)
        coverage = float(visible.mean())
        # Exclude nearly empty cameras and already complete views.
        if not .08 <= coverage <= .94:
            continue
        missing_near_measured = (~visible) & (
            distance_transform_edt(~visible) <= 6.)
        gap_fraction = float(missing_near_measured.mean())
        rows.append({"index": int(i), "measured_visible_fraction": coverage,
                     "near_surface_gap_fraction": gap_fraction})
    del scene
    return rows


def choose_gap_targets(rows, poses, count):
    """Greedy missing-surface visibility with pose diversity among RGB targets."""
    if count < 1:
        raise ValueError("count must be positive")
    # Stable tie breaking and no duplicate camera indices.
    candidates = {}
    for row in rows:
        i = int(row["index"])
        visible = float(row["measured_visible_fraction"])
        gap = float(row["near_surface_gap_fraction"])
        if i < 0 or i >= len(poses) or not np.isfinite([visible, gap]).all():
            raise ValueError("Invalid candidate score")
        if not (0 <= visible <= 1 and 0 <= gap <= 1):
            raise ValueError("Fractions outside [0,1]")
        candidates[i] = gap
    if len(candidates) < count:
        raise ValueError("Too few RGB cameras with visible measured support")
    chosen = []
    for _ in range(count):
        best, best_value = None, -np.inf
        for i in sorted(candidates):
            if i in chosen:
                continue
            if chosen:
                p = poses[i]
                distance = min(float(np.sqrt(
                    np.sum((p[:3, 3] - poses[j][:3, 3]) ** 2) +
                    .25 * np.sum((p[:3, 2] - poses[j][:3, 2]) ** 2)))
                    for j in chosen)
                diversity = min(1., distance / .55)
            else:
                diversity = 1.
            value = (candidates[i] + .01) * (.40 + .60 * diversity)
            if value > best_value:
                best, best_value = i, value
        chosen.append(best)
    return sorted(chosen)


def conservative_hole_depth(metric, sensor_projection, *, radius=14):
    """Add ONLY small locally supported missing surfaces, never refusion of known rays.

    The 72 retained sensor depths define known rays. At each RGB-only camera,
    find the nearest projected real depth for a missing pixel; permit predicted
    geometry only in a narrow band where it continues that measured surface.
    This rejects free-floating planar sheets far from any measured support.
    It is deliberately conservative: truly unseen walls remain UNKNOWN.
    """
    if metric.shape != sensor_projection.shape or metric.ndim != 2:
        raise ValueError("Predicted and sensor depth sizes differ")
    known = np.isfinite(sensor_projection) & (sensor_projection > 0)
    result = np.zeros(metric.shape, np.float32)
    if not known.any():
        return result, {"guarded_valid_fraction": 0., "added_pixels": 0}
    distance, indices = distance_transform_edt(
        ~known, return_indices=True)
    nearest = sensor_projection[tuple(indices)]
    tolerance = np.maximum(.15, .12 * nearest)
    allowed = ((~known) & (distance <= radius) & np.isfinite(metric)
               & (metric > 0) & (np.abs(metric - nearest) <= tolerance))
    result[allowed] = metric[allowed]
    return result, {"guarded_valid_fraction": float(allowed.mean()),
                    "added_pixels": int(allowed.sum())}
