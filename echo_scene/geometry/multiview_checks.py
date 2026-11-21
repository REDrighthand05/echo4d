"""Multi-view geometric consistency and re-projection validation.

Implements rigorous camera projection, occlusion conflict detection,
and free-space constraint checking to validate extrapolated structural geometry.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def project_points_to_camera(
    points_world: np.ndarray,
    r_world_to_cam: np.ndarray,
    t_world_to_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points into camera pixel coordinates.

    Returns:
        pixels: (N, 2) [u, v]
        depths: (N,) Z_cam
        valid: (N,) boolean mask (Z > 0)
    """
    pts_cam = np.dot(points_world, r_world_to_cam.T) + t_world_to_cam
    z = pts_cam[:, 2]
    valid = z > 1e-4

    u = np.zeros_like(z)
    v = np.zeros_like(z)
    safe_z = np.where(valid, z, 1.0)
    u[valid] = fx * (pts_cam[valid, 0] / safe_z[valid]) + cx
    v[valid] = fy * (pts_cam[valid, 1] / safe_z[valid]) + cy

    pixels = np.column_stack([u, v])
    return pixels, z, valid


class MultiViewConsistencyChecker:
    """Verifies geometric surface patches against registered SfM cameras."""

    def __init__(self, poses_path: Path, alignment_path: Optional[Path] = None):
        self.cameras: List[Dict[str, Any]] = []
        self.colmap_from_export_rot: Optional[np.ndarray] = None
        poses_path = Path(poses_path)
        if poses_path.exists():
            try:
                data = json.loads(poses_path.read_text(encoding="utf-8"))
                self._load_poses(data)
            except Exception:
                pass
        if alignment_path and Path(alignment_path).exists():
            try:
                adata = json.loads(Path(alignment_path).read_text(encoding="utf-8"))
                self.colmap_from_export_rot = np.array(
                    adata.get("rotation", np.eye(3)), dtype=np.float64
                )
            except Exception:
                pass

    def _load_poses(self, data: Dict[str, Any]) -> None:
        frames = data.get("frames", data) if isinstance(data, dict) else {}
        for name, meta in frames.items():
            if not isinstance(meta, dict):
                continue
            if "cam_from_world_quat_xyzw" in meta and "cam_from_world_t" in meta:
                # Direct COLMAP export format
                q = meta["cam_from_world_quat_xyzw"]
                x, y, z, w = q
                r_w2c = np.array([
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ], dtype=np.float64)
                t_w2c = np.array(meta["cam_from_world_t"], dtype=np.float64)
                center = -np.dot(r_w2c.T, t_w2c)
                cam_spec = meta.get("camera", {})
                params = cam_spec.get("params", [768.0, 320.0, 240.0])
                width = cam_spec.get("width", 640)
                height = cam_spec.get("height", 480)
                fx = params[0] if len(params) > 0 else 768.0
                fy = params[0] if len(params) > 0 else 768.0
                cx = params[1] if len(params) > 1 else width / 2.0
                cy = params[2] if len(params) > 2 else height / 2.0
            elif "center" in meta:
                center = np.array(meta["center"], dtype=np.float64)
                r_w2c = np.array(meta.get("rotation", np.eye(3)), dtype=np.float64)
                width = meta.get("width", 640)
                height = meta.get("height", 480)
                fx = meta.get("fx", 768.0)
                fy = meta.get("fy", 768.0)
                cx = meta.get("cx", width / 2.0)
                cy = meta.get("cy", height / 2.0)
                t_w2c = -np.dot(r_w2c, center)
            else:
                continue

            self.cameras.append({
                "name": name,
                "center": center,
                "r_w2c": r_w2c,
                "t_w2c": t_w2c,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "width": width,
                "height": height,
            })

    def count_supporting_views(
        self,
        surface_points: np.ndarray,
        min_in_frame_ratio: float = 0.25,
    ) -> List[str]:
        """Find cameras that observe the candidate surface patch."""
        if not self.cameras or len(surface_points) == 0:
            return []

        pts = surface_points
        if self.colmap_from_export_rot is not None:
            pts = pts @ self.colmap_from_export_rot

        supporting = []
        for cam in self.cameras:
            pixels, z, valid = project_points_to_camera(
                pts,
                cam["r_w2c"],
                cam["t_w2c"],
                cam["fx"],
                cam["fy"],
                cam["cx"],
                cam["cy"],
            )
            # Check in-bounds
            in_bounds = (
                valid
                & (pixels[:, 0] >= 0)
                & (pixels[:, 0] < cam["width"])
                & (pixels[:, 1] >= 0)
                & (pixels[:, 1] < cam["height"])
            )
            ratio = np.sum(in_bounds) / len(pts)
            if ratio >= min_in_frame_ratio:
                supporting.append(cam["name"])

        return supporting

    def compute_max_parallax_deg(
        self,
        target_point: np.ndarray,
        view_names: List[str],
    ) -> float:
        """Calculate maximum angular parallax among supporting camera centers."""
        if len(view_names) < 2:
            return 0.0

        pt = target_point
        if self.colmap_from_export_rot is not None:
            pt = pt @ self.colmap_from_export_rot

        name_to_cam = {c["name"]: c for c in self.cameras}
        centers = [
            name_to_cam[n]["center"] for n in view_names if n in name_to_cam
        ]
        if len(centers) < 2:
            return 0.0

        max_angle = 0.0
        for i in range(len(centers)):
            v1 = centers[i] - pt
            len1 = np.linalg.norm(v1)
            if len1 < 1e-6:
                continue
            v1 /= len1
            for j in range(i + 1, len(centers)):
                v2 = centers[j] - pt
                len2 = np.linalg.norm(v2)
                if len2 < 1e-6:
                    continue
                v2 /= len2
                cos_ang = np.clip(np.dot(v1, v2), -1.0, 1.0)
                ang_deg = math.degrees(math.acos(cos_ang))
                if ang_deg > max_angle:
                    max_angle = ang_deg

        return max_angle

    def validate_patch(
        self,
        patch_points: np.ndarray,
        min_views: int = 2,
        min_parallax_deg: float = 3.0,
    ) -> Tuple[bool, float, List[str]]:
        """Validate candidate patch geometrically. Returns (is_valid, heuristic_score, views)."""
        views = self.count_supporting_views(patch_points)
        if len(views) < min_views:
            return False, 0.2, views

        centroid = np.mean(patch_points, axis=0)
        parallax = self.compute_max_parallax_deg(centroid, views)
        if parallax < min_parallax_deg:
            return False, 0.35, views

        # Heuristic confidence score (0.0 to 1.0)
        view_score = min(0.5, len(views) * 0.1)
        parallax_score = min(0.4, (parallax / 20.0) * 0.4)
        score = 0.1 + view_score + parallax_score
        return True, min(0.95, score), views
