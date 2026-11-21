"""Conservative spatial structure completion for floor, walls, and ceiling.

Preserves all authentic observational geometry while conservatively extrapolating
bounded planar structures with multi-view consistency checks and strict evidence provenance.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .multiview_checks import MultiViewConsistencyChecker
from .plane_detection import detect_structural_planes
from .provenance import (
    EvidenceSource,
    PlaneModel,
    SceneGeometry,
    StructuralCategory,
)


def _build_plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Construct orthonormal 2D basis (u, v) on a plane."""
    n = normal / np.linalg.norm(normal)
    ref = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def _extrapolate_plane_grid(
    plane: PlaneModel,
    inlier_points: np.ndarray,
    grid_res: float = 0.05,
    max_extension: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate conservative triangulated planar surface within observed hull."""
    if len(inlier_points) < 20:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint32)

    center = np.mean(inlier_points, axis=0)
    u_axis, v_axis = _build_plane_basis(plane.normal)

    # Project to 2D
    diff = inlier_points - center
    u_coords = np.dot(diff, u_axis)
    v_coords = np.dot(diff, v_axis)

    u_min, u_max = np.min(u_coords) - max_extension, np.max(u_coords) + max_extension
    v_min, v_max = np.min(v_coords) - max_extension, np.max(v_coords) + max_extension

    u_steps = max(2, min(80, int((u_max - u_min) / grid_res)))
    v_steps = max(2, min(80, int((v_max - v_min) / grid_res)))

    # Bounded grid
    u_lin = np.linspace(u_min, u_max, u_steps)
    v_lin = np.linspace(v_min, v_max, v_steps)
    uu, vv = np.meshgrid(u_lin, v_lin)
    grid_u = uu.ravel()
    grid_v = vv.ravel()

    # Distance to observed inliers in 2D
    obs_uv = np.column_stack([u_coords, v_coords])
    grid_uv = np.column_stack([grid_u, grid_v])

    # Simple spatial filter: only keep grid vertices close to observed points
    from scipy.spatial import KDTree
    tree = KDTree(obs_uv)
    dists, _ = tree.query(grid_uv)
    valid_mask = dists <= max_extension

    idx_map = np.full(len(grid_uv), -1, dtype=np.int32)
    valid_indices = np.where(valid_mask)[0]
    idx_map[valid_indices] = np.arange(len(valid_indices))

    # Reconstruct 3D positions: X = center + u * u_axis + v * v_axis
    valid_u = grid_u[valid_mask]
    valid_v = grid_v[valid_mask]
    extrap_verts = (
        center[None, :]
        + valid_u[:, None] * u_axis[None, :]
        + valid_v[:, None] * v_axis[None, :]
    )

    # Triangulate adjacent grid quads
    tris = []
    for j in range(v_steps - 1):
        for i in range(u_steps - 1):
            idx00 = j * u_steps + i
            idx10 = j * u_steps + (i + 1)
            idx01 = (j + 1) * u_steps + i
            idx11 = (j + 1) * u_steps + (i + 1)

            v00, v10 = idx_map[idx00], idx_map[idx10]
            v01, v11 = idx_map[idx01], idx_map[idx11]

            if v00 >= 0 and v10 >= 0 and v01 >= 0:
                tris.append([v00, v10, v01])
            if v10 >= 0 and v11 >= 0 and v01 >= 0:
                tris.append([v10, v11, v01])

    extrap_tris = np.array(tris, dtype=np.uint32) if tris else np.zeros((0, 3), dtype=np.uint32)
    return extrap_verts, extrap_tris


class ConservativeExtrapolator:
    """Orchestrates structural completion with multi-view constraints."""

    def __init__(self, job_dir: Path):
        self.job_dir = Path(job_dir)
        self.poses_path = self.job_dir / "poses.json"
        self.checker = MultiViewConsistencyChecker(
            self.poses_path, self.job_dir / "alignment.json"
        )

    def process(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None,
        observed_mesh_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Detect planes, validate support, extrapolate, and save meshes."""
        # 1. Detect structural planes with calibrated up-vector if available
        up_vec = np.array([0.0, 1.0, 0.0])
        align_path = self.job_dir / "alignment.json"
        if align_path.exists():
            try:
                align_data = json.loads(align_path.read_text(encoding="utf-8"))
                rot = np.array(align_data.get("rotation", np.eye(3)), dtype=np.float64)
                # Second row/col corresponds to Y-up in viewer coordinates
                up_vec = rot[1] if rot.shape == (3, 3) else np.array([0.0, 1.0, 0.0])
            except Exception:
                pass

        planes = detect_structural_planes(
            points, normals=normals, min_support_points=120, up_vector=up_vec
        )

        manifest: Dict[str, Any] = {
            "planes_detected": len(planes),
            "planes": [],
            "extrapolated_verts": 0,
            "extrapolated_tris": 0,
        }

        all_extrap_verts: List[np.ndarray] = []
        all_extrap_tris: List[np.ndarray] = []
        all_extrap_cats: List[str] = []

        vert_offset = 0

        for idx, plane in enumerate(planes):
            inlier_pts = points[plane.inlier_indices]
            is_valid, score, views = self.checker.validate_patch(inlier_pts)
            plane.support_views = views
            plane.confidence_heuristic = score

            p_dict = plane.to_dict()
            p_dict["validated"] = is_valid
            manifest["planes"].append(p_dict)

            # Conservative rule: only extrapolate validated planes in floor/wall/ceiling
            if not is_valid or plane.category == StructuralCategory.UNKNOWN:
                continue

            v_cand, t_cand = _extrapolate_plane_grid(plane, inlier_pts)
            if len(v_cand) == 0 or len(t_cand) == 0:
                continue

            all_extrap_verts.append(v_cand)
            all_extrap_tris.append(t_cand + vert_offset)
            all_extrap_cats.extend([plane.category.value] * len(v_cand))
            vert_offset += len(v_cand)

        # Assemble structural geometry
        if all_extrap_verts:
            struct_verts = np.vstack(all_extrap_verts)
            struct_tris = np.vstack(all_extrap_tris)
            n_sv = len(struct_verts)
            struct_colors = np.full((n_sv, 3), [180, 200, 220], dtype=np.uint8)
            struct_normals = np.zeros((n_sv, 3), dtype=np.float64)

            struct_geom = SceneGeometry(
                vertices=struct_verts,
                triangles=struct_tris,
                normals=struct_normals,
                colors=struct_colors,
                evidence_tags=np.full(n_sv, EvidenceSource.GEOMETRIC_INFERRED.value),
                categories=np.array(all_extrap_cats, dtype=object),
            )
            struct_path = self.job_dir / "structural_mesh.ply"
            struct_geom.write_ply(struct_path)
            manifest["structural_mesh_ply"] = str(struct_path)
            manifest["extrapolated_verts"] = int(n_sv)
            manifest["extrapolated_tris"] = int(len(struct_tris))
        else:
            manifest["structural_mesh_ply"] = None

        # Copy or preserve observed_mesh.ply
        src_mesh = observed_mesh_path or (self.job_dir / "mesh.ply")
        obs_path = self.job_dir / "observed_mesh.ply"
        if src_mesh.exists() and not obs_path.exists():
            import shutil
            shutil.copyfile(src_mesh, obs_path)
        manifest["observed_mesh_ply"] = str(obs_path)

        # Save manifest
        manifest_path = self.job_dir / "structural_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest


def execute_structural_completion(job_dir: Path) -> Dict[str, Any]:
    """Execute conservative structural completion stage on completed job."""
    job_dir = Path(job_dir)
    points_bin = job_dir / "points.bin"
    if not points_bin.exists():
        return {"stage": "structural_completion", "ok": False, "reason": "missing points.bin"}

    raw = np.fromfile(points_bin, dtype=np.uint8)
    n = len(raw) // 15
    raw = raw[: n * 15].reshape(n, 15)
    points = raw[:, 0:12].copy().view("<f4").reshape(n, 3).astype(np.float64)

    extrapolator = ConservativeExtrapolator(job_dir)
    manifest = extrapolator.process(points)
    return {"stage": "structural_completion", "ok": True, "manifest": manifest}
