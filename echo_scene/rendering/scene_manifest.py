"""Normalized shared scene asset specification and manifest builder.

Ensures real-time viewer and offline high-quality renderer consume identical
geometry, cameras, lighting definitions, scale, and procedural materials.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class CameraSpecification:
    """Standard camera intrinsics and world extrinsics."""
    name: str
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    center: List[float]
    rotation: List[List[float]]
    fov_degrees: float = 60.0
    is_free_view: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LightSpecification:
    """Physical light source definition."""
    type: str  # 'directional', 'ambient', 'point'
    color: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    intensity: float = 1.0
    direction: Optional[List[float]] = None
    position: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SceneManifest:
    """Normalized scene manifest contract shared across all renderers."""
    schema_version: str = "1.0.0"
    job_id: str = ""
    geometry: Dict[str, Any] = field(default_factory=dict)
    materials: Dict[str, Any] = field(default_factory=dict)
    cameras: List[CameraSpecification] = field(default_factory=list)
    lighting: List[LightSpecification] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "geometry": self.geometry,
            "materials": self.materials,
            "cameras": [c.to_dict() for c in self.cameras],
            "lighting": [l.to_dict() for l in self.lighting],
            "metadata": self.metadata,
            "provenance": self.provenance,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SceneManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cameras = [CameraSpecification(**c) for c in data.get("cameras", [])]
        lighting = [LightSpecification(**l) for l in data.get("lighting", [])]
        return cls(
            schema_version=data.get("schema_version", "1.0.0"),
            job_id=data.get("job_id", ""),
            geometry=data.get("geometry", {}),
            materials=data.get("materials", {}),
            cameras=cameras,
            lighting=lighting,
            metadata=data.get("metadata", {}),
            provenance=data.get("provenance", {}),
        )


def build_scene_manifest(job_dir: Path) -> SceneManifest:
    """Construct complete SceneManifest from a reconstructed job directory."""
    job_dir = Path(job_dir)
    job_id = job_dir.name

    # 1. Geometry paths
    geometry: Dict[str, Any] = {}
    obs_mesh = job_dir / "observed_mesh.ply"
    if not obs_mesh.exists():
        obs_mesh = job_dir / "mesh.ply"
    if obs_mesh.exists():
        geometry["observed_mesh"] = str(obs_mesh)

    struct_mesh = job_dir / "structural_mesh.ply"
    if struct_mesh.exists():
        geometry["structural_mesh"] = str(struct_mesh)

    points_bin = job_dir / "points.bin"
    if points_bin.exists():
        geometry["points_bin"] = str(points_bin)

    voxel_glb = job_dir / "voxel_model.glb"
    if voxel_glb.exists():
        geometry["voxel_model_glb"] = str(voxel_glb)

    # 2. Cameras: 3 source camera views + 2 free exploration views
    cameras: List[CameraSpecification] = []
    poses_path = job_dir / "poses.json"
    align_path = job_dir / "alignment.json"
    align_rot = None
    if align_path.exists():
        try:
            adata = json.loads(align_path.read_text(encoding="utf-8"))
            align_rot = np.array(adata.get("rotation", np.eye(3)), dtype=np.float64)
        except Exception:
            pass

    source_centers = []

    if poses_path.exists():
        try:
            poses = json.loads(poses_path.read_text(encoding="utf-8"))
            frames = poses.get("frames", poses)
            sorted_names = sorted(frames.keys())
            if sorted_names:
                # Pick beginning, middle, and near end
                n = len(sorted_names)
                picks = [0, n // 2, max(0, n - 2)]
                for idx, pick_idx in enumerate(picks):
                    fname = sorted_names[pick_idx]
                    meta = frames[fname]
                    if "cam_from_world_quat_xyzw" in meta:
                        q = meta["cam_from_world_quat_xyzw"]
                        x, y, z, w = q
                        r_mat = [
                            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                        ]
                        t_vec = np.array(meta["cam_from_world_t"], dtype=np.float64)
                        raw_c = -np.dot(np.array(r_mat).T, t_vec)
                        if align_rot is not None:
                            center = np.dot(raw_c, align_rot.T).tolist()
                        else:
                            center = raw_c.tolist()
                        cam_spec = meta.get("camera", {})
                        params = cam_spec.get("params", [768.0, 320.0, 240.0])
                        width = cam_spec.get("width", 640)
                        height = cam_spec.get("height", 480)
                        fx = params[0] if len(params) > 0 else 768.0
                        fy = params[0] if len(params) > 0 else 768.0
                        cx = params[1] if len(params) > 1 else width / 2.0
                        cy = params[2] if len(params) > 2 else height / 2.0
                    else:
                        center = meta.get("center", [0.0, 1.0, -2.0])
                        r_mat = meta.get("rotation", np.eye(3).tolist())
                        width = meta.get("width", 640)
                        height = meta.get("height", 480)
                        fx = fy = 768.0
                        cx = width / 2.0
                        cy = height / 2.0

                    source_centers.append(center)
                    cameras.append(CameraSpecification(
                        name=f"source_view_{idx}_{fname}",
                        fx=float(fx),
                        fy=float(fy),
                        cx=float(cx),
                        cy=float(cy),
                        width=int(width),
                        height=int(height),
                        center=[float(c) for c in center],
                        rotation=r_mat,
                        is_free_view=False,
                    ))
        except Exception:
            pass

    # Free View 1: 45 degree elevated orbit
    avg_c = np.mean(source_centers, axis=0) if source_centers else np.array([0.0, 1.0, 0.0])
    cameras.append(CameraSpecification(
        name="free_view_1_elevated_orbit",
        fx=800.0, fy=800.0, cx=640.0, cy=360.0, width=1280, height=720,
        center=[float(avg_c[0] + 1.8), float(avg_c[1] + 2.2), float(avg_c[2] - 2.5)],
        rotation=np.eye(3).tolist(),
        is_free_view=True,
    ))

    # Free View 2: eye-level conversational perspective
    cameras.append(CameraSpecification(
        name="free_view_2_eye_level",
        fx=800.0, fy=800.0, cx=640.0, cy=360.0, width=1280, height=720,
        center=[float(avg_c[0] - 1.2), float(avg_c[1] + 0.4), float(avg_c[2] - 1.5)],
        rotation=np.eye(3).tolist(),
        is_free_view=True,
    ))

    # 3. Standard lighting
    lighting = [
        LightSpecification(
            type="directional",
            color=[1.0, 0.98, 0.95],
            intensity=2.5,
            direction=[-0.5, -1.0, -0.6],
        ),
        LightSpecification(
            type="directional",
            color=[0.7, 0.8, 1.0],
            intensity=1.0,
            direction=[0.6, -0.4, 0.7],
        ),
        LightSpecification(
            type="ambient",
            color=[0.9, 0.9, 0.95],
            intensity=0.8,
        ),
    ]

    manifest = SceneManifest(
        job_id=job_id,
        geometry=geometry,
        cameras=cameras,
        lighting=lighting,
        metadata={"up_vector": [0.0, 1.0, 0.0], "units": "meters"},
    )
    manifest.save(job_dir / "scene_manifest.json")
    return manifest
