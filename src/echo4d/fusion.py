"""Bounded CPU projective TSDF; known poses, static scene or supplied masks."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from skimage.measure import marching_cubes
from .model import Camera, DataError, Frame


@dataclass(frozen=True)
class FusionConfig:
    bounds: tuple[float, float, float, float, float, float]
    voxel: float = 0.04
    truncation: float = 0.12
    min_depth: float = 0.15
    max_depth: float = 8.0
    max_voxels: int = 4_000_000
    chunk_size: int = 131072
    min_observations: int = 1

    def __post_init__(self) -> None:
        if len(self.bounds) != 6 or not all(math.isfinite(v) and abs(v) <= 10000 for v in self.bounds):
            raise DataError("Bounds need six finite numbers within 10 km of the origin")
        if any(self.bounds[i + 3] <= self.bounds[i] for i in range(3)):
            raise DataError("Every maximum bound must exceed its minimum")
        for value in (self.voxel, self.truncation, self.min_depth, self.max_depth):
            if not math.isfinite(value) or value <= 0:
                raise DataError("Fusion distances must be finite and positive")
        if not 1e-4 <= self.voxel <= 100:
            raise DataError("voxel must be in [0.0001,100] metres")
        if self.truncation < self.voxel or self.max_depth <= self.min_depth:
            raise DataError("truncation must be >= voxel; max_depth must exceed min_depth")
        if type(self.min_observations) is not int or not 1 <= self.min_observations <= 100000:
            raise DataError("min_observations must be an integer in [1,100000]")
        if type(self.max_voxels) is not int or not 8 <= self.max_voxels <= 32_000_000:
            raise DataError("max_voxels must be in [8,32000000]")
        if type(self.chunk_size) is not int or not 1024 <= self.chunk_size <= 1_000_000:
            raise DataError("chunk_size must be in [1024,1000000]")
        shape = self.shape
        if any(n < 2 for n in shape) or math.prod(shape) > self.max_voxels:
            raise DataError(f"Grid {shape} needs {math.prod(shape):,} voxels; limit {self.max_voxels:,}. Enlarge voxel or shrink bounds.")

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(math.ceil((self.bounds[i + 3] - self.bounds[i]) / self.voxel) for i in range(3))


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    colors: np.ndarray
    observations: np.ndarray
    first_frame: np.ndarray
    last_frame: np.ndarray

    def validate(self) -> None:
        n = len(self.vertices)
        if not n or not len(self.faces) or self.vertices.shape != (n, 3) or not np.isfinite(self.vertices).all():
            raise DataError("No valid observed surface was reconstructed")
        if self.faces.shape[1:] != (3,) or self.faces.min() < 0 or self.faces.max() >= n:
            raise DataError("Invalid mesh topology")
        if self.colors.shape != (n, 3) or self.normals.shape != (n, 3):
            raise DataError("Invalid vertex attributes")
        if not np.isfinite(self.normals).all() or np.any(self.observations < 1):
            raise DataError("Invalid vertex support")


class TSDFVolume:
    def __init__(self, config: FusionConfig):
        self.config = config
        self.shape = config.shape
        self.origin = np.array(config.bounds[:3], dtype=np.float64) + config.voxel / 2
        self.tsdf = np.ones(self.shape, dtype=np.float32)
        self.weight = np.zeros(self.shape, dtype=np.uint32)
        self.color = np.zeros((*self.shape, 3), dtype=np.float32)
        self.color_weight = np.zeros(self.shape, dtype=np.uint32)
        self.first = np.full(self.shape, -1, dtype=np.int32)
        self.last = np.full(self.shape, -1, dtype=np.int32)
        self.integrated = 0
        self.last_index = -1
        self.last_t = -math.inf

    @property
    def allocated_bytes(self) -> int:
        return sum(a.nbytes for a in (self.tsdf, self.weight, self.color, self.color_weight, self.first, self.last))

    def integrate(self, frame: Frame, camera: Camera, frame_index: int,
                  cancelled: Callable[[], bool] | None = None) -> int:
        frame.validate(camera)
        if type(frame_index) is not int or frame_index <= self.last_index or frame_index > 2_000_000_000:
            raise DataError("Frame indices must increase and fit int32")
        if frame.t <= self.last_t:
            raise DataError("Integrated timestamps must strictly increase")
        cfg = self.config
        total = math.prod(self.shape)
        rotation, translation = frame.camera_to_world[:3, :3], frame.camera_to_world[:3, 3]
        sdf, weights = self.tsdf.ravel(), self.weight.ravel()
        first, last = self.first.ravel(), self.last.ravel()
        colors, color_weights = self.color.reshape(-1, 3), self.color_weight.ravel()
        updated = 0
        for start in range(0, total, cfg.chunk_size):
            if cancelled is not None and cancelled():
                raise InterruptedError("Fusion cancelled; discard this partial volume")
            ids = np.arange(start, min(total, start + cfg.chunk_size))
            coords = np.column_stack(np.unravel_index(ids, self.shape))
            world = self.origin + coords * cfg.voxel
            cam = (world - translation) @ rotation
            z = cam[:, 2]
            front = z > cfg.min_depth
            ids, cam, z = ids[front], cam[front], z[front]
            u = np.floor(camera.fx * cam[:, 0] / z + camera.cx + 0.5).astype(np.int64)
            v = np.floor(camera.fy * cam[:, 1] / z + camera.cy + 0.5).astype(np.int64)
            inside = (u >= 0) & (v >= 0) & (u < camera.width) & (v < camera.height)
            ids, u, v, z = ids[inside], u[inside], v[inside], z[inside]
            measured = frame.depth[v, u]
            distance = measured - z
            valid = (measured >= cfg.min_depth) & (measured <= cfg.max_depth) & (distance >= -cfg.truncation)
            if frame.excluded is not None:
                valid &= ~frame.excluded[v, u]
            ids, u, v, distance = ids[valid], u[valid], v[valid], distance[valid]
            if not len(ids):
                continue
            old = weights[ids].astype(np.float64)
            sdf[ids] = (sdf[ids] * old + np.clip(distance / cfg.truncation, -1, 1)) / (old + 1)
            first[ids[old == 0]] = frame_index
            last[ids] = frame_index
            weights[ids] += 1
            updated += len(ids)
            if frame.color is not None:
                close = np.abs(distance) <= cfg.truncation
                cid, cu, cv = ids[close], u[close], v[close]
                cw = color_weights[cid].astype(np.float64)
                colors[cid] = (colors[cid] * cw[:, None] + frame.color[cv, cu]) / (cw[:, None] + 1)
                color_weights[cid] += 1
        self.integrated += 1
        self.last_index, self.last_t = frame_index, frame.t
        return updated

    def extract_mesh(self) -> Mesh:
        cfg = self.config
        seen = self.weight >= cfg.min_observations
        if not np.any(seen & (self.tsdf < 0)) or not np.any(seen & (self.tsdf > 0)):
            raise DataError("No supported zero crossing: check bounds, poses, depth scale, and observations")
        # Requiring all eight observed corners avoids surfaces at unknown (+1) boundaries.
        valid_cells = np.ones(tuple(n - 1 for n in self.shape), dtype=bool)
        for x in (0, 1):
            for y in (0, 1):
                for z in (0, 1):
                    valid_cells &= seen[x:x+self.shape[0]-1, y:y+self.shape[1]-1, z:z+self.shape[2]-1]
        try:
            vertices, faces, normals, _ = marching_cubes(self.tsdf, 0, mask=seen,
                                                        gradient_direction="ascent", allow_degenerate=False)
        except (ValueError, RuntimeError) as error:
            raise DataError(f"No extractable surface: {error}") from error
        cells = np.floor(vertices[faces].mean(axis=1) + 1e-5).astype(np.int64)
        cells = np.clip(cells, 0, np.array(self.shape) - 2)
        faces = faces[valid_cells[tuple(cells.T)]]
        if not len(faces):
            raise DataError("No triangles have fully observed support; widen views or reduce min_observations")
        used, inverse = np.unique(faces, return_inverse=True)
        vertices, normals = vertices[used], normals[used]
        faces = inverse.reshape(-1, 3).astype(np.int32)
        nearest = np.clip(np.rint(vertices).astype(np.int64), 0, np.array(self.shape) - 1)
        indices = tuple(nearest.T)
        rgb = self.color[indices].copy()
        rgb[self.color_weight[indices] == 0] = [100, 154, 157]
        mesh = Mesh((self.origin + vertices * cfg.voxel).astype(np.float32), faces, normals,
                    np.clip(np.rint(rgb), 0, 255).astype(np.uint8), self.weight[indices].copy(),
                    self.first[indices].copy(), self.last[indices].copy())
        mesh.validate()
        return mesh
