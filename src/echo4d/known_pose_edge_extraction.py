"""Opt-in TSDF extraction at *observed* sign-changing voxel edges.

The default eight-corner extractor remains untouched.  This adapter keeps
every old fully-observed-cell triangle and admits extra triangles only if
all three marching-cubes vertices lie on sign-changing edges whose TWO
endpoints were actually integrated.  Additional triangles must be edge-
connected to an original triangle.  Unknown TSDF (+1) is never accepted as
a depth measurement or a supported edge endpoint.

This is a topology experiment, not a new depth estimator or RGB inpainting.
"""
from __future__ import annotations

import numpy as np
from skimage.measure import marching_cubes

from .fusion import Mesh, TSDFVolume
from .model import DataError


def _observed_edge_vertices(vertices: np.ndarray, seen: np.ndarray,
                            sdf: np.ndarray) -> np.ndarray:
    """A crossing vertex needs two observed endpoints with opposite TSDF signs."""
    shape = np.asarray(sdf.shape)
    lo = np.clip(np.floor(vertices + 1e-5).astype(np.int64), 0, shape - 1)
    hi = np.clip(np.ceil(vertices - 1e-5).astype(np.int64), 0, shape - 1)
    one_axis = np.count_nonzero(lo != hi, axis=1) == 1
    return (one_axis & seen[tuple(lo.T)] & seen[tuple(hi.T)] &
            (sdf[tuple(lo.T)] * sdf[tuple(hi.T)] < 0))


def _attached_faces(faces: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Reject newly extracted face islands with no shared edge to old geometry."""
    parent = np.arange(len(faces), dtype=np.int64)
    size = np.ones(len(faces), dtype=np.int64)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    def join(a: int, b: int) -> None:
        a, b = root(a), root(b)
        if a == b:
            return
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]

    owners: dict[tuple[int, int], int] = {}
    for i, face in enumerate(faces):
        a, b, c = (int(x) for x in face)
        for u, v in ((a, b), (b, c), (c, a)):
            edge = (min(u, v), max(u, v))
            if edge in owners:
                join(i, owners[edge])
            else:
                owners[edge] = i
    groups = np.fromiter((root(i) for i in range(len(faces))),
                         dtype=np.int64, count=len(faces))
    original_groups = np.unique(groups[original])
    return np.isin(groups, original_groups)


def extract_observed_edge_mesh(volume: TSDFVolume) -> tuple[Mesh, dict]:
    """Extract a Mesh without changing TSDF integration or inventing depth."""
    cfg = volume.config
    sdf = volume.tsdf
    seen = volume.weight >= cfg.min_observations
    if not np.any(seen & (sdf < 0)) or not np.any(seen & (sdf > 0)):
        raise DataError("No observed TSDF zero crossing")
    try:
        # Unmasked marching cubes is needed to enumerate partly observed
        # cells.  Its unknown-boundary triangles are rejected below.
        vertices, faces, normals, _ = marching_cubes(
            sdf, 0, gradient_direction="ascent", allow_degenerate=False)
    except (ValueError, RuntimeError) as exc:
        raise DataError(f"No candidate TSDF surface: {exc}") from exc
    shape = volume.shape
    all_eight = np.ones(tuple(n - 1 for n in shape), dtype=bool)
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                all_eight &= seen[x:x + shape[0] - 1,
                                  y:y + shape[1] - 1,
                                  z:z + shape[2] - 1]
    cells = np.floor(vertices[faces].mean(axis=1) + 1e-5).astype(np.int64)
    cells = np.clip(cells, 0, np.asarray(shape) - 2)
    original = all_eight[tuple(cells.T)]
    edge_supported = _observed_edge_vertices(vertices, seen, sdf)
    candidate = original | np.all(edge_supported[faces], axis=1)
    if not np.any(original):
        raise DataError("No fully observed original TSDF triangles to anchor extension")
    selected = faces[candidate]
    old_selected = original[candidate]
    attached = _attached_faces(selected, old_selected)
    selected = selected[attached]
    if not len(selected):
        raise DataError("No supported attached triangles")
    original_count = int(np.count_nonzero(original))
    added_count = int(len(selected) - original_count)
    used, inverse = np.unique(selected, return_inverse=True)
    vertices = vertices[used]
    normals = normals[used]
    selected = inverse.reshape(-1, 3).astype(np.int32)
    nearest = np.clip(np.rint(vertices).astype(np.int64), 0, np.asarray(shape) - 1)
    indices = tuple(nearest.T)
    if not np.all(seen[indices]):
        raise DataError("Extracted vertex lacks nearest-voxel observation support")
    rgb = volume.color[indices].copy()
    rgb[volume.color_weight[indices] == 0] = [100, 154, 157]
    mesh = Mesh((volume.origin + vertices * cfg.voxel).astype(np.float32),
                selected, normals,
                np.clip(np.rint(rgb), 0, 255).astype(np.uint8),
                volume.weight[indices].copy(),
                volume.first[indices].copy(), volume.last[indices].copy())
    mesh.validate()
    stats = {
        "baseline_fully_observed_faces": original_count,
        "candidate_observed_edge_faces": int(np.count_nonzero(candidate & ~original)),
        "attached_added_faces": added_count,
        "rejected_detached_added_faces": int(np.count_nonzero(candidate) -
                                             len(selected)),
        "output_faces": int(len(selected)),
        "unknown_endpoint_accepted": False,
        "integration_unchanged": True,
    }
    return mesh, stats
