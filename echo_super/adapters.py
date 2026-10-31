"""Real kernels — adapters wrapping verified implementations for the registry.

Three kernels graduate here from "verified on the bench" to "registered",
with the exact kind declarations of their CATALOG.md rows:

    orient_normals       points + normals -> normals   实测 · 已验
    fit_plane            points -> points (plane params in meta)  实测 · 已验
    poisson_reconstruct  points + normals -> mesh      实测 · 已验候选

Usage (registration is explicit, nothing happens at import time):

    from echo_super.adapters import register_real_kernels
    reg = register_real_kernels()            # fresh Registry, 3 real kernels
    reg = register_real_kernels(other_reg)   # or add into an existing one

Payload conventions (EchoObject.data):
  points   : (N,3) float array. For orient_normals the per-point source
             camera centers ride along, either as
             points.data == {"points": arr, "cam_centers": arr} or as
             points.meta["cam_centers"].
             For poisson_reconstruct, vertex colors may ride as
             points.data == {"points": arr, "colors": arr} ((N,3), float
             in [0,1] or uint8 in [0,255]).
  normals  : (N,3) float array (or {"normals": arr}).
  mesh     : {"vertices": (V,3), "triangles": (F,3), "vertex_colors": (V,3)
             or an empty (0,3) array when no colors were given}.

Every kernel returns an EchoObject whose meta carries the evidence grade
and review status (see KERNEL_META) next to its own outputs.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from .core import EchoObject
from .registry import Kernel, Registry

# Evidence grade + review status per kernel, mirrored into the meta of
# every EchoObject these kernels return.
KERNEL_META = {
    "orient_normals": {"evidence": "实测", "status": "已验"},
    "fit_plane": {"evidence": "实测", "status": "已验"},
    "poisson_reconstruct": {"evidence": "实测", "status": "已验候选"},
}


def _load_ds_module(name: str, filename: str):
    """Load a DS-delivered file from workorders/ by path (not a package)."""
    path = Path(__file__).resolve().parent / "workorders" / filename
    spec = importlib.util.spec_from_file_location(f"_echo_super_ds_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_orient_mod = _load_ds_module("orient_normals", "orient_normals_ds.py")
_plane_mod = _load_ds_module("fit_plane", "fit_plane_ds.py")

orient_normals = _orient_mod.orient_normals
fit_plane = _plane_mod.fit_plane


def poisson_reconstruct(points, normals, colors=None):
    """Kazhdan Poisson surface reconstruction (Open3D), density-trimmed.

    Inputs are plain numpy arrays, not EchoObjects:
        points   (N,3) float
        normals  (N,3) float, same N as points (orient them first,
                 e.g. with orient_normals)
        colors   optional (N,3): float in [0,1] or uint8 in [0,255]

    Calls open3d's create_from_point_cloud_poisson(depth=9), then drops
    every vertex whose density falls below the 15th percentile of the
    returned densities (the low-density fringe is where the implicit
    surface hallucinates), together with its triangles.

    Returns (vertices, triangles, vertex_colors): (V,3) float,
    (F,3) int, and (V,3) float in [0,1] — or an empty (0,3) array when
    no input colors were given.

    Raises ValueError on shape mismatch, empty input or non-finite data.
    """
    import open3d as o3d  # lazy: importing adapters must not require open3d

    pts = np.asarray(points, dtype=float)
    nrms = np.asarray(normals, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be an (N, 3) array")
    if nrms.shape != pts.shape:
        raise ValueError("normals must have the same (N, 3) shape as points")
    if pts.shape[0] == 0:
        raise ValueError("points must contain at least one point (N > 0)")
    if not np.isfinite(pts).all() or not np.isfinite(nrms).all():
        raise ValueError("points/normals contain non-finite values")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(nrms)
    if colors is not None:
        cols = np.asarray(colors)
        if cols.shape != pts.shape:
            raise ValueError("colors must have the same (N, 3) shape as points")
        if cols.dtype == np.uint8:
            cols = cols.astype(float) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(cols, dtype=float))

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9)
    densities = np.asarray(densities, dtype=float)
    keep = densities >= np.percentile(densities, 15)
    mesh.remove_vertices_by_mask(~keep)

    vertices = np.asarray(mesh.vertices, dtype=float)
    triangles = np.asarray(mesh.triangles, dtype=int)
    vertex_colors = np.asarray(mesh.vertex_colors, dtype=float).reshape(-1, 3)
    return vertices, triangles, vertex_colors


# ---------------------------------------------------------------------------
# EchoObject-level kernel functions (what the registry actually calls)

def _points_array(obj: EchoObject):
    """(points, colors-or-None, cam_centers-or-None) from a points object."""
    data = obj.data
    if isinstance(data, dict):
        cams = data.get("cam_centers")
        return (np.asarray(data["points"]),
                data.get("colors"),
                cams if cams is not None else obj.meta.get("cam_centers"))
    return np.asarray(data), None, obj.meta.get("cam_centers")


def _normals_array(obj: EchoObject):
    data = obj.data
    if isinstance(data, dict):
        return np.asarray(data["normals"])
    return np.asarray(data)


def _orient_normals_kernel(points_obj: EchoObject, normals_obj: EchoObject):
    pts, _colors, cams = _points_array(points_obj)
    if cams is None:
        raise ValueError(
            "orient_normals needs per-point source camera centers: "
            "points.data={'points': arr, 'cam_centers': arr} or "
            "points.meta['cam_centers']")
    oriented = orient_normals(pts, _normals_array(normals_obj), np.asarray(cams))
    return normals_obj.derive("normals", oriented,
                              source="kernel:orient_normals",
                              **KERNEL_META["orient_normals"])


def _fit_plane_kernel(points_obj: EchoObject):
    pts, _colors, _cams = _points_array(points_obj)
    normal, d, rms = fit_plane(pts)
    return points_obj.derive(
        "points", points_obj.data, source="kernel:fit_plane",
        plane_normal=[float(v) for v in normal], plane_d=float(d),
        plane_rms=float(rms), **KERNEL_META["fit_plane"])


def _poisson_kernel(points_obj: EchoObject, normals_obj: EchoObject):
    pts, colors, _cams = _points_array(points_obj)
    vertices, triangles, vertex_colors = poisson_reconstruct(
        pts, _normals_array(normals_obj), colors)
    return points_obj.derive(
        "mesh",
        {"vertices": vertices, "triangles": triangles,
         "vertex_colors": vertex_colors},
        source="kernel:poisson_reconstruct",
        **KERNEL_META["poisson_reconstruct"])


def register_real_kernels(registry: Registry | None = None) -> Registry:
    """Register the three verified real kernels; returns the registry.

    With no argument a fresh Registry is created; pass an existing one
    (e.g. the demo registry) to add the real kernels alongside it.
    """
    reg = registry if registry is not None else Registry()
    reg.add(Kernel("orient_normals", ("points", "normals"), "normals",
                   _orient_normals_kernel, cost=1.0))
    reg.add(Kernel("fit_plane", ("points",), "points",
                   _fit_plane_kernel, cost=1.0))
    reg.add(Kernel("poisson_reconstruct", ("points", "normals"), "mesh",
                   _poisson_kernel, cost=2.0))
    return reg


# ---------------------------------------------------------------------------
# DS batch 1 (2026-10-31): three more verified kernels, registered after
# Muse's bench verification (echo_super/ds_batch1.py holds the code;
# specs in echo_super/workorders/WO-*.md):
#   estimate_scale         points + known_size -> scale   估计 · 已验
#   build_replay_path      poses -> path                  实测 · 已验
#   surface_voxel_indices  points -> points               实测 · 已验
# Bench results: scale exact on a synthetic box (0.255 vs 0.25 nominal);
# replay path interpolates its knots exactly, unit look dirs; surface
# filter exact on solid blocks (56/64) and 0.7 s on the 295k-point room
# cloud — which also proved the fused cloud is a hollow shell (100% of
# its cubes are surface), so this kernel cannot shrink that model.

from .ds_batch1 import (  # noqa: E402
    build_replay_path as _build_replay_path_fn,
    estimate_scale as _estimate_scale_fn,
    surface_voxel_indices as _surface_voxel_fn,
)

KERNEL_META.update({
    "estimate_scale": {"evidence": "估计", "status": "已验"},
    "build_replay_path": {"evidence": "实测", "status": "已验"},
    "surface_voxel_indices": {"evidence": "实测", "status": "已验"},
})


def _estimate_scale_kernel(points_obj: EchoObject, known_size_obj: EchoObject):
    pts, _colors, _cams = _points_array(points_obj)
    scale, measured = _estimate_scale_fn(pts, float(known_size_obj.data))
    return points_obj.derive(
        "scale", {"scale": float(scale), "measured_size": float(measured)},
        source="kernel:estimate_scale", **KERNEL_META["estimate_scale"])


def _build_replay_path_kernel(poses_obj: EchoObject):
    data = poses_obj.data
    path, look_dirs = _build_replay_path_fn(
        np.asarray(data["centers"]), np.asarray(data["look_targets"]),
        int(poses_obj.meta.get("samples_per_segment", 10)))
    return poses_obj.derive(
        "path", {"path": path, "look_dirs": look_dirs},
        source="kernel:build_replay_path", **KERNEL_META["build_replay_path"])


def _surface_voxel_kernel(points_obj: EchoObject):
    pts, _colors, _cams = _points_array(points_obj)
    cube = points_obj.meta.get("cube")
    if cube is None:
        raise ValueError("surface_voxel_indices needs points.meta['cube']")
    idx = _surface_voxel_fn(pts, float(cube))
    return points_obj.derive(
        "points", pts[idx], source="kernel:surface_voxel_indices",
        surface_indices=idx, **KERNEL_META["surface_voxel_indices"])


def register_ds_batch1(registry: Registry | None = None) -> Registry:
    """Register the DS batch-1 kernels; returns the registry."""
    reg = registry if registry is not None else Registry()
    reg.add(Kernel("estimate_scale", ("points", "known_size"), "scale",
                   _estimate_scale_kernel, cost=1.0))
    reg.add(Kernel("build_replay_path", ("poses",), "path",
                   _build_replay_path_kernel, cost=1.0))
    reg.add(Kernel("surface_voxel_indices", ("points",), "points",
                   _surface_voxel_kernel, cost=1.0))
    return reg
