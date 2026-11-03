"""3DGS Gaussian splat -> coloured point cloud adapter kernel.

    from echo_super.kernels.gaussian_splat import register_gaussian_kernels
    reg = register_gaussian_kernels()          # fresh Registry, 1 kernel
    reg = register_gaussian_kernels(other)     # or add into an existing one

Contract:
  consumes : "gaussians" — wrap the .ply path explicitly,
             wrap(path, kind="gaussians"), because infer_kind maps
             .ply to "points" (which is what this kernel produces).
             The payload may also be a raw property dict as returned
             by ply_reader.read_ply_vertices / core.read_gaussian_props.
  produces : "points" — data {"points": (N,3) float64,
             "colors": (N,3) uint8}, the exact input contract of the
             frozen voxel solidification (echo_scene.voxels
             .build_voxel_mesh) and of the poisson adapter.
  Parameter: meta["min_alpha"] (float, default 0.0) — Gaussians whose
             sigmoid(opacity) falls below it are dropped; 0 keeps all.

Colour comes from the SH DC term only (core.SH_C0 relation); positions
are the Gaussian centres, untouched and in file order. The splat itself
is a fitted model artifact, so the evidence grade is 生成； the adapter
conversion is verified value-for-value against hand-built fixtures
(tests/test_gaussian_adapter.py), hence status 已验.

The PLY reader (ply_reader.py) is DS Flash work (WO-gs-ply-reader),
Muse-reviewed; the conversion core is Muse's.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...core import EchoObject
from ...registry import Kernel, Registry
from .core import gaussians_to_points, load_gaussian_ply

KERNEL_META = {
    "gaussian_to_points": {"evidence": "生成", "status": "已验"},
}


def _gaussian_to_points_kernel(gaussians_obj: EchoObject):
    data = gaussians_obj.data
    min_alpha = float(gaussians_obj.meta.get("min_alpha", 0.0))
    if isinstance(data, (str, Path)):
        out = load_gaussian_ply(data, min_alpha=min_alpha)
        points, colors = out["points"], out["colors"]
        n_in = out["n_vertices"]
    elif isinstance(data, dict):
        points, colors, kept = gaussians_to_points(data, min_alpha=min_alpha)
        n_in = int(len(data["x"]))
    else:
        raise ValueError(
            "gaussian_to_points needs a .ply path or a gaussian "
            f"property dict, got {type(data).__name__}")
    payload = {"points": np.asarray(points), "colors": np.asarray(colors)}
    return gaussians_obj.derive(
        "points", payload, source="kernel:gaussian_to_points",
        n_gaussians_in=n_in, n_points_out=int(len(points)),
        **KERNEL_META["gaussian_to_points"])


def register_gaussian_kernels(registry: Registry | None = None) -> Registry:
    """Register the 3DGS adapter kernel; returns the registry."""
    reg = registry if registry is not None else Registry()
    reg.add(Kernel("gaussian_to_points", ("gaussians",), "points",
                   _gaussian_to_points_kernel, cost=1.0))
    return reg
