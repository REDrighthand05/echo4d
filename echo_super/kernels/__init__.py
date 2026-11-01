"""Kernel packages that live outside adapters.py.

Currently: Echo Render (echo_render/) — the native CPU point/line renderer,
registered here following the adapters.py pattern (an EchoObject-level
wrapper + a register_* function + an evidence entry).

    from echo_super.kernels import register_echo_render_kernels
    reg = register_echo_render_kernels()          # fresh Registry, 1 kernel
    reg = register_echo_render_kernels(other)     # or add into an existing one

Payload conventions (EchoObject.data) for echo_render_frame:
  points : (N,3) float array, or a dict with keys
             "points"  (N,3) float            (required)
             "colors"  (N,3) uint8            (optional; soft default)
             "lines"   (M,2,3) float          (optional)
             "lines_rgb" (M,3) uint8          (optional)
             "camera"  {"K","R","t"}          (or meta["camera"])
  Render parameters ride in meta: width (1280), height (720),
  point_radius_px (2.0), glow_strength (0.6), background_rgb
  ((5, 7, 14)). Produces a "frame" object: (H, W, 3) uint8 array.

Evidence grade: 实测 · 已验 — Echo Render introduces no geometry of its own;
it rasterizes the points it is given, verified pixel-exact against
hand-computed projections and z-order on synthetic ground truth
(tests/test_echo_render.py).
"""
from __future__ import annotations

import numpy as np

from ..core import EchoObject
from ..registry import Kernel, Registry

KERNEL_META = {
    "echo_render_frame": {"evidence": "实测", "status": "已验"},
}

_DEFAULT_RGB = (205, 215, 230)


def _echo_render_kernel(points_obj: EchoObject):
    from .echo_render import render_frame  # lazy: numba import is slow

    data = points_obj.data
    lines = lines_rgb = camera = None
    if isinstance(data, dict):
        pts = np.asarray(data["points"])
        cols = data.get("colors")
        lines = data.get("lines")
        lines_rgb = data.get("lines_rgb")
        camera = data.get("camera")
    else:
        pts = np.asarray(data)
        cols = None
    if camera is None:
        camera = points_obj.meta.get("camera")
    if camera is None:
        raise ValueError(
            "echo_render_frame needs a camera: points.data['camera'] or "
            "points.meta['camera'] = {'K': ..., 'R': ..., 't': ...}")
    if cols is None:
        cols = np.full((pts.shape[0], 3), _DEFAULT_RGB, dtype=np.uint8)
    meta = points_obj.meta
    img = render_frame(
        pts, cols, lines, lines_rgb, camera=camera,
        width=int(meta.get("width", 1280)),
        height=int(meta.get("height", 720)),
        point_radius_px=float(meta.get("point_radius_px", 2.0)),
        glow_strength=float(meta.get("glow_strength", 0.6)),
        background_rgb=tuple(meta.get("background_rgb", (5, 7, 14))))
    return points_obj.derive("frame", img, source="kernel:echo_render_frame",
                             **KERNEL_META["echo_render_frame"])


def register_echo_render_kernels(registry: Registry | None = None) -> Registry:
    """Register the Echo Render kernel; returns the registry."""
    reg = registry if registry is not None else Registry()
    reg.add(Kernel("echo_render_frame", ("points",), "frame",
                   _echo_render_kernel, cost=1.0))
    return reg
