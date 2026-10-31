"""Echo Render — a fast, deterministic CPU rasterizer for coloured 3D
points and line segments.

Echo Render (engine codename MUS) draws exactly the geometry it is
given: perspective projection, a
depth-ordered splat for points (soft glow disc + opaque core), and
1px anti-aliased (Xiaolin Wu) line segments beneath the points, over a
flat background. Nothing is estimated, averaged or voted on.

Determinism design: projection runs in parallel (numba ``prange``,
each point writes only its own slot); splatting then runs in one
fixed order — points sorted far-to-near with a stable sort — and the
final composite is a pure function of the accumulation buffer. Same
inputs always produce the same output bytes, on any core count.

Dependencies: numpy + numba (PIL only, lazily, for PNG output in
``render_sequence``). No file formats are known to this module beyond
``.npy`` arrays in the sequence driver; all geometry arrives as plain
numpy arrays plus a camera dict.

Camera convention: pinhole, ``Xc = R @ X + t`` (world -> camera),
``u = fx * Xc.x / Xc.z + cx``, ``v = fy * Yc / Xc.z + cy`` with
``fx, cx`` from ``K[0,0], K[0,2]`` and ``fy, cy`` from ``K[1,1],
K[1,2]``. Points with ``Xc.z <= near`` are culled.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numba import njit, prange

__all__ = ["render_frame", "render_sequence"]

_DEFAULT_LINE_RGB = (96.0, 165.0, 250.0)
_DEFAULT_POINT_RGB = (205, 215, 230)


# ---------------------------------------------------------------------------
# Compiled kernels

@njit(cache=True)
def _project(xyz, R, t, K, near, u, v, z, ok):
    """Project (N,3) world points to pixel floats; prange-parallel."""
    fx = K[0, 0]
    cx = K[0, 2]
    fy = K[1, 1]
    cy = K[1, 2]
    r00 = R[0, 0]; r01 = R[0, 1]; r02 = R[0, 2]
    r10 = R[1, 0]; r11 = R[1, 1]; r12 = R[1, 2]
    r20 = R[2, 0]; r21 = R[2, 1]; r22 = R[2, 2]
    t0 = t[0]; t1 = t[1]; t2 = t[2]
    n = xyz.shape[0]
    for i in prange(n):
        x = xyz[i, 0]; y = xyz[i, 1]; w = xyz[i, 2]
        xc = r00 * x + r01 * y + r02 * w + t0
        yc = r10 * x + r11 * y + r12 * w + t1
        zc = r20 * x + r21 * y + r22 * w + t2
        z[i] = zc
        if zc > near:
            ok[i] = True
            u[i] = fx * xc / zc + cx
            v[i] = fy * yc / zc + cy
        else:
            ok[i] = False
            u[i] = 0.0
            v[i] = 0.0


@njit(cache=True)
def _wu_plot(acc, x, y, r, g, b, alpha, steep, width, height):
    """Alpha-blend one Wu pixel; (x, y) are in line space."""
    if steep:
        px = y
        py = x
    else:
        px = x
        py = y
    if 0 <= px < width and 0 <= py < height:
        keep = 1.0 - alpha
        acc[py, px, 0] = acc[py, px, 0] * keep + r * alpha
        acc[py, px, 1] = acc[py, px, 1] * keep + g * alpha
        acc[py, px, 2] = acc[py, px, 2] * keep + b * alpha


@njit(cache=True)
def _wu_line(acc, x0, y0, x1, y1, r, g, b, width, height):
    """Xiaolin Wu anti-aliased segment, blended into ``acc``."""
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    dx = x1 - x0
    dy = y1 - y0
    if dx == 0.0:
        _wu_plot(acc, int(np.floor(x0 + 0.5)), int(np.floor(y0)),
                 r, g, b, 1.0, steep, width, height)
        return
    grad = dy / dx

    # First endpoint.
    xend = np.floor(x0 + 0.5)
    yend = y0 + grad * (xend - x0)
    xgap = 1.0 - ((x0 + 0.5) - np.floor(x0 + 0.5))  # rfpart(x0 + 0.5)
    xpxl1 = int(xend)
    ypxl1 = int(np.floor(yend))
    fpart = yend - np.floor(yend)
    _wu_plot(acc, xpxl1, ypxl1, r, g, b, (1.0 - fpart) * xgap,
             steep, width, height)
    _wu_plot(acc, xpxl1, ypxl1 + 1, r, g, b, fpart * xgap,
             steep, width, height)
    intery = yend + grad

    # Second endpoint.
    xend = np.floor(x1 + 0.5)
    yend = y1 + grad * (xend - x1)
    xgap = (x1 + 0.5) - np.floor(x1 + 0.5)  # fpart(x1 + 0.5)
    xpxl2 = int(xend)
    ypxl2 = int(np.floor(yend))
    fpart = yend - np.floor(yend)
    _wu_plot(acc, xpxl2, ypxl2, r, g, b, (1.0 - fpart) * xgap,
             steep, width, height)
    _wu_plot(acc, xpxl2, ypxl2 + 1, r, g, b, fpart * xgap,
             steep, width, height)

    # Interior.
    for x in range(xpxl1 + 1, xpxl2):
        ybase = int(np.floor(intery))
        fpart = intery - np.floor(intery)
        _wu_plot(acc, x, ybase, r, g, b, 1.0 - fpart, steep, width, height)
        _wu_plot(acc, x, ybase + 1, r, g, b, fpart, steep, width, height)
        intery += grad


@njit(cache=True)
def _splat(acc, u, v, rgb, order, rc, rg, glow_strength, width, height):
    """Draw points in the given (far-to-near) order.

    Sequential on purpose: the fixed order is what makes the output
    bit-deterministic. rc = core radius, rg = glow radius; when
    glow_strength == 0 the caller passes rg == rc and the glow branch
    never fires.
    """
    rc2 = rc * rc
    rg2 = rg * rg
    for k in range(order.shape[0]):
        i = order[k]
        cu = u[i]
        cv = v[i]
        cr = rgb[i, 0]
        cg = rgb[i, 1]
        cb = rgb[i, 2]
        x_lo = int(np.floor(cu - rg))
        if x_lo < 0:
            x_lo = 0
        x_hi = int(np.ceil(cu + rg))
        if x_hi > width - 1:
            x_hi = width - 1
        y_lo = int(np.floor(cv - rg))
        if y_lo < 0:
            y_lo = 0
        y_hi = int(np.ceil(cv + rg))
        if y_hi > height - 1:
            y_hi = height - 1
        if x_lo > x_hi or y_lo > y_hi:
            continue
        for py in range(y_lo, y_hi + 1):
            dy = (py + 0.5) - cv
            dy2 = dy * dy
            if dy2 > rg2:
                continue
            for px in range(x_lo, x_hi + 1):
                dx = (px + 0.5) - cu
                d2 = dx * dx + dy2
                if d2 <= rc2:
                    acc[py, px, 0] = cr
                    acc[py, px, 1] = cg
                    acc[py, px, 2] = cb
                elif d2 <= rg2 and glow_strength > 0.0:
                    d = np.sqrt(d2)
                    w = glow_strength * (1.0 - d / rg) ** 2
                    acc[py, px, 0] += cr * w
                    acc[py, px, 1] += cg * w
                    acc[py, px, 2] += cb * w


# ---------------------------------------------------------------------------
# Input normalisation

def _as_points(points_xyz):
    pts = np.ascontiguousarray(points_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(
            f"points_xyz must have shape (N, 3), got {pts.shape}")
    return pts


def _as_colours(values, n, name):
    arr = np.asarray(values)
    if arr.shape != (n, 3):
        raise ValueError(f"{name} must have shape ({n}, 3), got {arr.shape}")
    if arr.dtype.kind == "f":
        out = arr.astype(np.float64)
        if out.size and out.max() <= 1.0:
            out = out * 255.0
        return np.ascontiguousarray(out, dtype=np.float32)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _as_camera(camera):
    if camera is None:
        raise ValueError("camera is required: dict with K, R, t")
    try:
        K = np.asarray(camera["K"], dtype=np.float64)
        R = np.asarray(camera["R"], dtype=np.float64)
        t = np.asarray(camera["t"], dtype=np.float64)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"camera must be a dict with K, R, t ({exc})") from exc
    if K.shape != (3, 3) or R.shape != (3, 3) or t.shape != (3,):
        raise ValueError(
            f"camera shapes must be K (3,3), R (3,3), t (3,); "
            f"got {K.shape}, {R.shape}, {t.shape}")
    return K, R, t


# ---------------------------------------------------------------------------
# Public API

def render_frame(points_xyz, points_rgb, lines_xyz=None, lines_rgb=None,
                 camera=None, width=1280, height=720,
                 point_radius_px=2.0, glow_strength=0.6,
                 glow_radius_factor=2.5, background_rgb=(5, 7, 14),
                 near=1e-4):
    """Render one frame of coloured points (+ optional line segments).

    Parameters
    ----------
    points_xyz : (N, 3) array-like of float — world-space points.
    points_rgb : (N, 3) array-like — uint8 colours 0..255, or float
        colours in [0, 1] (scaled to 0..255).
    lines_xyz : (M, 2, 3) array-like of float, optional — segment
        endpoints in world space. Drawn under the points; a segment is
        skipped if either endpoint is behind the near plane.
    lines_rgb : (M, 3) array-like, optional — segment colours; default
        is a light blue (96, 165, 250) for every segment.
    camera : dict {"K": (3,3), "R": (3,3), "t": (3,)} — pinhole
        intrinsics and world->camera rotation/translation. Required.
    width, height : output image size in pixels.
    point_radius_px : radius of each point's opaque core disc, in
        pixels (clamped to >= 0.5 so every visible point lands).
    glow_strength : 0..~1 scale of the additive glow halo around each
        point core; 0 disables the halo.
    glow_radius_factor : glow disc radius = core radius * this.
    background_rgb : (3,) background colour 0..255.
    near : points with camera-space z <= near are culled.

    Returns
    -------
    (height, width, 3) numpy.ndarray, dtype uint8.
    """
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    K, R, t = _as_camera(camera)
    pts = _as_points(points_xyz)
    n = pts.shape[0]
    rgb = _as_colours(points_rgb, n, "points_rgb")

    if lines_xyz is None:
        lines = np.zeros((0, 2, 3), dtype=np.float32)
    else:
        lines = np.ascontiguousarray(lines_xyz, dtype=np.float32)
        if lines.ndim != 3 or lines.shape[1] != 2 or lines.shape[2] != 3:
            raise ValueError(
                f"lines_xyz must have shape (M, 2, 3), got {lines.shape}")
    m = lines.shape[0]
    if lines_rgb is None:
        lrgb = np.empty((m, 3), dtype=np.float32)
        lrgb[:, 0] = _DEFAULT_LINE_RGB[0]
        lrgb[:, 1] = _DEFAULT_LINE_RGB[1]
        lrgb[:, 2] = _DEFAULT_LINE_RGB[2]
    else:
        lrgb = _as_colours(lines_rgb, m, "lines_rgb")

    bg = np.asarray(background_rgb, dtype=np.float32)
    if bg.shape != (3,):
        raise ValueError("background_rgb must have exactly 3 components")
    acc = np.empty((height, width, 3), dtype=np.float32)
    acc[:, :, 0] = bg[0]
    acc[:, :, 1] = bg[1]
    acc[:, :, 2] = bg[2]

    near_f = float(near)

    # Lines first (they sit behind the points).
    if m:
        ends = lines.reshape(m * 2, 3)
        lu = np.empty(m * 2, dtype=np.float32)
        lv = np.empty(m * 2, dtype=np.float32)
        lz = np.empty(m * 2, dtype=np.float32)
        lok = np.empty(m * 2, dtype=np.bool_)
        _project(ends, R, t, K, near_f, lu, lv, lz, lok)
        for s in range(m):
            a = 2 * s
            if lok[a] and lok[a + 1]:
                _wu_line(acc, float(lu[a]), float(lv[a]),
                         float(lu[a + 1]), float(lv[a + 1]),
                         float(lrgb[s, 0]), float(lrgb[s, 1]),
                         float(lrgb[s, 2]), width, height)

    # Points, far to near (stable: depth ties keep input order).
    if n:
        u = np.empty(n, dtype=np.float32)
        v = np.empty(n, dtype=np.float32)
        z = np.empty(n, dtype=np.float32)
        ok = np.empty(n, dtype=np.bool_)
        _project(pts, R, t, K, near_f, u, v, z, ok)
        idx = np.nonzero(ok)[0]
        if idx.size:
            order = idx[np.argsort(-z[idx], kind="stable")]
            rc = max(float(point_radius_px), 0.5)
            rg = rc * float(glow_radius_factor) if glow_strength > 0 else rc
            _splat(acc, u, v, rgb, order.astype(np.int64),
                   float(rc), float(rg), float(glow_strength),
                   width, height)

    return np.clip(np.rint(acc), 0, 255).astype(np.uint8)


def render_sequence(spec_json_path, out_dir):
    """Render a batch of frames described by a JSON spec file.

    The spec is a JSON object with optional global render parameters
    (``width``, ``height``, ``point_radius_px``, ``glow_strength``,
    ``glow_radius_factor``, ``background_rgb``, ``near`` — same meaning
    and defaults as :func:`render_frame`) and a required ``frames``
    list. Each frame is an object:

    - ``png``: output file name, e.g. ``"frame_000.png"``
    - ``points_npy``: path to a .npy float (N, 3) point array
    - ``colors_npy``: optional path to a .npy (N, 3) colour array;
      default colour (205, 215, 230) when absent
    - ``lines_npy``: optional path to a .npy float (M, 2, 3) array
    - ``lines_rgb``: optional [r, g, b] single colour for all segments
    - ``lines_rgb_npy``: optional path to a .npy (M, 3) colour array
    - ``camera``: {"K": [[...]], "R": [[...]], "t": [...]}
    - ``point_count``: optional int — draw only the first K points
      (progressive reveal)
    - ``point_indices``: optional list of ints — draw only these
      points (takes precedence over ``point_count``)
    - any global render parameter may be overridden per frame.

    Relative paths resolve against the spec file's directory. Returns
    the list of written PNG paths, in frame order.
    """
    from PIL import Image  # lazy: PNG writing only

    spec_path = Path(spec_json_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    base = spec_path.resolve().parent
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    param_keys = ("width", "height", "point_radius_px", "glow_strength",
                  "glow_radius_factor", "background_rgb", "near")
    globals_ = {k: spec[k] for k in param_keys if k in spec}

    written = []
    for frame in spec["frames"]:
        params = dict(globals_)
        for k in param_keys:
            if k in frame:
                params[k] = frame[k]
        xyz = np.load(base / frame["points_npy"]).astype(np.float32)
        if "colors_npy" in frame:
            rgb = np.load(base / frame["colors_npy"])
        else:
            rgb = np.full((xyz.shape[0], 3), _DEFAULT_POINT_RGB,
                          dtype=np.uint8)
        if "point_indices" in frame:
            sel = np.asarray(frame["point_indices"], dtype=np.int64)
            xyz, rgb = xyz[sel], rgb[sel]
        elif "point_count" in frame:
            k = min(int(frame["point_count"]), xyz.shape[0])
            xyz, rgb = xyz[:k], rgb[:k]

        lines = None
        lines_rgb = None
        if "lines_npy" in frame:
            lines = np.load(base / frame["lines_npy"]).astype(np.float32)
            if "lines_rgb_npy" in frame:
                lines_rgb = np.load(base / frame["lines_rgb_npy"])
            elif "lines_rgb" in frame:
                lines_rgb = np.tile(
                    np.asarray(frame["lines_rgb"], dtype=np.uint8),
                    (lines.shape[0], 1))

        img = render_frame(xyz, rgb, lines, lines_rgb,
                           camera=frame["camera"], **params)
        path = out / frame["png"]
        Image.fromarray(img).save(path)
        written.append(path)
    return written
