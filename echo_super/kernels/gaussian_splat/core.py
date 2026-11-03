"""3DGS Gaussian splat -> coloured point cloud conversion (Muse).

The conversion contract (what "a splat becomes points" means here):

  position  = the Gaussian centre (x, y, z), untouched
  colour    = the DC spherical-harmonic term only, mapped through the
              standard 3DGS relation  rgb = 0.5 + C0 * f_dc  with
              C0 = 1 / (2*sqrt(pi)) = 0.28209479177387814, clipped to
              [0, 1] and rounded to uint8. Higher SH bands are
              view-dependent; a point cloud has one colour per point,
              so the DC term is the honest view-independent colour.
  opacity   = alpha = sigmoid(opacity); points with alpha below the
              caller's min_alpha are dropped (default 0.0 keeps every
              Gaussian — every point survives unless asked otherwise).

Output arrays are exactly the solidification input contract of
echo_scene.voxels.build_voxel_mesh: points (N,3) float64, colours
(N,3) uint8 — and the {"points", "colors"} dict convention the
poisson adapter consumes.

Reading: binary little-endian files whose vertex element comes first
with all-scalar properties (the layout every 3DGS trainer writes) go
through one vectorized memmap read. Anything else (ascii, reordered
elements, trailing structure) delegates to ply_reader.read_ply_vertices,
the DS Flash reader verified in tests/test_gaussian_adapter.py — the
test suite also asserts both paths agree value-for-value.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .ply_reader import read_ply_vertices

SH_C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))

_PLY_DTYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}

GAUSSIAN_PROPS_MIN = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2")


def _header_vertex_layout(path):
    """Scan the PLY header; return (fmt, vertex_count, [(name, code)],
    data_offset) when the fast path applies, else None.

    Fast path conditions: binary_little_endian, the vertex element is
    the first element, and all its properties are scalar.
    """
    vertex = None
    collecting = False
    fmt = None
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError("malformed header: not a ply file")
        while True:
            line = fh.readline()
            if not line:
                raise ValueError("missing end_header")
            text = line.decode("ascii").strip()
            parts = text.split()
            if not parts or parts[0] in ("comment", "obj_info"):
                continue
            key = parts[0].lower()
            if key == "format":
                fmt = parts[1].lower()
            elif key == "element":
                if vertex is None and parts[1] == "vertex":
                    vertex = {"count": int(parts[2]), "props": []}
                    collecting = True
                elif vertex is None:
                    return None  # vertex is not the first element
                else:
                    collecting = False  # later elements: scan on, ignore
            elif key == "property" and collecting:
                if len(parts) != 3:
                    return None  # list property on the vertex element
                code = _PLY_DTYPES.get(parts[1].lower())
                if code is None:
                    return None
                vertex["props"].append((parts[2], code))
            elif key == "end_header":
                offset = fh.tell()
                break
    if fmt != "binary_little_endian" or vertex is None or not vertex["props"]:
        return None
    return fmt, vertex["count"], vertex["props"], offset


def read_gaussian_props(path) -> dict[str, np.ndarray]:
    """All vertex properties of a Gaussian .ply, as arrays.

    Uses the vectorized fast path when the layout allows it and the
    verified general reader otherwise; results are identical.
    """
    layout = _header_vertex_layout(path)
    if layout is None:
        return read_ply_vertices(path)
    _fmt, count, props, offset = layout
    dtype = np.dtype([(name, "<" + code) for name, code in props])
    table = np.memmap(path, dtype=dtype, mode="r", offset=offset,
                      shape=(count,))
    return {name: np.array(table[name]) for name, _ in props}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def gaussians_to_points(props: dict[str, np.ndarray],
                        min_alpha: float = 0.0):
    """Gaussian property dict -> (points, colors, kept_indices).

    points : (N,3) float64 Gaussian centres, input order preserved
    colors : (N,3) uint8 from the SH DC term (see module docstring)
    kept   : indices into the original vertex order that survived the
             opacity filter (all of them when min_alpha <= 0 or the
             file carries no opacity property)
    """
    missing = [k for k in GAUSSIAN_PROPS_MIN if k not in props]
    if missing:
        raise ValueError(
            f"not a gaussian splat vertex table; missing {missing}")
    points = np.stack([props["x"], props["y"], props["z"]],
                      axis=1).astype(np.float64)
    sh = np.stack([props["f_dc_0"], props["f_dc_1"], props["f_dc_2"]],
                  axis=1).astype(np.float64)
    rgb = np.clip(0.5 + SH_C0 * sh, 0.0, 1.0)
    colors = np.rint(rgb * 255.0).astype(np.uint8)
    if "opacity" in props and min_alpha > 0.0:
        keep = _sigmoid(np.asarray(props["opacity"])) >= min_alpha
        idx = np.nonzero(keep)[0]
        return points[idx], colors[idx], idx
    return points, colors, np.arange(len(points))


def load_gaussian_ply(path, min_alpha: float = 0.0) -> dict:
    """Gaussian .ply -> solidification-ready point cloud + stats."""
    props = read_gaussian_props(path)
    points, colors, kept = gaussians_to_points(props, min_alpha=min_alpha)
    return {
        "points": points,
        "colors": colors,
        "n_vertices": int(len(props["x"])),
        "n_kept": int(len(points)),
    }
