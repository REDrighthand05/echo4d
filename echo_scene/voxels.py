"""FROZEN FEATURE (user directive, 2026-10-31): solid voxel model.

The user declared this a foundational feature of the product: it must
stay available in the software and its behaviour must not be altered by
later work. The contract locked by tests/test_voxel_frozen.py:
every input point becomes exactly one cube, centred on the point, with
the point's own colour. Do not change this module's behaviour without
the user's explicit instruction.

Solid voxel model from the fused point cloud.

Every fused point becomes one solid cube centred on the point itself,
coloured with the point's own measured colour, sized at the same scale
as the viewer's largest point size. The result is a real model file
with actual volume -- the "pixel / low-poly" look of the viewer with
the point size turned up, except solid: it can be opened in any 3D
tool, printed, or dropped into a game engine.

Every measured point survives: cubes are centred on their points, not
snapped to a shared grid, so no point is dropped, merged, moved or
recoloured. (A snapped grid was tried first and silently swallowed a
third of the points into shared cells -- rejected for exactly that
reason.) Cubes may touch or slightly overlap where the cloud is dense,
exactly like the enlarged points in the viewer do.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# The 6 faces of a unit cube: (direction, 4 corner offsets wound CCW
# seen from outside). Windings verified against the normals in tests.
_FACES = (
    ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
    ((-1, 0, 0), ((0, 0, 1), (0, 1, 1), (0, 1, 0), (0, 0, 0))),
    ((0, 1, 0), ((0, 1, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0))),
    ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
    ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    ((0, 0, -1), ((0, 1, 0), (1, 1, 0), (1, 0, 0), (0, 0, 0))),
)
# The 8 distinct corners of a unit cube, and the 12 triangles that the
# 6 quad faces split into (corner indices into _CORNERS).
_CORNERS = np.array(
    [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
     (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], dtype=np.float64)
_FACE_CORNERS = (
    (1, 2, 6, 5),  # +X
    (4, 7, 3, 0),  # -X
    (7, 6, 2, 3),  # +Y
    (0, 1, 5, 4),  # -Y
    (4, 5, 6, 7),  # +Z
    (3, 2, 1, 0),  # -Z
)


def _tri_template() -> np.ndarray:
    tris = []
    for a, b, c, d in _FACE_CORNERS:
        tris.append((a, b, c))
        tris.append((a, c, d))
    return np.array(tris, dtype=np.int64)


def load_points_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.fromfile(path, dtype=np.uint8)
    n = len(raw) // 15
    raw = raw[: n * 15].reshape(n, 15)
    pts = raw[:, :12].copy().view("<f4").reshape(n, 3).astype(np.float64)
    cols = raw[:, 12:15].astype(np.uint8)
    return pts, cols


def build_voxel_mesh(points: np.ndarray, colors: np.ndarray):
    """Points -> (vertices float32, vertex colours uint8, triangles int32).

    Cube size = cloud extent / 220: the scale of the viewer's largest
    point size, and the pitch the fuse stage reduced the cloud on.
    """
    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    size = max(extent / 220.0, 1e-9)
    n = len(points)
    offsets = (_CORNERS - 0.5) * size
    verts = (points[:, None, :] + offsets[None, :, :]).reshape(n * 8, 3)
    vcols = np.repeat(colors, 8, axis=0)
    # (v0.6.63) Triangles built directly in int32: every index is
    # below 8*n, and 8*n-1 < 2**31 for any cloud that fits in memory,
    # so the int32 addition holds exactly the same integers as the old
    # int64 addition + cast -- and the int64 temporary (772 MB at
    # 638 frames) plus the cast pass disappear. The n guard keeps the
    # old int64 path for clouds past the int32 range, so the two forms
    # agree for every n, not just feasible ones.
    if n <= 268435455:  # (2**31 - 1 - 7) // 8
        tris = (_tri_template().astype(np.int32)[None, :, :]
                + (np.arange(n, dtype=np.int32) * 8)[:, None, None]).reshape(-1, 3)
    else:
        tris = (_tri_template()[None, :, :] + (np.arange(n) * 8)[:, None, None]).reshape(-1, 3)
        tris = tris.astype(np.int32)
    return verts.astype(np.float32), vcols, tris, {"size": size, "cubes": int(n)}


def write_ply(path: Path, verts: np.ndarray, cols: np.ndarray, tris: np.ndarray) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(verts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {len(tris)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    ).encode()
    # (v0.6.63) Same bytes as before, written in row chunks: the old
    # form built two whole-file temporaries (vbuf 151 MB + fbuf 197 MB
    # at 120 frames, 322 + 418 MB at 638) before writing anything.
    # Chunking by rows is byte-identical for ANY verts/tris length --
    # the file is one sequential byte stream either way -- so no
    # multiple-of-8/12 assumption is needed.
    chunk = 262144 * 8  # vertex rows per chunk; faces chunked by 12 rows below
    with open(path, "wb") as fh:
        fh.write(header)
        for s in range(0, len(verts), chunk):
            v = verts[s:s + chunk]
            c = cols[s:s + chunk]
            vbuf = np.empty((len(v), 15), dtype=np.uint8)
            vbuf[:, :12] = v.view(np.uint8).reshape(len(v), 12)
            vbuf[:, 12:15] = c
            fh.write(vbuf.tobytes())
        fchunk = 262144 * 12
        for s in range(0, len(tris), fchunk):
            t = tris[s:s + fchunk]
            fbuf = np.empty((len(t), 13), dtype=np.uint8)
            fbuf[:, 0] = 3
            fbuf[:, 1:13] = t.view(np.uint8).reshape(len(t), 12)
            fh.write(fbuf.tobytes())


def stage_voxel_model(job_dir: Path) -> dict:
    points_file = job_dir / "points.bin"
    if not points_file.exists():
        return {"stage": "voxel", "ok": False, "reason": "points.bin missing"}
    points, colors = load_points_bin(points_file)
    if len(points) == 0:
        return {"stage": "voxel", "ok": False, "reason": "empty point cloud"}
    verts, vcols, tris, meta = build_voxel_mesh(points, colors)
    ply_path = job_dir / "voxel_model.ply"
    write_ply(ply_path, verts, vcols, tris)
    glb_path = None
    try:
        # (v0.6.67) The voxel GLB is written with the in-repo
        # minimal writer -- the same one the export button
        # already uses for voxel_color.glb -- instead of
        # open3d: same verts/colours/triangles, as POSITION
        # float32 + COLOR_0 uint8 normalised + uint32 indices
        # in a real BIN chunk. Measured on the real jobs: 120
        # frames 12.2/13.8 s -> 2.2 s warm, peak 5.08 GB ->
        # 1.19 GB, file 725 MB -> 332 MB; 638 frames the
        # open3d branch never finished here (OOM-killed,
        # ~10 GB needed) while this writer finishes in ~7-10 s
        # at 2.45 GB peak. The open3d file also carried
        # computed smooth normals; this file carries none --
        # same as the PLY just above, which never had normals.
        # Lazy import: voxel_export imports this module at its
        # top level, so a top-level import here would be
        # circular; loaded as echo_scene.voxels it resolves.
        from .voxel_export import write_glb

        candidate = job_dir / "voxel_model.glb"
        write_glb(candidate, verts, vcols, tris)
        glb_path = str(candidate)
    except Exception:
        glb_path = None
    info = {
        "stage": "voxel",
        "ok": True,
        "cubes": meta["cubes"],
        "verts": int(len(verts)),
        "tris": int(len(tris)),
        "voxel_model_ply": str(ply_path),
        "voxel_model_glb": glb_path,
    }
    (job_dir / "voxel_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return info
