"""Coloured grid-voxel export (added 2026-11-03, user directive).

A second solidification route, next to the FROZEN per-point voxel
model in echo_scene/voxels.py (which stays untouched and
behaviourally locked): here the fused cloud is snapped to a regular
grid, every occupied cell becomes ONE cube, and the cube's colour is
the mean of the colours of the points inside the cell. The result is
the pixel-style solid scene as a *universal* model file -- the same
data as a vertex-coloured GLB and as a PLY -- ready to be edited or
dropped into other tools (game engines included).

Scale handling: a job's scale stage (echo_scene/scale.py) REWRITES
the job data in place when it applies a scale, so an applied scale
must never be multiplied in twice. This module therefore exports in
metres exactly when scale.json says the scene already is in metres
(or carries an unapplied factor, which is then applied here), and
otherwise exports the raw scene units with the unit explicitly
recorded as undefined in the info record and the GLB extras.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

# Cube corner/face templates are shared with the frozen voxel module
# (read-only import: this module never alters that one's behaviour).
from .voxels import _CORNERS, _tri_template, load_points_bin, write_ply

GLB_NAME = "voxel_color.glb"
PLY_NAME = "voxel_color.ply"
INFO_NAME = "voxel_color_info.json"


def voxelize(points: np.ndarray, colors: np.ndarray, voxel_size: float):
    """Snap points to a grid; one entry per occupied cell.

    Returns (centers float64 (M,3), mean_colors uint8 (M,3), meta).
    A cell's centre is the cell centre; its colour is the rounded mean
    of its points' colours. Empty input yields empty outputs.
    """
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if len(points) == 0:
        return (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8),
                {"voxels": 0, "voxel_size": float(voxel_size), "points_in": 0})
    keys = np.floor(points / voxel_size).astype(np.int64)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    counts = np.bincount(inverse, minlength=len(uniq)).astype(np.float64)
    sums = np.zeros((len(uniq), 3), dtype=np.float64)
    np.add.at(sums, inverse, colors.astype(np.float64))
    mean = np.rint(sums / counts[:, None])
    mean_colors = np.clip(mean, 0, 255).astype(np.uint8)
    centers = (uniq.astype(np.float64) + 0.5) * voxel_size
    meta = {"voxels": int(len(uniq)), "voxel_size": float(voxel_size),
            "points_in": int(len(points))}
    return centers, mean_colors, meta


def build_cube_mesh(centers: np.ndarray, colors: np.ndarray, size: float):
    """Occupied cells -> (vertices float32, vertex colours uint8,
    triangles int32): one cube of edge `size` per centre."""
    n = len(centers)
    offsets = (_CORNERS - 0.5) * size
    verts = (centers[:, None, :] + offsets[None, :, :]).reshape(n * 8, 3)
    vcols = np.repeat(colors, 8, axis=0)
    tris = (_tri_template()[None, :, :]
            + (np.arange(n) * 8)[:, None, None]).reshape(-1, 3)
    return verts.astype(np.float32), vcols, tris.astype(np.int32)


def _pad4(data: bytes, fill: bytes = b"\x00") -> bytes:
    return data + fill * ((-len(data)) % 4)


def write_glb(path: Path, verts: np.ndarray, cols: np.ndarray,
              tris: np.ndarray, extras: dict | None = None) -> None:
    """Minimal self-contained GLB (glTF 2.0 binary) writer.

    One mesh, one primitive: POSITION (float32), COLOR_0 (uint8,
    normalised) and uint32 indices. No external dependency, so the
    export works in the frozen app exactly as in the source tree.
    """
    idx = np.ascontiguousarray(tris, dtype=np.uint32)
    pos = np.ascontiguousarray(verts, dtype=np.float32)
    col = np.ascontiguousarray(cols, dtype=np.uint8)
    chunks = []
    views = []
    offset = 0
    for blob, target in ((idx.tobytes(), 34963), (pos.tobytes(), 34962),
                         (col.tobytes(), 34962)):
        blob = _pad4(blob)
        views.append({"buffer": 0, "byteOffset": offset,
                      "byteLength": len(blob), "target": target})
        chunks.append(blob)
        offset += len(blob)
    bin_chunk = b"".join(chunks)
    gltf = {
        "asset": {"version": "2.0", "generator": "EchoScene voxel_export",
                  "extras": extras or {}},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "voxel_color"}],
        # No material on purpose: with one present, common readers
        # (trimesh) file COLOR_0 away as a side attribute instead of
        # the mesh's vertex colours. Without it the default material
        # is white and the vertex colours ARE the model's colours.
        "meshes": [{"name": "voxel_color", "primitives": [{
            "attributes": {"POSITION": 1, "COLOR_0": 2},
            "indices": 0, "mode": 4}]}],
        "buffers": [{"byteLength": len(bin_chunk)}],
        "bufferViews": views,
        "accessors": [
            {"bufferView": 0, "componentType": 5125, "count": int(idx.size),
             "type": "SCALAR"},
            {"bufferView": 1, "componentType": 5126, "count": int(len(pos)),
             "type": "VEC3",
             "min": pos.min(axis=0).tolist() if len(pos) else [0, 0, 0],
             "max": pos.max(axis=0).tolist() if len(pos) else [0, 0, 0]},
            {"bufferView": 2, "componentType": 5121, "count": int(len(col)),
             "type": "VEC3", "normalized": True},
        ],
    }
    json_chunk = _pad4(json.dumps(gltf).encode("utf-8"), b" ")
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))  # glTF, v2
        fh.write(struct.pack("<II", len(json_chunk), 0x4E4F534A))  # JSON
        fh.write(json_chunk)
        fh.write(struct.pack("<II", len(bin_chunk), 0x004E4942))  # BIN
        fh.write(bin_chunk)


def scale_for_job(job_dir: Path):
    """Decide the export scale for a job.

    Returns (factor, units, record): multiply exported coordinates by
    `factor`; `units` is "meters" or "undefined". A scale.json with
    status "set" and applied=true means the scale stage already
    rewrote the job data in place -- factor stays 1.0 so the scale is
    never applied twice. The markers route likewise leaves the scene
    already in metres (its record says so). Only an unapplied estimate
    is multiplied in here.
    """
    rec = None
    try:
        rec = json.loads((job_dir / "scale.json").read_text())
    except (OSError, ValueError):
        rec = None
    if isinstance(rec, dict) and rec.get("status") == "set" \
            and rec.get("scale_to_meters"):
        if rec.get("applied"):
            return 1.0, "meters", rec
        if "already in metres" in str(rec.get("reason") or ""):
            return 1.0, "meters", rec
        return float(rec["scale_to_meters"]), "meters", rec
    return 1.0, "undefined", rec


def export_colored_voxels(job_dir: Path, out_dir: Path | None = None,
                          voxel_size: float | None = None,
                          formats=("glb", "ply")) -> dict:
    """Build the coloured grid-voxel model of a job's fused cloud.

    Reads job_dir/points.bin (+ scale.json), writes voxel_color.glb
    and/or voxel_color.ply plus voxel_color_info.json into out_dir
    (default: the job dir itself). Never touches the frozen model's
    voxel_model.* files.
    """
    job_dir = Path(job_dir)
    out_dir = Path(out_dir) if out_dir is not None else job_dir
    points_file = job_dir / "points.bin"
    if not points_file.exists():
        return {"stage": "voxel-export", "ok": False,
                "reason": "points.bin missing (run the job first)"}
    points, colors = load_points_bin(points_file)
    if len(points) == 0:
        return {"stage": "voxel-export", "ok": False,
                "reason": "empty point cloud"}
    factor, units, _rec = scale_for_job(job_dir)
    if factor != 1.0:
        points = points * factor
    if voxel_size is None:
        extent = float(np.linalg.norm(points.max(axis=0)
                                      - points.min(axis=0)))
        voxel_size = max(extent / 220.0, 1e-9)
    centers, mean_colors, meta = voxelize(points, colors, voxel_size)
    verts, vcols, tris = build_cube_mesh(centers, mean_colors,
                                         meta["voxel_size"])
    out_dir.mkdir(parents=True, exist_ok=True)
    extras = {"units": units, "voxel_size": meta["voxel_size"],
              "voxels": meta["voxels"], "scale_factor_applied": factor}
    files = {}
    if "glb" in formats:
        glb_path = out_dir / GLB_NAME
        write_glb(glb_path, verts, vcols, tris, extras=extras)
        files["glb"] = str(glb_path)
    if "ply" in formats:
        ply_path = out_dir / PLY_NAME
        write_ply(ply_path, verts, vcols, tris)
        files["ply"] = str(ply_path)
    info = {
        "stage": "voxel-export",
        "ok": True,
        "voxels": meta["voxels"],
        "points_in": meta["points_in"],
        "voxel_size": meta["voxel_size"],
        "units": units,
        "scale_factor_applied": factor,
        "files": files,
    }
    if units == "undefined":
        info["note"] = ("unit undefined: no scale estimate for this job, "
                        "coordinates are raw scene units")
    (out_dir / INFO_NAME).write_text(
        json.dumps(info, ensure_ascii=False, indent=2))
    info["info_json"] = str(out_dir / INFO_NAME)
    return info
