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


# The 6 faces of a unit cube, in the frozen _tri_template's face
# order (+X, -X, +Y, -Y, +Z, -Z), with the neighbour direction
# each face looks at. build_cube_mesh needs only the direction;
# the corner layout stays the frozen template's. The order is
# pinned by tests/test_v48.py (face normal == direction).
_FACE_DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
              (0, -1, 0), (0, 0, 1), (0, 0, -1))


def _pack_cell_codes(keys: np.ndarray):
    """Pack (n, 3) int64 cell coords into one int64 per cell.

    code = ((x - xmin) * ny + (y - ymin)) * nz + (z - zmin): a
    bijection from the span box to [0, nx*ny*nz), so a neighbour
    cell's code is the cell's code plus a fixed delta per
    direction (+-ny*nz, +-nz, +-1). The code formula alone is
    only injective INSIDE the box -- a neighbour outside the box
    can alias an occupied cell's code -- so the caller must mask
    out cells on that side of the box before testing. Returns
    (codes, ordered_codes, deltas) or None when the span product
    does not fit in int64 (absurdly sparse grids only; the caller
    then falls back to the structured path)."""
    mn = keys.min(axis=0)
    span = keys.max(axis=0) - mn + 1
    total = int(span[0]) * int(span[1]) * int(span[2])
    if total >= 2 ** 62:
        return None
    s1, s2 = int(span[1]), int(span[2])
    codes = ((keys[:, 0] - mn[0]) * s1
             + (keys[:, 1] - mn[1])) * s2 + (keys[:, 2] - mn[2])
    codes = np.ascontiguousarray(codes, dtype=np.int64)
    order = np.argsort(codes)
    deltas = (s1 * s2, -s1 * s2, s2, -s2, 1, -1)
    return codes, codes[order], deltas


def _codes_hit(ordered: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Exact membership test of vals in the sorted codes array."""
    pos = np.searchsorted(ordered, vals)
    hit = np.zeros(len(vals), dtype=bool)
    if len(ordered):
        pos_c = np.clip(pos, 0, len(ordered) - 1)
        hit = (pos < len(ordered)) & (ordered[pos_c] == vals)
    return hit


def _neighbour_occupied(keys: np.ndarray, shift) -> np.ndarray:
    """keys (n, 3) int64 cell coords; shift (dx, dy, dz). Returns a
    bool array: True where the shifted cell is also occupied.

    Structured-view fallback for span products too large to pack
    (see _pack_cell_codes). Sorting and searching use the same
    field-wise order, so the test is exact."""
    rec = np.dtype([("x", "<i8"), ("y", "<i8"), ("z", "<i8")])
    flat = np.ascontiguousarray(keys).view(rec).reshape(-1)
    order = np.argsort(flat)
    ordered = flat[order]
    nbr = np.ascontiguousarray(
        keys + np.asarray(shift, dtype=np.int64)).view(rec).reshape(-1)
    pos = np.searchsorted(ordered, nbr)
    hit = np.zeros(len(keys), dtype=bool)
    if len(ordered):
        pos_c = np.clip(pos, 0, len(ordered) - 1)
        hit = (pos < len(ordered)) & (ordered[pos_c] == nbr)
    return hit


def build_cube_mesh(centers: np.ndarray, colors: np.ndarray, size: float):
    """Occupied cells -> (vertices float32, vertex colours uint8,
    triangles int32): one cube of edge `size` per centre.

    Only faces with no occupied neighbour are emitted (v0.6.48):
    a face between two adjacent cubes can never be seen, so
    emitting it only bloats the GLB/PLY. On the real 30 s cloud
    (283,180 voxels) this drops triangles 3,398,160 -> 776,912
    and the GLB binary chunk 74.8 -> 25.2 MB, with the visible
    surface unchanged. Vertices are remapped to the corners
    actually used, so fully interior cubes vanish from the mesh
    entirely. Centres that are not on the grid (within 1e-4 of a
    cell) fall back to emitting all faces, exactly as before."""
    n = len(centers)
    offsets = (_CORNERS - 0.5) * size
    verts = (centers[:, None, :] + offsets[None, :, :]).reshape(n * 8, 3)
    vcols = np.repeat(colors, 8, axis=0)
    template = _tri_template()
    tris_all = (template[None, :, :]
                + (np.arange(n) * 8)[:, None, None]).reshape(-1, 3)
    if n == 0:
        return verts.astype(np.float32), vcols, tris_all.astype(np.int32)
    # Cell coords of each centre; off-grid input keeps the old
    # emit-everything behaviour (never silently change geometry).
    keys_f = np.asarray(centers, dtype=np.float64) / size - 0.5
    keys = np.rint(keys_f).astype(np.int64)
    if float(np.max(np.abs(keys_f - keys))) > 1e-4:
        return verts.astype(np.float32), vcols, tris_all.astype(np.int32)
    keep_face = np.ones((n, 6), dtype=bool)
    packed = _pack_cell_codes(keys)
    if packed is not None:
        codes, ordered, deltas = packed
        # A neighbour can only be occupied when the cell is not
        # on that side of the span box; outside the box the code
        # delta can alias an occupied cell's code (the code
        # formula is only injective inside the box).
        kmax = keys.max(axis=0)
        kmin = keys.min(axis=0)
        in_bounds = (
            keys[:, 0] < kmax[0], keys[:, 0] > kmin[0],
            keys[:, 1] < kmax[1], keys[:, 1] > kmin[1],
            keys[:, 2] < kmax[2], keys[:, 2] > kmin[2],
        )
        for f, delta in enumerate(deltas):
            hit = _codes_hit(ordered, codes + delta) & in_bounds[f]
            keep_face[:, f] = ~hit
    else:
        for f, shift in enumerate(_FACE_DIRS):
            keep_face[:, f] = ~_neighbour_occupied(keys, shift)
    # Triangle t belongs to face t // 2 (two triangles per face).
    keep_tri = np.repeat(keep_face, 2, axis=1)
    tris = tris_all.reshape(n, 12, 3)[keep_tri]
    # Remap to the corners actually used: a global vertex index is
    # cube * 8 + corner, so the used set comes straight off tris.
    used = np.unique(tris)
    remap = np.full(n * 8, -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return (verts[used].astype(np.float32), vcols[used],
            remap[tris].astype(np.int32))


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
    # (v0.6.68) Stream the BIN chunk instead of joining it: the
    # three padded lengths are pure arithmetic on the array
    # sizes -- (nbytes + 3) & ~3 is exactly what _pad4 produces
    # -- so the JSON header (offsets, byteLength, total) is
    # known before any blob exists, and each blob is written
    # and dropped in turn. The old form held the three blobs
    # plus a second whole-file join (about 450 MB at 638
    # frames) at once. Same bytes on disk, same JSON; measured
    # on the real jobs in exp/glb_stream68.py.
    def _plen(nbytes):
        return (nbytes + 3) & ~3
    lens = (_plen(idx.nbytes), _plen(pos.nbytes), _plen(col.nbytes))
    bin_len = lens[0] + lens[1] + lens[2]
    views = []
    offset = 0
    for length, target in zip(lens, (34963, 34962, 34962)):
        views.append({"buffer": 0, "byteOffset": offset,
                      "byteLength": length, "target": target})
        offset += length
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
        "buffers": [{"byteLength": bin_len}],
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
    total = 12 + 8 + len(json_chunk) + 8 + bin_len
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))  # glTF, v2
        fh.write(struct.pack("<II", len(json_chunk), 0x4E4F534A))  # JSON
        fh.write(json_chunk)
        fh.write(struct.pack("<II", bin_len, 0x004E4942))  # BIN
        for arr in (idx, pos, col):
            fh.write(_pad4(arr.tobytes()))


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
              "voxels": meta["voxels"], "triangles": int(len(tris)),
              "scale_factor_applied": factor}
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
        "triangles": int(len(tris)),
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

# ----------------------------------------------------------------------
# TEXTURED MESH EXPORT (v0.6.25)
#
# A real photo-textured surface mesh -- NOT the vertex-coloured voxel
# cubes. The chain is built only from things the job already keeps:
#
#   points.bin        the fused, corroborated cloud (fuse stage)
#   poses.json        one COLMAP pose + camera per registered frame
#   frames/*.jpg      the extracted frames themselves
#   depth/*.npy       per-frame relative inverse depth maps
#
# Steps, each honest about what it is:
#   1. surface: Screened Poisson reconstruction over the cloud
#      (open3d, shipped in the frozen stack; Poisson-class, the
#      principled global reconstruction the project standard
#      allows -- no voting, no TSDF). Density trim drops the
#      low-confidence hallucinated caps Poisson grows over holes.
#   2. view assignment: every mesh vertex is projected into every
#      registered view; a view only counts when the vertex is
#      in-frame, front-facing, AND passes a depth-map occlusion
#      test. The depth maps are relative inverse depth, so each
#      frame gets its own robust affine fit (rel ~= a*(1/Z) + b,
#      the same fit fuse uses, fitted here on mesh vertices
#      instead of COLMAP sparse points -- no pycolmap needed).
#      Best view per vertex = most frontal and nearest. A triangle
#      takes its vertices' majority view; a triangle no view can
#      see falls back to its geometrically best view, and is
#      counted as untextured in the info record.
#   3. atlas: one tile per used view in a 4096 atlas (the view
#      JPEG resized into its tile); each triangle corner's UV is
#      its projection into its assigned view, mapped into that
#      view's tile. Texel density is atlas-limited -- a tile is
#      4096 / ceil(sqrt(views)) px, so a 100-view job gets ~409 px
#      per view. That is the v1 limit, recorded in the info file.
#   4. files: textured_mesh.glb (embedded PNG texture, the
#      universal file -- Blender opens it, Three.js loads it),
#      textured_mesh.obj + .mtl + .png (second reader path),
#      mesh.ply (the Poisson surface, vertex colours sampled
#      from each vertex's best view -- the file stage_present
#      already looks for), textured_mesh_info.json.
#
# Scale: projection happens in the job's scene (COLMAP) units,
# exactly like fuse; the exported positions are then multiplied
# by scale_for_job's factor, the same convention as the voxel
# export. The mesh.ply stays in scene units, like the pipeline's
# own files.
#
# What this deliberately does NOT do: it does not write
# mesh_vertices.bin / mesh_indices.bin -- the viewer reads those
# when present, but their binary layout is not in any file on
# this machine, and a wrong-format buffer would corrupt the
# scene page. mesh.ply carries the surface instead.
#
# Failure modes are dicts, never exceptions out of the public
# entry point: missing cloud / poses / frames / Open3D / an
# empty mesh all return {ok: False, reason: ...}.
# ----------------------------------------------------------------------

TEXTURED_GLB_NAME = "textured_mesh.glb"
TEXTURED_OBJ_NAME = "textured_mesh.obj"
TEXTURED_MTL_NAME = "textured_mesh.mtl"
TEXTURED_PNG_NAME = "textured_mesh.png"
TEXTURED_INFO_NAME = "textured_mesh_info.json"
MESH_PLY_NAME = "mesh.ply"
TEXTURE_ATLAS_SIZE = 4096
_POISSON_MAX_POINTS = 300000


def _quat_xyzw_to_matrix(q):
    """Unit quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _camera_params(camera):
    """COLMAP camera record -> (fx, fy, cx, cy, k1, k2).

    Handles the models the pipeline can produce (SINGLE camera
    mode); anything else is an honest error, not a guess.
    """
    model = str(camera.get("model", ""))
    params = [float(v) for v in camera.get("params", [])]
    if model == "SIMPLE_PINHOLE" and len(params) == 3:
        f, cx, cy = params
        return f, f, cx, cy, 0.0, 0.0
    if model == "PINHOLE" and len(params) == 4:
        fx, fy, cx, cy = params
        return fx, fy, cx, cy, 0.0, 0.0
    if model == "SIMPLE_RADIAL" and len(params) == 4:
        f, cx, cy, k1 = params
        return f, f, cx, cy, k1, 0.0
    if model == "RADIAL" and len(params) == 5:
        f, cx, cy, k1, k2 = params
        return f, f, cx, cy, k1, k2
    raise ValueError(f"unsupported camera model: {model} "
                     f"({len(params)} params)")


def _project(points, rot, trans, fx, fy, cx, cy, k1, k2, width, height):
    """Project world points into one camera.

    Returns (u, v, z, in_frame): pixel coords, camera-space depth,
    and the in-frame + in-front mask. COLMAP distortion model
    (radial, on normalised coords, before the focal multiply).
    """
    xc = points @ rot.T + trans
    z = xc[:, 2]
    valid = z > 0.05
    zs = np.where(valid, z, 1.0)
    x = xc[:, 0] / zs
    y = xc[:, 1] / zs
    r2 = x * x + y * y
    d = 1.0 + k1 * r2 + k2 * r2 * r2
    u = fx * (x * d) + cx
    v = fy * (y * d) + cy
    in_frame = (valid & (u >= 0) & (u < width)
                & (v >= 0) & (v < height))
    return u, v, z, in_frame


def _robust_affine_fit(x, y):
    """Fit y ~= a*x + b with median-residual trimming.

    Same shape as fuse's fit: one polyfit, then two rounds that
    drop points further than 3x the median residual. Returns
    (a, b) or None when there is nothing to fit on.
    """
    if len(x) < 10:
        return None
    a, b = np.polyfit(x, y, 1)
    for _ in range(2):
        resid = np.abs(y - (a * x + b))
        keep = resid <= max(3.0 * np.median(resid), 1e-9)
        if keep.sum() < max(10, len(x) // 4):
            break
        a, b = np.polyfit(x[keep], y[keep], 1)
    return float(a), float(b)


def _build_surface_mesh(points, voxel_size, poisson_depth=8,
                        log=None, camera_centers=None):
    """Poisson surface over the cloud. Returns (verts, tris, normals).

    Raises ImportError when open3d is not importable (the frozen
    stack ships it; a build without it fails honestly instead of
    silently producing a worse mesh). Raises ValueError on an
    empty or degenerate result.

    Orientation: a room cloud is a hollow shell, and Poisson's
    inside/outside is decided by the point normals. Normals are
    therefore flipped to face the nearest camera centre before
    the reconstruction, so the surface faces the cameras that
    photographed it -- otherwise the mesh comes out inside-out
    and no triangle passes the facing test at texture time.
    """
    import open3d as o3d

    def _say(msg):
        if log is not None:
            log(msg)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.asarray(points, dtype=np.float64))
    pcd = pcd.voxel_down_sample(voxel_size)
    if len(pcd.points) > _POISSON_MAX_POINTS:
        idx = np.random.default_rng(0).choice(
            len(pcd.points), _POISSON_MAX_POINTS, replace=False)
        pcd = pcd.select_by_index(idx.tolist())
    _say(f"带贴图网格：点云 {len(points)} 点，"
         f"降采样后 {len(pcd.points)} 点做曲面重建…")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2.5, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=30)
    if camera_centers is not None and len(camera_centers):
        # face the cameras: flip any normal pointing away from
        # the nearest camera centre (chunked, so memory stays flat)
        pts = np.asarray(pcd.points, dtype=np.float64)
        nrm = np.asarray(pcd.normals, dtype=np.float64)
        centres = np.asarray(camera_centers, dtype=np.float64)
        for s in range(0, len(pts), 50000):
            chunk = pts[s:s + 50000]
            d2 = ((chunk[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
            nearest = centres[np.argmin(d2, axis=1)]
            to_cam = nearest - chunk
            to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True)
            facing = np.einsum("ij,ij->i", nrm[s:s + 50000], to_cam)
            nrm[s:s + 50000] = np.where(
                (facing < 0)[:, None], -nrm[s:s + 50000],
                nrm[s:s + 50000])
        pcd.normals = o3d.utility.Vector3dVector(nrm)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(poisson_depth), scale=1.1, linear_fit=False)
    densities = np.asarray(densities, dtype=np.float64)
    if len(densities):
        keep = densities >= np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask((~keep).tolist())
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    if len(tris) == 0 or len(verts) == 0:
        raise ValueError("Poisson reconstruction produced no surface")
    _say(f"带贴图网格：曲面 {len(tris)} 个三角形，"
         f"{len(verts)} 个顶点。")
    return verts, tris, normals


def _sample_vertex_colors(verts, best_view, best_u, best_v,
                          view_frames, log=None):
    """Per-vertex RGB from each vertex's best view (nearest pixel).

    view_frames: list of (frame_name, numpy RGB array or None).
    Vertices with no view stay mid-grey -- visible in the file
    as "no photo saw this", not silently black.
    """
    colors = np.full((len(verts), 3), 128, dtype=np.uint8)
    by_view = {}
    for i, v in enumerate(best_view):
        if v >= 0:
            by_view.setdefault(int(v), []).append(i)
    for v, idxs in by_view.items():
        arr = view_frames[v][1]
        if arr is None:
            continue
        idxs = np.asarray(idxs, dtype=np.int64)
        h, w = arr.shape[0], arr.shape[1]
        px = np.clip(np.rint(best_u[idxs]).astype(np.int64), 0, w - 1)
        py = np.clip(np.rint(best_v[idxs]).astype(np.int64), 0, h - 1)
        colors[idxs] = arr[py, px]
    return colors


_DEDUP_CHUNK = 1_000_000


def _chunked_unique_packed(key):
    """(uniq, first, inv) identical to np.unique(key,
    return_index=True, return_inverse=True), but no single unique
    call ever sees more than _DEDUP_CHUNK keys.

    Measured (v0.6.73, exp/chunkuniq73.py, real 120-frame task,
    6,788,892 corners, 1,313,755 uniques): the single np.unique
    over the packed keys is the function's peak -- argsort perm +
    sorted copy + cumsum + inv, ~360 MB in a forked child vs
    ~270 MB for this form, wall time unchanged (~0.79 s both).
    Per chunk: unique (small transient), then ONE unique over the
    concatenation of the chunk unique sets (~1.5M keys at 120
    frames, not 6.8M); first is the min over chunks via
    np.minimum.at on searchsorted ranks, inv is rank[chunk_inv]
    per chunk. The chunk inverses are kept until consumed (54 MB
    at 120 frames, the same order as the final inv).
    """
    n = len(key)
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy()
    uniq_parts = []
    first_parts = []
    inv_parts = []
    for start in range(0, n, _DEDUP_CHUNK):
        stop = min(start + _DEDUP_CHUNK, n)
        uq, first_c, inv_c = np.unique(
            key[start:stop], return_index=True, return_inverse=True)
        uniq_parts.append(uq)
        first_parts.append(first_c.astype(np.int64) + start)
        inv_parts.append(inv_c)
    if len(uniq_parts) == 1:
        uniq = uniq_parts[0]
    else:
        uniq = np.unique(np.concatenate(uniq_parts))
    first = np.full(len(uniq), np.iinfo(np.int64).max, dtype=np.int64)
    inv = np.empty(n, dtype=np.int64)
    for uq, first_c, inv_c, start in zip(
            uniq_parts, first_parts, inv_parts,
            range(0, n, _DEDUP_CHUNK)):
        stop = min(start + _DEDUP_CHUNK, n)
        rank = np.searchsorted(uniq, uq)
        np.minimum.at(first, rank, first_c)
        inv[start:stop] = rank[inv_c]
        del rank
    return uniq, first, inv


def write_textured_glb(path, verts, normals, tris, uvs, png_bytes,
                       extras=None):
    """Minimal self-contained textured GLB (glTF 2.0 binary).

    One mesh, one primitive, INDEXED triangles: POSITION + NORMAL
    + TEXCOORD_0, one material with a baseColorTexture whose image
    is the atlas PNG embedded in the BIN chunk. Corners are
    deduplicated by (vertex, uv) -- a vertex shared by two
    triangles with the same UV is stored once; a seam vertex,
    textured from two different views, appears once per UV.
    Writing it flat (triangles x 3) would bloat the file ~6x.
    Mirrors write_glb's chunk/padding style.
    """
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    corner_vi = tris.reshape(-1)
    corner_uv = np.asarray(uvs, dtype=np.float32).reshape(-1, 2)
    # (v0.6.70) Dedup by one packed int64 key per corner instead
    # of a (corners, 3) int64 key table: key = vi<<40 | uq<<20 | vq,
    # where uq/vq are the same round(uv*1e6) values the old form
    # computed. The packing is order-preserving (vi, uq, vq each
    # in its own bit field, most significant first), so np.unique
    # over the packed keys returns the same unique set, first
    # indices and inverse map as np.unique(keys, axis=0) -- the
    # GLB is byte-identical -- while holding one 8-byte array per
    # corner instead of three and a 1-D sort instead of a row
    # sort. Measured in exp/dedup_ab70.py: the dedup step was the
    # function's peak (1,077 MB at 120 frames); packing drops it
    # by ~0.4 GB. Out-of-range inputs (vertex index >= 2**22,
    # quantised UV outside [0, 2**20), negative indices) fall
    # back to the old row-wise unique, so the result is identical
    # there too.
    _uq = corner_uv[:, 0].astype(np.float64)
    _uq *= 1e6
    np.rint(_uq, out=_uq)
    _vq = corner_uv[:, 1].astype(np.float64)
    _vq *= 1e6
    np.rint(_vq, out=_vq)
    if (len(corner_vi) == 0 or
            (corner_vi.min() >= 0 and corner_vi.max() < (1 << 22) and
             _uq.min() >= 0 and _uq.max() < (1 << 20) and
             _vq.min() >= 0 and _vq.max() < (1 << 20))):
        _key = ((corner_vi << 40) |
                (_uq.astype(np.int64) << 20) |
                _vq.astype(np.int64))
        del _uq, _vq
        # (v0.6.73) Chunked unique: identical (uniq, first, inv),
        # peak ~90 MB lower at 120 frames, wall time unchanged --
        # see _chunked_unique_packed above. The single np.unique
        # was the function's peak (v0.6.72 REPORT section 1).
        _uniq_key, first, inv = _chunked_unique_packed(_key)
        _uniq_vi = _uniq_key >> 40
        pos = np.ascontiguousarray(
            np.asarray(verts, dtype=np.float32)[_uniq_vi])
        nrm = np.ascontiguousarray(
            np.asarray(normals, dtype=np.float32)[_uniq_vi])
    else:
        uv_q = np.round(corner_uv.astype(np.float64) * 1e6).astype(np.int64)
        keys = np.empty((len(corner_vi), 3), dtype=np.int64)
        keys[:, 0] = corner_vi
        keys[:, 1] = uv_q[:, 0]
        keys[:, 2] = uv_q[:, 1]
        _uniq, first, inv = np.unique(keys, axis=0, return_index=True,
                                      return_inverse=True)
        pos = np.ascontiguousarray(
            np.asarray(verts, dtype=np.float32)[_uniq[:, 0]])
        nrm = np.ascontiguousarray(
            np.asarray(normals, dtype=np.float32)[_uniq[:, 0]])
    tex = np.ascontiguousarray(corner_uv[first])
    idx = inv.reshape(-1).astype(np.uint32)
    # (v0.6.71) Stream the BIN chunk instead of joining it: the
    # five padded lengths are pure arithmetic on the array
    # sizes -- (nbytes + 3) & ~3 is exactly what _pad4 produces
    # -- so the JSON header (offsets, byteLength, total) is
    # known before any blob exists, and each blob is written
    # and dropped in turn. The old form held the five blobs
    # plus a second whole-file join (about 83 MB at 120
    # frames, about 137 MB at 638) at once. Same bytes on
    # disk, same JSON; measured on the real jobs in
    # exp/stream_ab71.py.
    img = bytes(png_bytes)
    def _plen(nbytes):
        return (nbytes + 3) & ~3
    lens = (_plen(idx.nbytes), _plen(pos.nbytes), _plen(nrm.nbytes),
            _plen(tex.nbytes), _plen(len(img)))
    bin_len = sum(lens)
    views = []
    offset = 0
    for length, target in zip(lens[:4], (34963, 34962, 34962, 34962)):
        views.append({"buffer": 0, "byteOffset": offset,
                      "byteLength": length, "target": target})
        offset += length
    views.append({"buffer": 0, "byteOffset": offset,
                  "byteLength": lens[4]})
    gltf = {
        "asset": {"version": "2.0",
                  "generator": "EchoScene textured-mesh export",
                  "extras": extras or {}},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "textured_mesh"}],
        "meshes": [{"name": "textured_mesh", "primitives": [{
            "attributes": {"POSITION": 1, "NORMAL": 2, "TEXCOORD_0": 3},
            "indices": 0, "material": 0, "mode": 4}]}],
        "materials": [{"name": "textured",
                       "pbrMetallicRoughness": {
                           "baseColorTexture": {"index": 0},
                           "metallicFactor": 0.0,
                           "roughnessFactor": 1.0}}],
        "textures": [{"sampler": 0, "source": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987,
                      "wrapS": 10497, "wrapT": 10497}],
        "images": [{"bufferView": 4, "mimeType": "image/png",
                    "name": "texture"}],
        "buffers": [{"byteLength": bin_len}],
        "bufferViews": views,
        "accessors": [
            {"bufferView": 0, "componentType": 5125,
             "count": int(idx.size), "type": "SCALAR"},
            {"bufferView": 1, "componentType": 5126,
             "count": int(len(pos)), "type": "VEC3",
             "min": pos.min(axis=0).tolist() if len(pos) else [0, 0, 0],
             "max": pos.max(axis=0).tolist() if len(pos) else [0, 0, 0]},
            {"bufferView": 2, "componentType": 5126,
             "count": int(len(nrm)), "type": "VEC3"},
            {"bufferView": 3, "componentType": 5126,
             "count": int(len(tex)), "type": "VEC2"},
        ],
    }
    json_chunk = _pad4(json.dumps(gltf).encode("utf-8"), b" ")
    total = 12 + 8 + len(json_chunk) + 8 + bin_len
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(json_chunk), 0x4E4F534A))
        fh.write(json_chunk)
        fh.write(struct.pack("<II", bin_len, 0x004E4942))
        for arr in (idx, pos, nrm, tex):
            fh.write(_pad4(arr.tobytes()))
        fh.write(_pad4(img))


def write_textured_obj(path, verts, corner_normals, tris, uvs,
                       mtl_name, png_name):
    """OBJ + MTL pair sharing the atlas PNG (same UVs as the GLB).

    Vertices are shared (one `v` per mesh vertex); UVs and normals
    are per corner, so each face references (vertex / corner-uv /
    corner-normal). OBJ vt origin is the BOTTOM-left of the image,
    opposite to glTF's top-left, so v is flipped here and nowhere else.
    """
    path = Path(path)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"mtllib {mtl_name}\n")
        fh.write("o textured_mesh\n")
        for x, y, z in verts:
            fh.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for u, v in uvs:
            fh.write(f"vt {u:.6f} {1.0 - v:.6f}\n")
        for x, y, z in corner_normals:
            fh.write(f"vn {x:.6f} {y:.6f} {z:.6f}\n")
        fh.write("usemtl textured\n")
        for t, (a, b, c) in enumerate(tris):
            base = 3 * t
            fh.write(f"f {a + 1}/{base + 1}/{base + 1} "
                     f"{b + 1}/{base + 2}/{base + 2} "
                     f"{c + 1}/{base + 3}/{base + 3}\n")
    with open(path.with_name(mtl_name), "w", encoding="utf-8") as fh:
        fh.write("newmtl textured\n")
        fh.write("Ka 1.0 1.0 1.0\n")
        fh.write("Kd 1.0 1.0 1.0\n")
        fh.write("Ks 0.0 0.0 0.0\n")
        fh.write(f"map_Kd {png_name}\n")


def export_textured_mesh(job_dir, out_dir=None, atlas_size=None,
                         poisson_depth=8, log=None):
    """Build the photo-textured mesh of a finished job.

    Reads job_dir/points.bin + poses.json + frames/ + depth/
    (depth maps optional -- a view without one still counts,
    on geometry alone). Writes textured_mesh.glb / .obj / .mtl /
    .png, mesh.ply and textured_mesh_info.json into out_dir
    (default: the job dir). Never raises: every failure returns
    {ok: False, reason: ...}.
    """
    from PIL import Image

    def _say(msg):
        if log is not None:
            log(msg)

    job_dir = Path(job_dir)
    out_dir = Path(out_dir) if out_dir is not None else job_dir
    atlas_size = int(atlas_size or TEXTURE_ATLAS_SIZE)
    try:
        points_file = job_dir / "points.bin"
        if not points_file.exists():
            return {"stage": "textured-mesh", "ok": False,
                    "reason": "points.bin missing (run the job first)"}
        points, _colors = load_points_bin(points_file)
        if len(points) == 0:
            return {"stage": "textured-mesh", "ok": False,
                    "reason": "empty point cloud"}
        poses_file = job_dir / "poses.json"
        if not poses_file.exists():
            return {"stage": "textured-mesh", "ok": False,
                    "reason": "poses.json missing (run the job first)"}
        poses = json.loads(poses_file.read_text(encoding="utf-8"))
        if not isinstance(poses, dict) or not poses:
            return {"stage": "textured-mesh", "ok": False,
                    "reason": "no registered camera poses"}
        frames_dir = job_dir / "frames"
        depth_dir = job_dir / "depth"
        views = []
        for name, pose in sorted(poses.items()):
            frame = frames_dir / name
            if not frame.exists():
                continue
            try:
                fx, fy, cx, cy, k1, k2 = _camera_params(
                    pose["camera"])
            except (KeyError, ValueError) as exc:
                _say(f"带贴图网格：跳过 {name}（相机参数不明：{exc}）。")
                continue
            rot = _quat_xyzw_to_matrix(
                pose["cam_from_world_quat_xyzw"])
            trans = np.array(pose["cam_from_world_t"],
                             dtype=np.float64)
            cam = pose["camera"]
            views.append({
                "name": name, "frame": frame,
                "depth": depth_dir / (Path(name).stem + ".npy"),
                "rot": rot, "trans": trans,
                "center": -rot.T @ trans,
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "k1": k1, "k2": k2,
                "width": int(cam["width"]),
                "height": int(cam["height"]),
            })
        if not views:
            return {"stage": "textured-mesh", "ok": False,
                    "reason": "no usable camera views "
                              "(frames or camera parameters missing)"}
        # --- surface ---------------------------------------------
        extent = float(np.linalg.norm(points.max(axis=0)
                                      - points.min(axis=0)))
        voxel_size = max(extent / 220.0, 1e-9)
        cam_centres = [np.array(v["center"], dtype=np.float64)
                       for v in views]
        try:
            verts, tris, normals = _build_surface_mesh(
                points, voxel_size, poisson_depth, log=_say,
                camera_centers=cam_centres)
        except ImportError:
            return {"stage": "textured-mesh", "ok": False,
                    "reason": "Open3D is not in this build "
                              "(surface reconstruction unavailable)"}
        except Exception as exc:
            return {"stage": "textured-mesh", "ok": False,
                    "reason": f"surface reconstruction failed: {exc}"}
        n_verts = len(verts)
        # safety net: if the mesh still faces away from the
        # cameras (Poisson's inside/outside can flip on thin
        # shells), flip winding + normals outright -- a mesh
        # whose faces point away from every camera textures
        # nothing at all.
        # (v0.6.69) Nearest-camera check in 50k-vertex blocks:
        # the old form materialised (n_verts, n_views, 3)
        # float64 in one array -- 26 GB at 638 frames, and the
        # export died. Each vertex's math is unchanged, so the
        # per-vertex values are identical and the median is
        # over the same multiset; only the peak transient
        # drops (one block: ~0.8 GB at 638 views). Measured in
        # exp/lean_capture69.py + exp/flip_ab69.py.
        _cc = np.asarray(cam_centres, dtype=np.float64)
        _dots = []
        for _s in range(0, n_verts, 50000):
            _v = verts[_s:_s + 50000]
            _n = normals[_s:_s + 50000]
            _d2 = ((_v[:, None, :] - _cc[None, :, :]) ** 2).sum(-1)
            _near = _cc[np.argmin(_d2, axis=1)]
            _to = _near - _v
            _to /= np.linalg.norm(_to, axis=1, keepdims=True)
            _dots.append(np.einsum("ij,ij->i", _n, _to))
        if float(np.median(np.concatenate(_dots))) < 0:
            tris = tris[:, [0, 2, 1]]
            normals = -normals
            _say("带贴图网格：曲面朝向翻过来了（朝向相机）。")
        # --- per-vertex best view --------------------------------
        best_view = np.full(n_verts, -1, dtype=np.int64)
        best_any = np.full(n_verts, -1, dtype=np.int64)
        best_score = np.full(n_verts, -np.inf, dtype=np.float64)
        best_any_score = np.full(n_verts, -np.inf, dtype=np.float64)
        best_u = np.zeros(n_verts, dtype=np.float64)
        best_v = np.zeros(n_verts, dtype=np.float64)
        best_any_u = np.zeros(n_verts, dtype=np.float64)
        best_any_v = np.zeros(n_verts, dtype=np.float64)
        for vi, view in enumerate(views):
            u, v, z, in_frame = _project(
                verts, view["rot"], view["trans"],
                view["fx"], view["fy"], view["cx"], view["cy"],
                view["k1"], view["k2"], view["width"],
                view["height"])
            if not in_frame.any():
                continue
            # facing: normal towards the camera
            to_cam = view["center"][None, :] - verts
            dist = np.linalg.norm(to_cam, axis=1)
            dist = np.where(dist > 0, dist, 1.0)
            cosang = np.einsum("ij,ij->i", normals,
                               to_cam / dist[:, None])
            facing = in_frame & (cosang > 0)
            if not facing.any():
                continue
            score_any = np.where(facing, cosang / np.maximum(z, 1e-6),
                                 -np.inf)
            better_any = score_any > best_any_score
            best_any_score = np.where(better_any, score_any,
                                      best_any_score)
            best_any = np.where(better_any, vi, best_any)
            best_any_u = np.where(better_any, u, best_any_u)
            best_any_v = np.where(better_any, v, best_any_v)
            # depth-map occlusion test (this view's own affine fit)
            ok_depth = np.zeros(n_verts, dtype=bool)
            if view["depth"].exists():
                try:
                    depth = np.load(view["depth"])
                    dh, dw = depth.shape
                    sx = view["width"] / float(dw)
                    sy = view["height"] / float(dh)
                    cand = np.nonzero(facing)[0]
                    step = max(1, len(cand) // 20000)
                    sub = cand[::step]
                    px = np.clip((u[sub] / sx).astype(np.int64),
                                 0, dw - 1)
                    py = np.clip((v[sub] / sy).astype(np.int64),
                                 0, dh - 1)
                    rel = depth[py, px].astype(np.float64)
                    fit = _robust_affine_fit(1.0 / z[sub], rel)
                    if fit is not None and fit[0] > 0:
                        a, b = fit
                        pxa = np.clip((u / sx).astype(np.int64),
                                      0, dw - 1)
                        pya = np.clip((v / sy).astype(np.int64),
                                      0, dh - 1)
                        rel_all = depth[pya, pxa].astype(np.float64)
                        denom = rel_all - b
                        valid_denom = denom > 1e-6
                        z_map = np.where(valid_denom, a / np.where(
                            valid_denom, denom, 1.0), np.inf)
                        tol = np.maximum(0.05 * z_map, 0.03)
                        ok_depth = (facing & valid_denom
                                    & (np.abs(z - z_map) <= tol))
                    else:
                        ok_depth = facing.copy()
                except Exception as exc:
                    _say(f"带贴图网格：{view['name']} 深度图不可用"
                         f"（{exc}），按几何打分。")
                    ok_depth = facing.copy()
            else:
                ok_depth = facing.copy()
            score = np.where(ok_depth, cosang / np.maximum(z, 1e-6),
                             -np.inf)
            better = score > best_score
            best_score = np.where(better, score, best_score)
            best_view = np.where(better, vi, best_view)
            best_u = np.where(better, u, best_u)
            best_v = np.where(better, v, best_v)
        _say(f"带贴图网格：{len(views)} 个视角打分完成，"
             f"{int((best_view >= 0).sum())}/{n_verts} 个顶点"
             "通过深度遮挡检验。")
        # --- per-triangle view (majority vote of its corners) ---
        corner_view = best_view[tris]
        corner_any = best_any[tris]
        corner_score = best_score[tris]
        tri_view = np.where(corner_view >= 0, corner_view, corner_any)
        # majority: a corner wins if it agrees with another corner,
        # else the corner with the best score wins
        same01 = tri_view[:, 0] == tri_view[:, 1]
        same02 = tri_view[:, 0] == tri_view[:, 2]
        same12 = tri_view[:, 1] == tri_view[:, 2]
        pick0 = same01 | same02
        pick1 = (~pick0) & same12
        best_corner = np.argmax(
            np.where(corner_view >= 0, corner_score,
                     np.where(corner_any >= 0,
                              best_any_score[tris], -np.inf)), axis=1)
        tri_view = np.where(pick0, tri_view[:, 0],
                            np.where(pick1, tri_view[:, 1],
                                     tri_view[np.arange(len(tris)),
                                              best_corner]))
        untextured_mask = tri_view < 0
        untextured = int(untextured_mask.sum())
        tri_view = np.where(untextured_mask, 0, tri_view)
        used = sorted(set(int(v) for v in tri_view))
        # --- atlas ------------------------------------------------
        grid = int(np.ceil(np.sqrt(len(used))))
        tile = max(128, atlas_size // grid)
        tile_origin = {}
        atlas = Image.new("RGB", (atlas_size, atlas_size), (0, 0, 0))
        for slot, vi in enumerate(used):
            ox, oy = (slot % grid) * tile, (slot // grid) * tile
            tile_origin[vi] = (ox, oy)
            try:
                with Image.open(views[vi]["frame"]) as img:
                    atlas.paste(img.convert("RGB").resize(
                        (tile, tile), Image.LANCZOS), (ox, oy))
            except Exception as exc:
                _say(f"带贴图网格：{views[vi]['name']} 贴图读失败"
                     f"（{exc}），这块保持黑色。")
        # the black texel untextured triangles point at (a 4x4
        # block, so bilinear/mipmap sampling stays black too)
        atlas.paste(Image.new("RGB", (4, 4), (0, 0, 0)), (0, 0))
        # --- corner arrays (one vertex per triangle corner) -----
        # a corner's UV comes from re-projecting it into its
        # triangle's assigned view (its own best view may differ)
        uv = np.zeros((len(tris), 3, 2), dtype=np.float64)
        for vi in used:
            mask = (tri_view[:, None].repeat(3, axis=1) == vi)
            if not mask.any():
                continue
            view = views[vi]
            cu, cv, _z, _ok = _project(
                verts[tris][mask], view["rot"], view["trans"],
                view["fx"], view["fy"], view["cx"], view["cy"],
                view["k1"], view["k2"], view["width"],
                view["height"])
            ox, oy = tile_origin[vi]
            ax = ox + cu / view["width"] * tile
            ay = oy + cv / view["height"] * tile
            # glTF UV origin is the TOP-left of the image, so no
            # v-flip here (the OBJ writer flips v on its own).
            uv[mask, 0] = ax / atlas_size
            uv[mask, 1] = ay / atlas_size
        uv = np.clip(uv.reshape(-1, 2), 0.0, 1.0)
        # triangles no camera ever saw point at the black texel
        # at the atlas origin -- flat black, not a smeared photo
        # from a view that never saw them
        if untextured:
            uv[untextured_mask.repeat(3)] = 0.0
        factor, units, _rec = scale_for_job(job_dir)
        corners = verts[tris].reshape(-1, 3) * factor
        corner_normals = normals[tris].reshape(-1, 3)
        # --- mesh.ply (coloured surface, scene units) ------------
        view_arrays = []
        for vi in used:
            try:
                with Image.open(views[vi]["frame"]) as img:
                    view_arrays.append(np.asarray(
                        img.convert("RGB")))
            except Exception:
                view_arrays.append(None)
        frame_arrays = [None] * len(views)
        for slot, vi in enumerate(used):
            frame_arrays[vi] = view_arrays[slot]
        # vertex colours: best depth-passing view, else best
        # geometric view, else mid-grey (see _sample_vertex_colors)
        eff_view = np.where(best_view >= 0, best_view, best_any)
        eff_u = np.where(best_view >= 0, best_u, best_any_u)
        eff_v = np.where(best_view >= 0, best_v, best_any_v)
        vcolors = _sample_vertex_colors(
            verts, eff_view, eff_u, eff_v,
            [(views[i]["name"], frame_arrays[i])
             for i in range(len(views))])
        out_dir.mkdir(parents=True, exist_ok=True)
        write_ply(out_dir / MESH_PLY_NAME,
                  verts.astype(np.float32), vcolors,
                  tris.astype(np.int32))
        # --- atlas PNG + GLB + OBJ -------------------------------
        import io as _io
        buf = _io.BytesIO()
        atlas.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        (out_dir / TEXTURED_PNG_NAME).write_bytes(png_bytes)
        extras = {"units": units, "triangles": int(len(tris)),
                  "views_used": len(used),
                  "atlas_size": atlas_size, "tile": tile,
                  "scale_factor_applied": factor,
                  "untextured_triangles": untextured}
        write_textured_glb(out_dir / TEXTURED_GLB_NAME,
                           verts * factor, normals, tris, uv, png_bytes,
                           extras=extras)
        write_textured_obj(out_dir / TEXTURED_OBJ_NAME,
                           verts * factor, corner_normals, tris, uv,
                           TEXTURED_MTL_NAME, TEXTURED_PNG_NAME)
        info = {
            "stage": "textured-mesh",
            "ok": True,
            "triangles": int(len(tris)),
            "vertices": int(n_verts),
            "views_used": len(used),
            "views_used_names": [views[i]["name"] for i in used],
            "views_total": len(views),
            "atlas_size": atlas_size,
            "tile": tile,
            "units": units,
            "scale_factor_applied": factor,
            "untextured_triangles": untextured,
            "files": {
                "glb": str(out_dir / TEXTURED_GLB_NAME),
                "obj": str(out_dir / TEXTURED_OBJ_NAME),
                "mtl": str(out_dir / TEXTURED_MTL_NAME),
                "png": str(out_dir / TEXTURED_PNG_NAME),
                "ply": str(out_dir / MESH_PLY_NAME),
            },
            "note": ("texture is photo-sampled per triangle from "
                     "its best visible frame; atlas tiles are "
                     f"{tile}px, so texel density is atlas-limited"),
        }
        (out_dir / TEXTURED_INFO_NAME).write_text(
            json.dumps(info, ensure_ascii=False, indent=2))
        info["info_json"] = str(out_dir / TEXTURED_INFO_NAME)
        _say(f"带贴图网格好了：{len(tris)} 个三角形，"
             f"{len(used)} 个视角，{untextured} 个三角形没拍到。")
        return info
    except Exception as exc:
        return {"stage": "textured-mesh", "ok": False,
                "reason": f"{type(exc).__name__}: {exc}"}

# ---------------------------------------------------------------------------
# CAMERA ROUTE EXPORT (v0.6.26)
# ---------------------------------------------------------------------------
# The finished job's recovered camera path, as two files in the job
# folder -- the third export item (after the coloured volume and the
# textured mesh), and the one whose consumer is the NEXT tool, not
# this app: another AI, or Blender.
#
# What a finished job already has (no new reconstruction work):
#   poses.json -- one COLMAP camera pose per registered frame:
#     cam_from_world_quat_xyzw + cam_from_world_t + the shared
#     camera record (SINGLE camera mode). Raw COLMAP frame.
#   alignment.json -- when the fuse stage levelled the scene, the
#     rigid rotation that maps the COLMAP frame onto the exported
#     frame (+Y up, the same frame as points.bin / voxel_color.glb /
#     textured_mesh.glb). Absent when no rotation was applied.
#   scale.json -- status "set" + applied means poses.json is
#     already in metres (the scale stage rewrote it in place);
#     otherwise the route is in arbitrary COLMAP units and the
#     files say so. Reuses scale_for_job, the same rule as the
#     voxel export, so the route and the scene can never disagree
#     about units.
#   frame_capture_times.json -- frame name -> sensor-clock seconds,
#     only when a capture record paired. Optional.
#
# Format choice, and why:
#   camera_route.json -- the AI handoff. JSON is what another AI
#     reads directly: per-frame centre + world-from-camera
#     quaternion + forward/up, times, intrinsics, units and the
#     frame convention, all in one self-describing file. No binary
#     to parse, nothing to guess.
#   camera_route.glb -- the Blender handoff. A LINE_STRIP polyline
#     through the centres in the SAME exported frame as the scene
#     GLBs, so it overlays the scene exactly, plus one perspective
#     camera node with a translation+rotation animation (one
#     keyframe per registered frame), so Blender plays the camera
#     along the route. FBX/Alembic would be the other native
#     Blender camera formats, but no writer for either exists in
#     this codebase and hand-rolling one is not honest work -- GLB
#     is already the product's universal format and this module
#     already has a verified hand-rolled GLB writer.
#
# Deliberately NOT smoothed: the polyline joins the measured
# camera centres as they are. The replay page's smoothed figure
# path (build_replay_path) is a presentation choice for the walk,
# not data -- smoothing here would invent a path the camera never
# flew. A job needs only poses.json: points.bin is not required,
# so a job whose fuse failed can still hand its route over.
# ---------------------------------------------------------------------------
ROUTE_JSON_NAME = "camera_route.json"
ROUTE_GLB_NAME = "camera_route.glb"


def _matrix_to_quat_xyzw(R):
    """3x3 rotation matrix -> unit quaternion (x, y, z, w).

    Branch on the largest diagonal (Shepperd's method): stable
    for 180-degree rotations, where the trace form degenerates.
    """
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def _route_frames(job_dir):
    """Read poses.json (+ alignment.json + scale.json + times) and
    return (frames, meta): frames is a list of dicts with centre /
    orientation in the EXPORTED frame; meta carries units, the
    camera record, the time source and counts. Raises ValueError
    with a plain reason on any honest failure.
    """
    job_dir = Path(job_dir)
    poses_file = job_dir / "poses.json"
    if not poses_file.exists():
        raise ValueError("poses.json missing (run the job first)")
    try:
        poses = json.loads(poses_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"poses.json unreadable: {exc}")
    if not isinstance(poses, dict) or not poses:
        raise ValueError("no registered camera poses")

    # The fuse stage's export rotation (COLMAP frame -> exported
    # frame, +Y up). Column convention: a world point p becomes
    # E @ p; a world-from-camera rotation R becomes E @ R. Absent
    # means the points were exported unrotated. An unreadable
    # sidecar is a failure, not a guess: the wrong frame is worse
    # than no export.
    export_rot = None
    align_file = job_dir / "alignment.json"
    if align_file.exists():
        try:
            align = json.loads(align_file.read_text(encoding="utf-8"))
            export_rot = np.asarray(align["rotation"], dtype=np.float64)
            if export_rot.shape != (3, 3):
                raise ValueError("rotation is not 3x3")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"alignment.json unreadable: {exc}")

    factor, units, scale_rec = scale_for_job(job_dir)
    units_label = "meters" if units == "meters" else "colmap_units"

    times = {}
    times_file = job_dir / "frame_capture_times.json"
    if times_file.exists():
        try:
            raw = json.loads(times_file.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                times = {str(k): float(v) for k, v in raw.items()}
        except (OSError, ValueError):
            times = {}

    frames = []
    skipped = 0
    camera = None
    for name in sorted(poses):
        pose = poses[name]
        try:
            rot = _quat_xyzw_to_matrix(
                pose["cam_from_world_quat_xyzw"])
            trans = np.asarray(pose["cam_from_world_t"],
                               dtype=np.float64)
            if rot.shape != (3, 3) or trans.shape != (3,):
                raise ValueError("bad pose shape")
            if camera is None:
                camera = pose["camera"]
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        centre = -rot.T @ trans
        r_wfc = rot.T  # world-from-camera, COLMAP frame
        if export_rot is not None:
            centre = export_rot @ centre
            r_wfc = export_rot @ r_wfc
        centre = centre * factor
        forward = r_wfc @ np.array([0.0, 0.0, 1.0])
        up = r_wfc @ np.array([0.0, -1.0, 0.0])
        right = r_wfc @ np.array([1.0, 0.0, 0.0])
        frames.append({
            "frame": name,
            "t_seconds": times.get(name),
            "center": [float(v) for v in centre],
            "quat_xyzw": [float(v) for v in
                          _matrix_to_quat_xyzw(r_wfc)],
            "forward": [float(v) for v in forward],
            "up": [float(v) for v in up],
            "_right": [float(v) for v in right],
            "_r_wfc": r_wfc,
        })
    if len(frames) < 2:
        raise ValueError(
            f"fewer than 2 usable camera poses "
            f"({len(frames)} usable, {skipped} skipped)")
    if camera is None:
        raise ValueError("no camera record in poses.json")

    # Animation time axis: sensor times when every frame has one
    # (shifted to start at 0), else the frame index. Mixing the
    # two would put the camera in the wrong place at the wrong time.
    if all(f["t_seconds"] is not None for f in frames):
        t0 = frames[0]["t_seconds"]
        anim_times = [float(f["t_seconds"] - t0) for f in frames]
        time_source = "capture_times"
    else:
        anim_times = [float(i) for i in range(len(frames))]
        time_source = "frame_index"

    centres = np.array([f["center"] for f in frames], dtype=np.float64)
    steps = np.linalg.norm(centres[1:] - centres[:-1], axis=1)
    route_length = float(steps.sum())

    meta = {
        "units": units_label,
        "scale_factor_applied": float(factor),
        "scale_status": (scale_rec or {}).get("status")
        if isinstance(scale_rec, dict) else None,
        "scale_to_meters": (scale_rec or {}).get("scale_to_meters")
        if isinstance(scale_rec, dict) else None,
        "alignment_applied": export_rot is not None,
        "camera": camera,
        "time_source": time_source,
        "anim_times": anim_times,
        "skipped_frames": skipped,
        "route_length": route_length,
    }
    return frames, meta


def _write_route_glb(path, frames, meta):
    """GLB: LINE_STRIP polyline through the centres + one
    perspective camera node animated along the route (one
    keyframe per registered frame). Same exported frame as the
    scene GLBs, so it overlays them exactly in Blender. glTF
    cameras look down -Z, so the node rotation maps local +Z to
    -forward and local +Y to the camera up.
    """
    centres = np.ascontiguousarray(
        [f["center"] for f in frames], dtype=np.float32)
    times = np.ascontiguousarray(meta["anim_times"], dtype=np.float32)
    n = len(frames)
    # Node rotation per frame: columns are [right | up | -forward].
    node_quats = np.zeros((n, 4), dtype=np.float32)
    for i, f in enumerate(frames):
        r_wfc = f["_r_wfc"]
        right = r_wfc @ np.array([1.0, 0.0, 0.0])
        up = r_wfc @ np.array([0.0, -1.0, 0.0])
        forward = r_wfc @ np.array([0.0, 0.0, 1.0])
        r_node = np.column_stack((right, up, -forward))
        node_quats[i] = _matrix_to_quat_xyzw(r_node).astype(np.float32)

    fx, fy, _cx, _cy, _k1, _k2 = _camera_params(meta["camera"])
    width = int(meta["camera"]["width"])
    height = int(meta["camera"]["height"])
    yfov = float(2.0 * np.arctan2(height, 2.0 * fy))
    route_length = meta["route_length"]
    zfar = max(50.0, 20.0 * route_length)

    chunks = []
    views = []
    offset = 0
    for blob, target in ((centres.tobytes(), 34962),
                         (times.tobytes(), None),
                         (centres.tobytes(), None),
                         (node_quats.tobytes(), None)):
        blob = _pad4(blob)
        view = {"buffer": 0, "byteOffset": offset,
                "byteLength": len(blob)}
        if target is not None:
            view["target"] = target
        views.append(view)
        chunks.append(blob)
        offset += len(blob)
    bin_chunk = b"".join(chunks)
    gltf = {
        "asset": {"version": "2.0",
                  "generator": "EchoScene voxel_export",
                  "extras": {"units": meta["units"],
                             "frames": n,
                             "route_length": route_length,
                             "time_source": meta["time_source"]}},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [
            {"mesh": 0, "name": "camera_route"},
            {"camera": 0, "name": "route_camera",
             "translation": [float(v) for v in centres[0]],
             "rotation": [float(v) for v in node_quats[0]]},
        ],
        "cameras": [{"type": "perspective", "name": "route_camera",
                     "perspective": {
                         "yfov": yfov,
                         "aspectRatio": width / height,
                         "znear": 0.01, "zfar": zfar}}],
        "meshes": [{"name": "camera_route", "primitives": [{
            "attributes": {"POSITION": 0}, "mode": 3}]}],
        "animations": [{
            "name": "camera_route",
            "samplers": [
                {"input": 1, "interpolation": "LINEAR", "output": 2},
                {"input": 1, "interpolation": "LINEAR", "output": 3},
            ],
            "channels": [
                {"sampler": 0,
                 "target": {"node": 1, "path": "translation"}},
                {"sampler": 1,
                 "target": {"node": 1, "path": "rotation"}},
            ],
        }],
        "buffers": [{"byteLength": len(bin_chunk)}],
        "bufferViews": views,
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": n,
             "type": "VEC3",
             "min": centres.min(axis=0).tolist(),
             "max": centres.max(axis=0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": n,
             "type": "SCALAR",
             "min": [float(times.min())], "max": [float(times.max())]},
            {"bufferView": 2, "componentType": 5126, "count": n,
             "type": "VEC3",
             "min": centres.min(axis=0).tolist(),
             "max": centres.max(axis=0).tolist()},
            {"bufferView": 3, "componentType": 5126, "count": n,
             "type": "VEC4",
             "min": node_quats.min(axis=0).tolist(),
             "max": node_quats.max(axis=0).tolist()},
        ],
    }
    json_chunk = _pad4(json.dumps(gltf).encode("utf-8"), b" ")
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(json_chunk), 0x4E4F534A))
        fh.write(json_chunk)
        fh.write(struct.pack("<II", len(bin_chunk), 0x004E4942))
        fh.write(bin_chunk)


def export_camera_route(job_dir, out_dir=None, log=None):
    """Export a finished job's camera route.

    Reads poses.json (+ alignment.json + scale.json +
    frame_capture_times.json). Writes camera_route.json (the AI
    handoff: per-frame centre + orientation + times + intrinsics
    + units) and camera_route.glb (the Blender handoff: the same
    path as a polyline + an animated camera) into out_dir
    (default: the job dir). Never raises: every failure returns
    {ok: False, reason: ...}.
    """
    def _say(msg):
        if log is not None:
            log(msg)

    job_dir = Path(job_dir)
    out_dir = Path(out_dir) if out_dir is not None else job_dir
    try:
        frames, meta = _route_frames(job_dir)
    except ValueError as exc:
        return {"stage": "camera-route", "ok": False,
                "reason": str(exc)}
    except Exception as exc:
        return {"stage": "camera-route", "ok": False,
                "reason": f"{type(exc).__name__}: {exc}"}
    try:
        fx, fy, cx, cy, k1, k2 = _camera_params(meta["camera"])
    except (KeyError, ValueError) as exc:
        return {"stage": "camera-route", "ok": False,
                "reason": f"unsupported camera model: {exc}"}
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        pub_frames = []
        for f in frames:
            pub_frames.append({
                "frame": f["frame"],
                "t_seconds": f["t_seconds"],
                "center": f["center"],
                "quat_xyzw": f["quat_xyzw"],
                "forward": f["forward"],
                "up": f["up"],
            })
        info = {
            "format": "echoscene-camera-route",
            "version": 1,
            "job": job_dir.name,
            "units": meta["units"],
            "scale": {
                "status": meta["scale_status"],
                "scale_to_meters": meta["scale_to_meters"],
                "factor_applied": meta["scale_factor_applied"],
            },
            "frame": {
                "convention": (
                    "Exported frame: the same frame as points.bin, "
                    "voxel_color.glb and textured_mesh.glb (+Y is up). "
                    "Centres are camera positions in that frame; "
                    "quat_xyzw is the world-from-camera rotation "
                    "(the camera's own axes expressed in world "
                    "coordinates), quaternion order x, y, z, w. "
                    "COLMAP cameras look down their local +Z; the "
                    "forward vector is the world direction the "
                    "camera was pointing."),
                "alignment_applied": meta["alignment_applied"],
                "quaternion_order": "xyzw",
                "quaternion_frame": "world-from-camera",
            },
            "camera": {
                "model": str(meta["camera"].get("model", "")),
                "width": int(meta["camera"]["width"]),
                "height": int(meta["camera"]["height"]),
                "params": [float(v) for v in
                           meta["camera"]["params"]],
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "k1": k1, "k2": k2,
            },
            "time_source": meta["time_source"],
            "frames": pub_frames,
            "route_length": meta["route_length"],
            "frame_count": len(frames),
            "skipped_frames": meta["skipped_frames"],
            "files": {
                "json": str(out_dir / ROUTE_JSON_NAME),
                "glb": str(out_dir / ROUTE_GLB_NAME),
            },
        }
        (out_dir / ROUTE_JSON_NAME).write_text(
            json.dumps(info, ensure_ascii=False, indent=2))
        _write_route_glb(out_dir / ROUTE_GLB_NAME, frames, meta)
        info["stage"] = "camera-route"
        info["ok"] = True
        _say(f"相机路线好了：{len(frames)} 帧，路线 "
             f"{meta['route_length']:.2f} {meta['units']}。")
        return info
    except Exception as exc:
        return {"stage": "camera-route", "ok": False,
                "reason": f"{type(exc).__name__}: {exc}"}

# ---------------------------------------------------------------------------
# BATCH EXPORT (v0.6.27)
# ---------------------------------------------------------------------------
# One run, several finished jobs: every job's available export assets
# collected into one folder, one subfolder per job. The fifth export
# item (after the coloured volume, the textured mesh and the camera
# route) -- and the one whose consumer is the NEXT tool, not this app:
# another AI that wants the whole shelf at once, not one job at a time.
#
# What a job's assets are (all built by the exporters above, nothing
# new invented here):
#   coloured volume  -- voxel_color.glb + voxel_color.ply (+ info)
#   textured mesh    -- textured_mesh.glb + .obj + .mtl + .png +
#                       mesh.ply (+ info). The slow one (Poisson +
#                       per-view texturing, minutes per job): when the
#                       job folder already holds a finished set, those
#                       files are COPIED, not rebuilt -- they are the
#                       job's own export, and rebuilding them would
#                       only produce the same files slower. When the
#                       job has none, it is built for real.
#   camera route     -- camera_route.json + camera_route.glb
#
# Honesty rules, the whole point of the batch:
#   * A job that has nothing to export (no points.bin AND no
#     poses.json -- an unfinished job, or a folder that is not a job
#     at all) is SKIPPED, never faked as exported. Its reason is in
#     the log, in the per-job status, and in batch_export.json.
#   * An asset that cannot be built is SKIPPED with its own reason
#     (no frames for the textured mesh, no poses for the route, ...),
#     while the job's other assets still export. A job counts as
#     exported when at least one asset lands; it counts as skipped
#     only when none does.
#   * Nothing here raises: a builder throwing, a copy failing, a
#     folder that vanished mid-run -- all land as a skipped asset or
#     a skipped job with the reason attached.
#   * The batch never writes into the job folders. Everything lands
#     under the output folder the user picked, so a batch run cannot
#     disturb a finished job.
# ---------------------------------------------------------------------------
import shutil

BATCH_INFO_NAME = "batch_export.json"
TEXTURED_FILE_NAMES = (
    TEXTURED_GLB_NAME, TEXTURED_OBJ_NAME, TEXTURED_MTL_NAME,
    TEXTURED_PNG_NAME, MESH_PLY_NAME, TEXTURED_INFO_NAME,
)


def _batch_job_out_dir(out_root, job_name, taken):
    """A safe subfolder name for one job: letters/digits kept,
    everything else becomes '_', and a second job with the same
    name gets -2, -3, ... so two jobs can never overwrite each
    other."""
    safe = "".join(c if (c.isalnum() or c in "-_") else "_"
                   for c in str(job_name)).strip("_") or "job"
    name = safe
    n = 2
    while name in taken:
        name = f"{safe}-{n}"
        n += 1
    taken.add(name)
    return out_root / name


def _batch_copy_textured(job_dir, out_dir):
    """Copy a job's already-finished textured-mesh set into the
    batch folder. Returns the list of copied names, or None when
    the job has no finished set (the caller then builds it)."""
    job_dir = Path(job_dir)
    out_dir = Path(out_dir)
    if not (job_dir / TEXTURED_GLB_NAME).exists():
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in TEXTURED_FILE_NAMES:
        src = job_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
            copied.append(name)
    return copied or None


def _batch_one_asset(fn, job_dir, out_dir, log):
    """Run one asset builder and normalise the outcome to
    {ok, files|reason, ...}. The builders already return
    {ok: False, reason} instead of raising, but the try/except
    is the backstop: a throw is a skipped asset, never a dead
    batch."""
    def _say(msg):
        if log is not None:
            log(msg)

    try:
        info = fn(job_dir, out_dir)
    except Exception as exc:
        return {"ok": False,
                "reason": f"{type(exc).__name__}: {exc}"}
    if not isinstance(info, dict) or not info.get("ok"):
        reason = ""
        if isinstance(info, dict):
            reason = str(info.get("reason") or "未知原因")
        return {"ok": False, "reason": reason or "未知原因"}
    files = {}
    if isinstance(info.get("files"), dict):
        files = {k: str(v) for k, v in info["files"].items()}
    out = {"ok": True, "files": files}
    for key in ("voxels", "triangles", "views_used",
                "untextured_triangles", "frame_count", "route_length"):
        if key in info:
            out[key] = info[key]
    return out


def export_batch(job_dirs, out_root, log=None):
    """Export several finished jobs in one run.

    job_dirs: the jobs the user picked (paths or strings).
    out_root: the folder the user picked; one subfolder per job
        is created under it, plus batch_export.json (the summary).
    Returns {stage: "batch-export", ok, out_root, jobs: [...],
    exported, skipped, total}. ok is True when the run itself
    completed (even when every job was skipped -- that is a
    result, reported per job, not a crash); False only when the
    output folder itself could not be used.
    """
    def _say(msg):
        if log is not None:
            log(msg)

    out_root = Path(out_root)
    try:
        out_root.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return {"stage": "batch-export", "ok": False,
                "reason": f"输出文件夹打不开：{exc}"}
    jobs = list(job_dirs or [])
    taken = set()
    results = []
    for idx, raw in enumerate(jobs):
        job_dir = Path(str(raw))
        name = job_dir.name or str(raw)
        _say(f"批量导出：第 {idx + 1}/{len(jobs)} 个任务：{name}…")
        entry = {"name": name, "dir": str(job_dir),
                 "status": "skipped", "assets": {}}
        if not job_dir.exists() or not job_dir.is_dir():
            entry["reason"] = "任务文件夹不存在"
            _say(f"批量导出：{name} 跳过（任务文件夹不存在）。")
            results.append(entry)
            continue
        has_points = (job_dir / "points.bin").exists()
        has_poses = (job_dir / "poses.json").exists()
        if not has_points and not has_poses:
            entry["reason"] = ("这个任务还没做完"
                               "（没有点云也没有相机位姿）")
            _say(f"批量导出：{name} 跳过（还没做完，没有点云"
                 f"也没有相机位姿）。")
            results.append(entry)
            continue
        out_dir = _batch_job_out_dir(out_root, name, taken)
        entry["out_dir"] = str(out_dir)
        assets = entry["assets"]
        # -- coloured volume (fast, always built fresh) ----------
        if has_points:
            assets["voxel"] = _batch_one_asset(
                lambda j, o: export_colored_voxels(j, o),
                job_dir, out_dir, log)
        else:
            assets["voxel"] = {
                "ok": False,
                "reason": "points.bin missing (run the job first)"}
        # -- textured mesh: copy the job's own finished set when
        #    it has one, otherwise build it for real ------------
        if has_points and has_poses:
            copied = None
            try:
                copied = _batch_copy_textured(job_dir, out_dir)
            except Exception as exc:
                assets["textured"] = {
                    "ok": False,
                    "reason": f"复制现成的带贴图网格失败：{exc}"}
            if copied is not None:
                assets["textured"] = {
                    "ok": True, "copied": True,
                    "files": {c: str(out_dir / c) for c in copied}}
            elif "textured" not in assets:
                assets["textured"] = _batch_one_asset(
                    lambda j, o: export_textured_mesh(j, o, log=log),
                    job_dir, out_dir, log)
        elif not has_points:
            assets["textured"] = {
                "ok": False,
                "reason": "points.bin missing (run the job first)"}
        else:
            assets["textured"] = {
                "ok": False,
                "reason": "poses.json missing (run the job first)"}
        # -- camera route (fast, always built fresh) --------------
        if has_poses:
            assets["route"] = _batch_one_asset(
                lambda j, o: export_camera_route(j, o, log=log),
                job_dir, out_dir, log)
        else:
            assets["route"] = {
                "ok": False,
                "reason": "poses.json missing (run the job first)"}
        done = [k for k, v in assets.items() if v.get("ok")]
        missed = [(k, v.get("reason", ""))
                  for k, v in assets.items() if not v.get("ok")]
        if done:
            entry["status"] = "exported"
            entry["exported_assets"] = done
            _say(f"批量导出：{name} 好了（"
                 + "、".join(done) + "）。"
                 + (f"跳过："
                    + "、".join(f"{k}（{r}）" for k, r in missed)
                    if missed else ""))
        else:
            entry["status"] = "skipped"
            entry["reason"] = ("、".join(
                f"{k}：{r}" for k, r in missed) or "没有可导出的内容")
            _say(f"批量导出：{name} 跳过（{entry['reason']}）。"
                 f"什么都没导出来。")
        results.append(entry)
    exported = sum(1 for r in results if r["status"] == "exported")
    skipped = len(results) - exported
    info = {
        "stage": "batch-export",
        "ok": True,
        "format": "echoscene-batch-export",
        "version": 1,
        "out_root": str(out_root),
        "jobs": results,
        "exported": exported,
        "skipped": skipped,
        "total": len(results),
    }
    try:
        (out_root / BATCH_INFO_NAME).write_text(
            json.dumps(info, ensure_ascii=False, indent=2))
        info["info_json"] = str(out_root / BATCH_INFO_NAME)
    except Exception as exc:
        return {"stage": "batch-export", "ok": False,
                "reason": f"批量导出的汇总写不进去：{exc}",
                "jobs": results, "exported": exported,
                "skipped": skipped, "total": len(results)}
    _say(f"批量导出：{exported} 个任务导好了，{skipped} 个跳过。"
         f"文件夹：{out_root}")
    return info
