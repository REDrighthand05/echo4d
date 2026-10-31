"""Mesh stage: Poisson surface reconstruction from the fused point cloud.

Consumes a job's points.bin (fuse output, 15 bytes/point: float32 xyz +
uint8 rgb) plus the camera centres in poses.json, and produces:
  mesh.ply          full Poisson mesh (density-trimmed)
  mesh_vertices.bin web payload: 15 bytes/vertex, same layout as points.bin
  mesh_indices.bin  web payload: uint32 little-endian triangle indices
                    (the web mesh is quadric-decimated to <= 250k triangles)
Stage info is written to mesh_info.json (kept separate from job.json,
which belongs to the fuse stage).

Method notes (deterministic throughout):
  * oversized inputs are voxel-decimated keeping the lowest-index point
    per voxel (voxel edge = cloud extent / 220, grown until <= 400k);
  * normals are Open3D PCA estimates (KNN 20), each flipped toward the
    nearest camera centre from poses.json;
  * Poisson solve (depth 9, linear_fit), then the lowest-15%-density
    vertices are removed with their triangles and the mesh is cropped
    to the cloud's 1-99 percentile box - the low-density iso-surface is
    where the solver balloons into empty space.
"""
import json
import time
from pathlib import Path

import numpy as np

try:
    from .fuse import _quat_xyzw_to_matrix
except ImportError:  # loose-script execution
    from fuse import _quat_xyzw_to_matrix

MAX_INPUT_POINTS = 400_000
WEB_MAX_TRIANGLES = 250_000
POISSON_DEPTH = 9
DENSITY_TRIM_PERCENTILE = 15


def _read_points_bin(path: Path):
    raw = np.fromfile(path, dtype=np.uint8)
    n = len(raw) // 15
    raw = raw[: n * 15].reshape(n, 15)
    points = raw[:, 0:12].copy().view("<f4").reshape(n, 3).astype(np.float64)
    colors = raw[:, 12:15].copy()
    return points, colors


def _voxel_decimate(points, colors, max_points=MAX_INPUT_POINTS):
    """Keep the lowest-index point of each voxel; grow the voxel until
    the cloud fits max_points. Starts at extent/220 per the stage spec."""
    if len(points) <= max_points:
        return points, colors
    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    voxel = max(extent / 220.0, 1e-6)
    while True:
        keys = np.floor(points / voxel).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        if len(first) <= max_points or voxel > extent:
            order = np.sort(first)
            return points[order], colors[order]
        voxel *= 1.25


def _camera_centers(job_dir: Path):
    """World-space camera centres from poses.json. fuse's convention is
    world = (cam - t) @ R, so the centre (cam = 0) is (-t) @ R."""
    poses = json.loads((job_dir / "poses.json").read_text())
    centers = []
    for pose in poses.values():
        rot = _quat_xyzw_to_matrix(pose["cam_from_world_quat_xyzw"])
        t = np.array(pose["cam_from_world_t"], dtype=np.float64)
        centers.append((-t) @ rot)
    if not centers:
        raise ValueError("poses.json contains no camera poses")
    return np.array(centers)


def _orient_normals(points, normals, centers):
    """Flip each normal toward its nearest camera centre (chunked)."""
    nearest = np.empty(len(points), dtype=np.int64)
    for s in range(0, len(points), 16384):
        chunk = points[s : s + 16384]
        d = chunk[:, None, :] - centers[None, :, :]
        nearest[s : s + len(chunk)] = np.argmin(
            np.einsum("ijk,ijk->ij", d, d), axis=1
        )
    flip = np.einsum("ij,ij->i", normals, centers[nearest] - points) < 0
    out = normals.copy()
    out[flip] *= -1
    return out


def _write_vertices_bin(verts, vcols, path: Path):
    n = len(verts)
    data = np.empty((n, 15), dtype=np.uint8)
    data[:, 0:12] = verts.astype("<f4").view(np.uint8).reshape(n, 12)
    data[:, 12:15] = vcols
    data.tofile(path)


def _mesh_arrays(mesh):
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    if mesh.has_vertex_colors():
        vcols = np.clip(np.asarray(mesh.vertex_colors) * 255.0, 0, 255).astype(np.uint8)
    else:
        vcols = np.full((len(verts), 3), 255, np.uint8)
    return verts, tris, vcols


def _stage_mesh_impl(job_dir: Path, o3d) -> dict:
    t0 = time.time()
    points_path = job_dir / "points.bin"
    if not points_path.exists():
        raise ValueError(f"missing input {points_path}")
    points, colors = _read_points_bin(points_path)
    n_input = len(points)
    if n_input < 100:
        raise ValueError(f"too few input points ({n_input})")
    points, colors = _voxel_decimate(points, colors)
    lo = np.percentile(points, 1, axis=0)
    hi = np.percentile(points, 99, axis=0)
    centers = _camera_centers(job_dir)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=20))
    pcd.normals = o3d.utility.Vector3dVector(
        _orient_normals(points, np.asarray(pcd.normals), centers)
    )

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH, linear_fit=True
    )
    densities = np.asarray(densities, dtype=np.float64)
    thr = np.percentile(densities, DENSITY_TRIM_PERCENTILE)
    mesh.remove_vertices_by_mask(densities < thr)
    mesh.remove_unreferenced_vertices()
    mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(lo, hi))
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    tris_full = len(mesh.triangles)
    if tris_full == 0:
        raise ValueError("Poisson mesh is empty after density trim/crop")
    o3d.io.write_triangle_mesh(str(job_dir / "mesh.ply"), mesh)

    if tris_full > WEB_MAX_TRIANGLES:
        web = mesh.simplify_quadric_decimation(WEB_MAX_TRIANGLES)
        web.remove_unreferenced_vertices()
    else:
        web = mesh
    verts, tris, vcols = _mesh_arrays(web)
    _write_vertices_bin(verts, vcols, job_dir / "mesh_vertices.bin")
    tris.astype("<u4").tofile(job_dir / "mesh_indices.bin")

    return {
        "stage": "mesh",
        "ok": True,
        "input_points": int(n_input),
        "points_used": int(len(points)),
        "verts_full": int(len(mesh.vertices)),
        "tris_full": int(tris_full),
        "verts_web": int(len(verts)),
        "tris_web": int(len(tris)),
        "mesh_ply": str(job_dir / "mesh.ply"),
        "mesh_vertices_bin": str(job_dir / "mesh_vertices.bin"),
        "mesh_indices_bin": str(job_dir / "mesh_indices.bin"),
        "seconds": round(time.time() - t0, 2),
    }


def stage_mesh(job_dir) -> dict:
    job_dir = Path(job_dir)
    try:
        import open3d as o3d
    except Exception as exc:  # never break the pipeline on a missing lib
        info = {"stage": "mesh", "ok": False, "reason": f"open3d unavailable: {exc}"}
    else:
        try:
            info = _stage_mesh_impl(job_dir, o3d)
        except Exception as exc:
            info = {"stage": "mesh", "ok": False, "reason": str(exc)}
    try:
        (job_dir / "mesh_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2)
        )
    except OSError:
        pass
    return info


if __name__ == "__main__":
    import sys

    print(json.dumps(stage_mesh(Path(sys.argv[1])), ensure_ascii=False, indent=2))
