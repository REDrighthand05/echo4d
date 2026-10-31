"""Poisson surface reconstruction experiment (deterministic, no voting).

Rebuilds the raw back-projected cloud from a job's poses + depth, orients
normals toward the observing cameras (known geometry, deterministic rule),
then solves the Poisson equation for an indicator function whose gradient
matches the normal field (Kazhdan et al.) and extracts the iso-surface.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import echo_scene.fuse as F


def build_raw_cloud(job: Path):
    poses = json.loads((job / "poses.json").read_text())
    rel_vals, inv_z, per_frame = F._collect_anchor_pairs(job, job / "depth")
    a_glob, b_glob = F._robust_affine_fit(inv_z, rel_vals)
    fits = {}
    for stem, (f_rel, f_inv) in per_frame.items():
        if len(f_rel) < 60:
            continue
        a_i, b_i = F._robust_affine_fit(f_inv, f_rel)
        if a_i > 0 and F._fit_mad(f_inv, f_rel, a_i, b_i) <= F._fit_mad(f_inv, f_rel, a_glob, b_glob):
            fits[stem] = (a_i, b_i)
    all_pts, all_col, all_cam = [], [], []
    for name, pose in sorted(poses.items()):
        stem = Path(name).stem
        npy = job / "depth" / f"{stem}.npy"
        img_path = job / "frames" / name
        if not npy.exists() or not img_path.exists():
            continue
        a, b = fits.get(stem, (a_glob, b_glob))
        rel = np.load(npy).astype(np.float64)
        denom = rel - b
        valid = denom > 1e-4
        z_map = np.where(valid, a / np.where(valid, denom, 1.0), np.nan)
        z_hi = np.nanpercentile(z_map[valid], 99) if valid.any() else np.nan
        valid &= z_map < z_hi * 1.5
        cam = pose["camera"]["params"]
        f, cx, cy = float(cam[0]), float(cam[1]), float(cam[2])
        rot = F._quat_xyzw_to_matrix(pose["cam_from_world_quat_xyzw"])
        t = np.array(pose["cam_from_world_t"], dtype=np.float64)
        cam_center = -rot.T @ t
        stride = 4
        ys, xs = np.mgrid[0:rel.shape[0]:stride, 0:rel.shape[1]:stride]
        zs = z_map[::stride, ::stride]
        ok = valid[::stride, ::stride] & np.isfinite(zs)
        if zs.shape[0] > 1 and zs.shape[1] > 1:
            jump = np.zeros_like(zs)
            jump[:, :-1] = np.maximum(jump[:, :-1], np.abs(zs[:, 1:] - zs[:, :-1]))
            jump[:-1, :] = np.maximum(jump[:-1, :], np.abs(zs[1:, :] - zs[:-1, :]))
            ok &= jump <= 0.18 * np.maximum(zs, 1e-6)
        if not ok.any():
            continue
        xs_f, ys_f, zs_f = xs[ok].astype(np.float64), ys[ok].astype(np.float64), zs[ok]
        cam_pts = np.stack([(xs_f - cx) / f * zs_f, (ys_f - cy) / f * zs_f, zs_f], axis=1)
        world = (cam_pts - t) @ rot
        img = np.asarray(Image.open(img_path).convert("RGB"))
        all_pts.append(world)
        all_col.append(img[::stride, ::stride][ok])
        all_cam.append(np.broadcast_to(cam_center, world.shape).copy())
    return (np.concatenate(all_pts), np.concatenate(all_col), np.concatenate(all_cam))


def main(job_dir: str, target: int = 400_000):
    import open3d as o3d

    job = Path(job_dir)
    out = job / "poisson"
    out.mkdir(exist_ok=True)
    pts, cols, cams = build_raw_cloud(job)
    print("raw:", len(pts), flush=True)
    lo = np.percentile(pts, 1, axis=0)
    hi = np.percentile(pts, 99, axis=0)
    keep = np.all((pts >= lo) & (pts <= hi), axis=1)
    pts, cols, cams = pts[keep], cols[keep], cams[keep]
    step = max(1, len(pts) // target)
    pts, cols, cams = pts[::step], cols[::step], cams[::step]
    print("decimated:", len(pts), flush=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=20))
    normals = np.asarray(pcd.normals)
    to_cam = cams - pts
    flip = np.einsum("ij,ij->i", normals, to_cam) < 0
    normals[flip] *= -1
    pcd.normals = o3d.utility.Vector3dVector(normals)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9, linear_fit=True
    )
    print("mesh raw:", len(mesh.vertices), "verts,", len(mesh.triangles), "tris", flush=True)
    bbox = o3d.geometry.AxisAlignedBoundingBox(lo, hi)
    mesh = mesh.crop(bbox)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    print("mesh cropped:", len(mesh.vertices), "verts,", len(mesh.triangles), "tris", flush=True)
    o3d.io.write_triangle_mesh(str(out / "mesh_poisson.ply"), mesh)

    verts = np.asarray(mesh.vertices)
    vcols = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8) if mesh.has_vertex_colors() else np.full((len(verts), 3), 200, np.uint8)
    data = np.empty((len(verts), 15), dtype=np.uint8)
    data[:, 0:12] = verts.astype(np.float32).view(np.uint8).reshape(len(verts), 12)
    data[:, 12:15] = vcols
    data.tofile(out / "mesh_vertices.bin")
    print("POISSON_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
