"""Poisson review pass: density-trimmed mesh + shaded mesh renders.

Second look at the Poisson reconstruction for a job, per the one-by-one
review rule:
  1. Raw back-projected cloud (reused from poisson_exp) is trimmed to its
     1-99 percentile box, deterministically decimated, and cached to
     poisson/input_cloud.npz so reruns skip the rebuild.
  2. Normals: Open3D KNN(20) PCA estimate, oriented toward each point's
     source camera (same deterministic rule as poisson_exp).
  3. Poisson solve (depth=9, linear_fit). Vertices whose solver density
     is below the 15th percentile are removed with their triangles (the
     low-density iso-surface is where ballooning/tendrils live), then the
     mesh is cropped to the percentile box.
  4. A tiny flat-shaded triangle renderer (numpy + PIL) produces a
     perspective and a top-down view so the MESH (not a vertex splat)
     can be judged.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import poisson_exp


def get_input_cloud(job: Path, target: int = 400_000):
    """Return (pts, cols, cams, lo, hi); cache the decimated cloud."""
    cache = job / "poisson" / "input_cloud.npz"
    if cache.exists():
        d = np.load(cache)
        print("cache hit:", cache, flush=True)
        return d["points"], d["colors"], d["cams"], d["lo"], d["hi"]
    pts, cols, cams = poisson_exp.build_raw_cloud(job)
    print("raw:", len(pts), flush=True)
    lo = np.percentile(pts, 1, axis=0)
    hi = np.percentile(pts, 99, axis=0)
    keep = np.all((pts >= lo) & (pts <= hi), axis=1)
    pts, cols, cams = pts[keep], cols[keep], cams[keep]
    step = max(1, len(pts) // target)
    pts, cols, cams = pts[::step], cols[::step], cams[::step]
    print("decimated:", len(pts), flush=True)
    cache.parent.mkdir(exist_ok=True)
    np.savez(cache, points=pts, colors=cols, cams=cams, lo=lo, hi=hi)
    return pts, cols, cams, lo, hi


def build_trimmed_mesh(pts, cols, cams, lo, hi):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=20))
    normals = np.asarray(pcd.normals)
    flip = np.einsum("ij,ij->i", normals, cams - pts) < 0
    normals[flip] *= -1
    pcd.normals = o3d.utility.Vector3dVector(normals)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9, linear_fit=True
    )
    densities = np.asarray(densities, dtype=np.float64)
    before = (len(mesh.vertices), len(mesh.triangles))
    print("mesh raw:", before[0], "verts,", before[1], "tris", flush=True)

    thr = np.percentile(densities, 15)
    remove = densities < thr
    mesh.remove_vertices_by_mask(remove)
    mesh.remove_unreferenced_vertices()
    mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(lo, hi))
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    after = (len(mesh.vertices), len(mesh.triangles))
    print("mesh trimmed:", after[0], "verts,", after[1], "tris",
          f"(density thr {thr:.4g}, removed {int(remove.sum())} verts)", flush=True)
    return mesh, before, after


def save_vertices_bin(mesh, path: Path):
    verts = np.asarray(mesh.vertices)
    if mesh.has_vertex_colors():
        vcols = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
    else:
        vcols = np.full((len(verts), 3), 200, np.uint8)
    data = np.empty((len(verts), 15), dtype=np.uint8)
    data[:, 0:12] = verts.astype(np.float32).view(np.uint8).reshape(len(verts), 12)
    data[:, 12:15] = vcols
    data.tofile(path)


def _face_colors(tris_v, tris_c, normals_view, light):
    """Flat shading: mean vertex color scaled by |normal . light| mix."""
    base = tris_c.mean(axis=1)  # (M,3) float 0..255
    d = np.abs(np.einsum("ij,j->i", normals_view, light))
    shade = 0.30 + 0.70 * d
    return np.clip(base * shade[:, None], 0, 255).astype(np.uint8)


def render_perspective(verts, cols, tris, path, W=900, H=900,
                       az_deg=30.0, el_deg=25.0):
    az, el = np.radians(az_deg), np.radians(el_deg)
    ctr = (verts.max(0) + verts.min(0)) / 2
    p = verts - ctr
    eye = np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
    fwd = -eye / np.linalg.norm(eye)
    right = np.cross(fwd, [0, -1, 0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    cam = np.stack([p @ right, p @ up, p @ fwd], axis=1)  # z away from viewer
    diag = np.linalg.norm(verts.max(0) - verts.min(0))
    zoff = 1.6 * diag
    f = 0.5 * W / np.tan(np.radians(30))
    tv = cam[tris]                       # (M,3,3)
    tz = tv[:, :, 2] + zoff
    sx = f * tv[:, :, 0] / tz + W / 2
    sy = -f * tv[:, :, 1] / tz + H / 2    # screen y down, world up is +cam-y
    meanz = tz.mean(axis=1)
    # face normals in camera space
    n = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True); ln[ln == 0] = 1
    n = n / ln
    light = np.array([-0.35, 0.55, -0.75]); light /= np.linalg.norm(light)
    colors = _face_colors(tv, cols[tris].astype(np.float64), n, light)
    order = np.argsort(-meanz)
    img = Image.new("RGB", (W, H), (11, 14, 19))
    d = ImageDraw.Draw(img)
    for i in order:
        if tz[i].min() <= 1e-3:
            continue
        poly = [(float(sx[i, k]), float(sy[i, k])) for k in range(3)]
        if max(p[0] for p in poly) < 0 or max(p[1] for p in poly) < 0:
            continue
        if min(p[0] for p in poly) > W or min(p[1] for p in poly) > H:
            continue
        d.polygon(poly, fill=tuple(int(v) for v in colors[i]))
    img.save(path)
    print("saved", path, flush=True)


def render_topdown(verts, cols, tris, path, W=900, H=900):
    xs, ys, zs = verts[:, 0], verts[:, 1], verts[:, 2]
    span = max(xs.max() - xs.min(), zs.max() - zs.min())
    sx = (xs - xs.min()) / span * (W - 40) + 20
    sy = (zs - zs.min()) / span * (H - 40) + 20
    tv = verts[tris]
    n = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True); ln[ln == 0] = 1
    n = n / ln
    light = np.array([-0.35, -0.75, 0.30]); light /= np.linalg.norm(light)
    colors = _face_colors(tv, cols[tris].astype(np.float64), n, light)
    depth = tv[:, :, 1].mean(axis=1)     # +Y is down in COLMAP world: far first
    order = np.argsort(-depth)
    sp = np.stack([sx, sy], axis=1)[tris]
    img = Image.new("RGB", (W, H), (11, 14, 19))
    d = ImageDraw.Draw(img)
    for i in order:
        poly = [(float(sp[i, k, 0]), float(sp[i, k, 1])) for k in range(3)]
        d.polygon(poly, fill=tuple(int(v) for v in colors[i]))
    img.save(path)
    print("saved", path, flush=True)


def main(job_dir: str):
    import open3d as o3d

    job = Path(job_dir)
    out = job / "poisson"
    out.mkdir(exist_ok=True)
    pts, cols, cams, lo, hi = get_input_cloud(job)
    mesh, before, after = build_trimmed_mesh(pts, cols, cams, lo, hi)
    o3d.io.write_triangle_mesh(str(out / "mesh_poisson_trimmed.ply"), mesh)
    save_vertices_bin(mesh, out / "mesh_trimmed_vertices.bin")

    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    if mesh.has_vertex_colors():
        vcols = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
    else:
        vcols = np.full((len(verts), 3), 200, np.uint8)
    render_perspective(verts, vcols, tris, out / "shaded_view.png")
    render_topdown(verts, vcols, tris, out / "shaded_top.png")
    print(f"SUMMARY before={before} after={after}", flush=True)
    print("REVIEW_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
