"""Measured-only Stanford Lounge: complete SMALL planar holes with bounded support.

No supplied scan is read. Planar patches are *inferred* from measured RGB-D,
and are exported separately with purple provenance; never call them measured
surfaces. Large unobserved regions deliberately remain empty.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

from echo4d.model import Camera, DataError
from lounge_spatial_prior_real import VOXEL_M, read_frames, select_views
from tum_xyz_open3d_320 import export_glb, free_view, fuse, panel, render

CELL_M = .03
MAX_HOLE_M2 = .12
MAX_RADIUS_M = .18


def bounded_planar_holes(occupied, *, cell_m=CELL_M,
                         max_area_m2=MAX_HOLE_M2, max_radius_m=MAX_RADIUS_M):
    """Find only tiny, enclosed holes or 1-pixel cracks in observed planar faces.

    Never fill open room boundaries or unobserved backs of objects.
    """
    mask = np.asarray(occupied, dtype=bool)
    if mask.ndim != 2 or cell_m <= 0 or max_area_m2 <= 0 or max_radius_m <= 0:
        raise ValueError("Invalid planar grid or metric limits")
    # A one-cell close bridges very small TSDF cracks; enclosed holes are
    # additionally eligible only if surrounded by rasterized real triangles.
    closed = ndimage.binary_closing(mask, structure=np.ones((3, 3), bool))
    candidate = ndimage.binary_fill_holes(closed) & ~mask
    labels, count = ndimage.label(candidate)
    distance = ndimage.distance_transform_edt(~mask) * cell_m
    result = np.zeros_like(mask)
    for label in range(1, count + 1):
        ys, xs = np.nonzero(labels == label)
        if not len(ys):
            continue
        if len(ys) * cell_m ** 2 > max_area_m2:
            continue
        if np.max(distance[ys, xs]) > max_radius_m:
            continue
        result[ys, xs] = True
    return result


def planar_patch_mesh(o3d, measured, *, seed=7, max_planes=4):
    """Rasterize actual near-plane triangles, then bridge bounded small gaps."""
    xyz = np.asarray(measured.vertices)
    faces = np.asarray(measured.triangles)
    colors = np.asarray(measured.vertex_colors)
    if len(xyz) < 5000 or len(faces) < 5000 or len(colors) != len(xyz):
        raise DataError("Measured RGB mesh is missing geometry or colors")
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(xyz), min(len(xyz), 100000), replace=False)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz[sample])
    all_vertices, all_colors, all_faces = [], [], []
    details = []
    for k in range(max_planes):
        if len(cloud.points) < 4000:
            break
        plane, inliers = cloud.segment_plane(
            distance_threshold=.025, ransac_n=3, num_iterations=500)
        if len(inliers) < 4000:
            break
        normal = np.asarray(plane[:3], dtype=float)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            break
        normal /= norm
        origin = -float(plane[3]) * normal / norm
        reference = np.array([0., 0., 1.])
        if abs(np.dot(reference, normal)) > .90:
            reference = np.array([0., 1., 0.])
        u = np.cross(normal, reference)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        residual = np.abs(xyz @ normal + float(plane[3]) / norm)
        near = residual < .022
        planar_faces = faces[np.all(near[faces], axis=1)]
        if len(planar_faces) < 2000:
            cloud = cloud.select_by_index(inliers, invert=True)
            continue
        edge = xyz[planar_faces]
        lengths = np.maximum.reduce([
            np.linalg.norm(edge[:, 0] - edge[:, 1], axis=1),
            np.linalg.norm(edge[:, 1] - edge[:, 2], axis=1),
            np.linalg.norm(edge[:, 2] - edge[:, 0], axis=1)])
        planar_faces = planar_faces[lengths < .13]
        if len(planar_faces) < 2000:
            cloud = cloud.select_by_index(inliers, invert=True)
            continue
        used = np.unique(planar_faces)
        points2 = np.column_stack(((xyz[used] - origin) @ u,
                                   (xyz[used] - origin) @ v))
        lo = np.percentile(points2, 1, axis=0) - .06
        hi = np.percentile(points2, 99, axis=0) + .06
        dims = np.ceil((hi - lo) / CELL_M).astype(int) + 2
        if np.any(dims < 24) or np.any(dims > 800) or np.prod(dims) > 450000:
            cloud = cloud.select_by_index(inliers, invert=True)
            continue
        # Rasterize real measured triangles, not a convex hull of observations.
        canvas = Image.new("1", (int(dims[0]), int(dims[1])), 0)
        draw = ImageDraw.Draw(canvas)
        projected = np.column_stack(((xyz - origin) @ u,
                                     (xyz - origin) @ v))
        ij = (projected - lo) / CELL_M
        for tri in planar_faces:
            poly = [(float(ij[t, 0]), float(ij[t, 1])) for t in tri]
            draw.polygon(poly, fill=1)
        occupied = np.asarray(canvas, dtype=bool).T
        hole = bounded_planar_holes(occupied)
        ys, xs = np.nonzero(hole)
        if not len(xs):
            details.append(dict(plane=k, observed_faces=len(planar_faces),
                                patch_cells=0))
            cloud = cloud.select_by_index(inliers, invert=True)
            continue
        # Nearest *measured planar* RGB only; color is interpolated, not new RGB.
        tree = cKDTree(points2)
        base_index = sum(len(part) for part in all_vertices)
        vertices = np.empty((len(xs) * 4, 3), float)
        grid_uv = np.empty((len(xs) * 4, 2), float)
        triangles = np.empty((len(xs) * 2, 3), np.int32)
        for i, (x, y) in enumerate(zip(xs, ys)):
            quad = lo + CELL_M * np.array([
                [x, y], [x+1, y], [x+1, y+1], [x, y+1]], float)
            grid_uv[4*i:4*i+4] = quad
            vertices[4*i:4*i+4] = (origin + quad[:, 0, None] * u
                                          + quad[:, 1, None] * v)
            b = base_index + 4*i
            triangles[2*i] = [b, b+1, b+2]
            triangles[2*i+1] = [b, b+2, b+3]
        _, nearest = tree.query(grid_uv, workers=2)
        patch_colors = colors[used[nearest]]
        all_vertices.append(vertices)
        all_colors.append(patch_colors)
        all_faces.append(triangles)
        details.append(dict(plane=k, observed_faces=len(planar_faces),
                            patch_cells=len(xs), patch_area_m2=len(xs)*CELL_M**2))
        cloud = cloud.select_by_index(inliers, invert=True)
    patch = o3d.geometry.TriangleMesh()
    if all_vertices:
        patch.vertices = o3d.utility.Vector3dVector(np.concatenate(all_vertices))
        patch.triangles = o3d.utility.Vector3iVector(np.concatenate(all_faces))
        patch.vertex_colors = o3d.utility.Vector3dVector(np.concatenate(all_colors))
        patch.compute_vertex_normals()
    return patch, details


def experiment(work, output, receipt):
    import open3d as o3d
    work.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    dataset = o3d.data.LoungeRGBDImages(data_root=str(work))
    params = o3d.io.read_pinhole_camera_trajectory(
        dataset.trajectory_log_path).parameters
    _, selected, poses, pose_indices = select_views(params, dataset.color_paths)
    intr = params[pose_indices[0]].intrinsic
    k = intr.intrinsic_matrix
    full = Camera(intr.width, intr.height, float(k[0, 0]), float(k[1, 1]),
                  float(k[0, 2]), float(k[1, 2]))
    if (full.width, full.height) != (640, 480):
        raise DataError("Unexpected real Lounge RGB-D calibration")
    half = full.scaled(2)
    for i in selected:
        q = params[pose_indices[i]].intrinsic
        if (q.width != full.width or q.height != full.height
                or not np.allclose(q.intrinsic_matrix, k, atol=1e-3)):
            raise DataError("Lounge camera intrinsics vary across frames")
    frames, originals, photos = read_frames(dataset, selected, poses, full, half)
    measured, _ = fuse(o3d, [frames[i] for i in selected], half,
                       pixel_stride=1, voxel=VOXEL_M)
    patch, planes = planar_patch_mesh(o3d, measured)
    hybrid = o3d.geometry.TriangleMesh(measured) + patch
    provenance = o3d.geometry.TriangleMesh(hybrid)
    origin_color = np.empty((len(hybrid.vertices), 3), float)
    origin_color[:len(measured.vertices)] = [.10, .85, .35]
    origin_color[len(measured.vertices):] = [.73, .27, .92]
    provenance.vertex_colors = o3d.utility.Vector3dVector(origin_color)
    for name, mesh in (
        ("MEASURED-144-only.glb", measured),
        ("MEASURED-plus-INFERRED-small-planes.glb", hybrid),
        ("ORIGIN-green-MEASURED-purple-INFERRED.glb", provenance),
    ):
        export_glb(mesh, output / name)
        print("Exported", name, len(mesh.triangles), "faces", flush=True)
    labels = ["REAL / FREE", "MEASURED 144",
              "SMALL PLANE FILL", "SOURCE GREEN/PURPLE"]
    boards, views = [], []
    for i in photos:
        Image.fromarray(originals[i]).save(output / f"REAL-lounge-{i:04d}.png")
        photo = np.asarray(Image.fromarray(originals[i]).resize(
            (half.width, half.height), Image.Resampling.LANCZOS))
        imgs = [photo] + [render(o3d, m, half, poses[i])
                          for m in (measured, hybrid, provenance)]
        name = f"recorded-{i:04d}.png"
        boards.append(panel(output / name, imgs, labels))
        views.append(dict(kind="recorded", frame=i, file=name))
    for i, side in ((photos[0], -1.), (photos[-1], 1.)):
        pose = free_view(poses[i], side)
        imgs = [render(o3d, m, half, pose)
                for m in (measured, hybrid, provenance)]
        name = f"free-{i:04d}.png"
        boards.append(panel(output / name,
                            [np.full_like(imgs[0], 20)] + imgs, labels))
        views.append(dict(kind="free", reference_frame=i, file=name))
    board = Image.new("RGB", (half.width*4, (half.height+27)*5))
    for j, image in enumerate(boards):
        board.paste(image, (0, j*(half.height+27)))
    board.save(output / "lounge-five-view-small-planar-holes.png")
    receipt.update(
        status="real_meshes_exported_visual_review_pending",
        original_resolution=[full.width, full.height],
        fusion_resolution=[half.width, half.height],
        measured_frames=len(selected),
        measured_triangles=len(measured.triangles),
        inferred_patch_triangles=len(patch.triangles),
        planes=planes, photos=photos, views=views,
        provenance="GREEN=MEASURED sensor RGB-D; PURPLE=PLANAR INFERRED from measured support; no scan PRIOR",
        visual_verdict="PENDING HUMAN REVIEW; free orbit FAIL until validated",
        issue_74="FAIL unchanged",
        old_desk="FAIL 16.7779cm > 15cm unchanged")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    receipt = {
        "scene": "Stanford Lounge original 640x480 real RGB-D",
        "source": "Open3D LoungeRGBDImages; 144 measured views and known poses",
        "status": "running",
        "run_url": (f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/"
                    f"{os.getenv('GITHUB_RUN_ID')}" if os.getenv("GITHUB_RUN_ID") else None)}
    try:
        experiment(args.work, args.output, receipt)
        code = 0
    except Exception as exc:
        traceback.print_exc()
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        code = 1
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "result.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"status": receipt["status"],
                          "error": receipt.get("error")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
