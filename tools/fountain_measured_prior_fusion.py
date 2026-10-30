"""Real Fountain indoor RGB-D, known keyframe poses and a LABELED existing mesh PRIOR.

Open3D's official SampleFountainRGBDImages contains 33 real RGB-D keyframes,
the associated camera poses and an existing reconstructed mesh. The supplied
mesh is NEVER described as independent reconstruction from these 33 images.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import traceback

import numpy as np
from PIL import Image

from echo4d.model import Camera, DataError, Frame
from tum_xyz_open3d_320 import fuse, export_glb, render, free_view, panel


def read_sample(o3d, dataset):
    trajectory = o3d.io.read_pinhole_camera_trajectory(dataset.keyframe_poses_log_path)
    params = trajectory.parameters
    if not (len(dataset.color_paths) == len(dataset.depth_paths) == len(params) == 33):
        raise DataError("Fountain requires 33 aligned color/depth/pose keyframes")
    intrinsic = params[0].intrinsic
    k = intrinsic.intrinsic_matrix
    width, height = intrinsic.width, intrinsic.height
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise DataError("Unexpected calibrated Fountain image dimensions")
    full = Camera(width, height, float(k[0, 0]), float(k[1, 1]),
                  float(k[0, 2]), float(k[1, 2]))
    half = full.scaled(2)
    frames, originals = [], {}
    for i, (color_path, depth_path, param) in enumerate(
            zip(dataset.color_paths, dataset.depth_paths, params)):
        with Image.open(color_path) as img:
            color = np.asarray(img.convert("RGB"), dtype=np.uint8)
        with Image.open(depth_path) as img:
            depth = np.asarray(img)
        if color.shape != (height, width, 3) or depth.shape != (height, width):
            raise DataError("Fountain original RGB-D does not match pose intrinsics")
        if depth.dtype.kind not in "ui":
            raise DataError("Expected measured integer sensor depth")
        if i in (0, 16, 32):
            originals[i] = color.copy()
        metric = np.ascontiguousarray(depth[::2, ::2].astype(np.float32) / 1000.)
        metric[(metric < .2) | (metric > 4.)] = 0.
        pose = np.linalg.inv(np.asarray(param.extrinsic, dtype=np.float64))
        frames.append(Frame(t=float(i), camera_to_world=pose, depth=metric,
                            color=np.ascontiguousarray(color[::2, ::2])).validate(half))
    if sum(np.count_nonzero(f.depth) for f in frames) < 100_000:
        raise DataError("No usable metric depth in 33 real keyframes")
    return frames, originals, half, (width, height)


def fuse_prior(measured, supplied, o3d):
    from scipy.spatial import cKDTree
    if len(supplied.triangles) < 1000:
        raise DataError("Official prior has no mesh triangles")
    if len(supplied.triangles) > 250_000:
        supplied = supplied.simplify_quadric_decimation(250_000)
    p = np.asarray(supplied.vertices)
    m = np.asarray(measured.vertices)
    if not len(p) or not len(m):
        raise DataError("Empty measured or existing PRIOR geometry")
    dist, nearest = cKDTree(m).query(p, workers=2)
    # Both are in the provided pose coordinate system. Do not silently ICP-align
    # unrelated shapes or invent a coordinate transform.
    median = float(np.median(dist))
    if median > .50:
        raise DataError("Existing mesh and RGB-D camera poses not aligned: " + str(median))
    if supplied.has_vertex_colors():
        colors = np.asarray(supplied.vertex_colors).copy()
    else:
        colors = np.tile(np.array([.55, .57, .60]), (len(p), 1))
        good = dist < .08
        colors[good] = np.asarray(measured.vertex_colors)[nearest[good]]
    supplied.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0., 1.))
    faces = np.asarray(supplied.triangles)
    missing = np.min(dist[faces], axis=1) > .05
    used, inverse = np.unique(faces[missing], return_inverse=True)
    supplement = o3d.geometry.TriangleMesh()
    supplement.vertices = o3d.utility.Vector3dVector(p[used])
    supplement.triangles = o3d.utility.Vector3iVector(inverse.reshape(-1, 3))
    supplement.vertex_colors = o3d.utility.Vector3dVector(colors[used])
    hybrid = measured + supplement
    provenance = o3d.geometry.TriangleMesh(hybrid)
    origin_colors = np.empty((len(hybrid.vertices), 3))
    origin_colors[:len(measured.vertices)] = [.15, .85, .35]
    origin_colors[len(measured.vertices):] = [.98, .47, .14]
    provenance.vertex_colors = o3d.utility.Vector3dVector(origin_colors)
    return supplied, hybrid, provenance, {
        "measured_triangles": len(measured.triangles),
        "existing_prior_triangles": len(supplied.triangles),
        "supplemental_PRIOR_triangles": int(missing.sum()),
        "hybrid_triangles": len(hybrid.triangles),
        "prior_to_measured_median_distance_m": median,
    }


def experiment(work, output, receipt):
    import open3d as o3d
    work.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    # Downloaded via Open3D's own release assets over verified HTTPS.
    dataset = o3d.data.SampleFountainRGBDImages(data_root=str(work))
    frames, originals, camera, original_size = read_sample(o3d, dataset)
    supplied = o3d.io.read_triangle_mesh(dataset.reconstruction_path)
    measured, _ = fuse(o3d, frames, camera, pixel_stride=1, voxel=.015)
    prior, hybrid, provenance, counts = fuse_prior(measured, supplied, o3d)
    for name, mesh in (
            ("MEASURED-only.glb", measured),
            ("PRIOR-only.glb", prior),
            ("PRIOR-assisted.glb", hybrid),
            ("MEASURED-green-PRIOR-orange.glb", provenance)):
        export_glb(mesh, output / name)
    panels, views = [], []
    for i in (0, 16, 32):
        Image.fromarray(originals[i]).save(output / f"REAL-fountain-{i:02d}.png")
        photo = np.asarray(Image.fromarray(originals[i]).resize(
            (camera.width, camera.height), Image.Resampling.LANCZOS))
        a = render(o3d, measured, camera, frames[i].camera_to_world)
        b = render(o3d, hybrid, camera, frames[i].camera_to_world)
        name = f"recorded-{i:02d}.png"
        panels.append(panel(output / name, [photo, a, b],
                            ["REAL SENSOR RGB", "MEASURED TSDF", "PRIOR ASSISTED"]))
        views.append({"kind": "recorded", "frame": i, "file": name})
    for i, side in ((10, -1.), (23, 1.)):
        pose = free_view(frames[i].camera_to_world, side)
        a = render(o3d, measured, camera, pose)
        b = render(o3d, hybrid, camera, pose)
        name = f"free-{i:02d}.png"
        panels.append(panel(output / name, [np.full_like(a, 20), a, b],
                            ["FREE VIEW / NO PHOTO", "MEASURED TSDF", "PRIOR ASSISTED"]))
        views.append({"kind": "free", "frame": i, "file": name})
    board = Image.new("RGB", (camera.width * 3, (camera.height + 27) * 5))
    for i, view in enumerate(panels):
        board.paste(view, (0, (camera.height + 27) * i))
    board.save(output / "fountain-five-view-MEASURED-vs-PRIOR.png")
    receipt.update(status="real_mesh_exported_visual_review_pending",
                   geometry=counts, views=views,
                   measured_frames=len(frames), source_resolution=original_size,
                   fusion_resolution=[camera.width, camera.height],
                   geometry_source="33 real measured depth frames and supplied keyframe poses",
                   prior_source="Open3D SampleFountainRGBDImages supplied pre-existing mesh",
                   provenance="green=MEASURED, orange=pre-existing PRIOR",
                   free_orbit="FAIL until five-view visual review",
                   issue_74="FAIL unchanged", historical_desk="FAIL 16.7779cm > 15cm unchanged")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    receipt = {"scene": "Real Fountain RGB-D indoor scene",
               "source": "Open3D SampleFountainRGBDImages (official Open3D GitHub release)",
               "status": "running",
               "run_url": (f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/"
                           f"{os.getenv('GITHUB_RUN_ID')}" if os.getenv("GITHUB_RUN_ID") else None)}
    try:
        experiment(args.work, args.output, receipt)
        code = 0
    except Exception as e:
        traceback.print_exc()
        receipt.update(status="failed", error=f"{type(e).__name__}: {e}")
        code = 1
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"status": receipt["status"], "geometry": receipt.get("geometry"),
                          "error": receipt.get("error")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
