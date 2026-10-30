"""Show the COMPLETE pre-existing Fountain scan PRIOR, colorized from 33 real RGB-D keyframes.

Geometry of PRIOR is supplied by Open3D, not recovered by Echo4D.
Open3D's mature rigid color-map optimization projects the original sensor RGB
onto the supplied scan. All five comparisons use identical camera poses.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import traceback

import numpy as np
from PIL import Image
from echo4d.model import DataError
from tum_xyz_open3d_320 import fuse, export_glb, render, free_view, panel
from fountain_measured_prior_fusion import read_sample


def experiment(work, output, receipt):
    import open3d as o3d
    work.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    dataset = o3d.data.SampleFountainRGBDImages(data_root=str(work))
    frames, originals, camera, original_size = read_sample(o3d, dataset)
    supplied = o3d.io.read_triangle_mesh(dataset.reconstruction_path)
    if len(supplied.triangles) < 1000:
        raise DataError("No real pre-existing mesh")
    if len(supplied.triangles) > 250_000:
        supplied = supplied.simplify_quadric_decimation(250_000)
    rgbds = []
    for color_path, depth_path in zip(dataset.color_paths, dataset.depth_paths):
        color = o3d.io.read_image(color_path)
        depth = o3d.io.read_image(depth_path)
        rgbds.append(o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000., depth_trunc=4.,
            convert_rgb_to_intensity=False))
    trajectory = o3d.io.read_pinhole_camera_trajectory(dataset.keyframe_poses_log_path)
    measured, _ = fuse(o3d, frames, camera, pixel_stride=1, voxel=.015)
    print("Real measured-only Mesh complete", len(measured.triangles), flush=True)
    # The 0-iteration reference projects original RGB with original poses.
    prior0, _ = o3d.pipelines.color_map.run_rigid_optimizer(
        o3d.geometry.TriangleMesh(supplied), rgbds, trajectory,
        o3d.pipelines.color_map.RigidOptimizerOption(maximum_iteration=0))
    print("PRIOR original-pose color projection complete", flush=True)
    trajectory = o3d.io.read_pinhole_camera_trajectory(dataset.keyframe_poses_log_path)
    prior15, _ = o3d.pipelines.color_map.run_rigid_optimizer(
        o3d.geometry.TriangleMesh(supplied), rgbds, trajectory,
        o3d.pipelines.color_map.RigidOptimizerOption(maximum_iteration=15))
    print("PRIOR 15-iteration color projection complete", flush=True)
    if not prior0.has_vertex_colors() or not prior15.has_vertex_colors():
        raise DataError("Open3D prior color-map optimization returned no RGB")
    for name, mesh in (("MEASURED-only.glb", measured),
                       ("FULL-PRIOR-color-0iter.glb", prior0),
                       ("FULL-PRIOR-color-15iter.glb", prior15)):
        export_glb(mesh, output / name)
    views, boards = [], []
    for i in (0, 16, 32):
        photo = Image.fromarray(originals[i])
        photo.save(output / f"REAL-fountain-{i:02d}.png")
        real = np.asarray(photo.resize((camera.width, camera.height), Image.Resampling.LANCZOS))
        pose = frames[i].camera_to_world
        columns = [real, render(o3d, measured, camera, pose),
                   render(o3d, prior0, camera, pose),
                   render(o3d, prior15, camera, pose)]
        name = f"recorded-{i:02d}.png"
        boards.append(panel(output / name, columns,
                            ["REAL RGB", "MEASURED TSDF", "FULL PRIOR / 0", "FULL PRIOR / 15"]))
        views.append({"kind": "recorded", "frame": i, "file": name})
    for i, side in ((10, -1.), (23, 1.)):
        pose = free_view(frames[i].camera_to_world, side)
        a = render(o3d, measured, camera, pose)
        b = render(o3d, prior0, camera, pose)
        c = render(o3d, prior15, camera, pose)
        name = f"free-{i:02d}.png"
        boards.append(panel(output / name, [np.full_like(a, 20), a, b, c],
                            ["FREE / NO PHOTO", "MEASURED TSDF", "FULL PRIOR / 0", "FULL PRIOR / 15"]))
        views.append({"kind": "free", "frame": i, "file": name})
    board = Image.new("RGB", (camera.width * 4, (camera.height + 27) * 5))
    for i, image in enumerate(boards):
        board.paste(image, (0, (camera.height + 27) * i))
    board.save(output / "fountain-five-view-full-PRIOR-color-map.png")
    receipt.update(status="real_five_view_exported_review_pending",
                   measured_frames=len(frames), source_resolution=original_size,
                   measured_triangles=len(measured.triangles),
                   FULL_PRIOR_triangles=len(prior15.triangles),
                   prior_role="Pre-existing Open3D supplied reconstruction, never independent recovery",
                   comparison="Measured-only TSDF vs full PRIOR with 0 and 15 iterations of RGB projection",
                   views=views, issue_74="FAIL unchanged",
                   old_desk="FAIL 16.7779cm > 15cm unchanged")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    receipt = {"scene": "Real Fountain indoor RGB-D", "status": "running",
               "source": "Open3D SampleFountainRGBDImages (33 real sensor frames and supplied scan)",
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
        print(json.dumps({"status": receipt["status"], "error": receipt.get("error"),
                          "measured_triangles": receipt.get("measured_triangles"),
                          "FULL_PRIOR_triangles": receipt.get("FULL_PRIOR_triangles")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
