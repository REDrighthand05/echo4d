"""Full 3000-frame Stanford Lounge TSDF re-fusion using previously clipped measured far depth.

Single integration change from pinned official Open3D: depth_trunc 3.0m -> 4.5m.
Same original 640x480 calibration, 3000 estimated poses, voxel 3/512m,
SDF truncation 0.04m. Original observed mesh is visual A/B only, NOT a prior.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import subprocess
import numpy as np
from PIL import Image, ImageDraw

UPSTREAM_SHA = "1e7b17438687a0b0c1e5a7187321ac7044afe275"
BASELINE_DEPTH_M = 3.0
NEW_DEPTH_M = 4.5
VOXEL_M = 3.0 / 512.0
SDF_TRUNC_M = 0.04

def run(source: Path, upstream: Path, output: Path):
    import open3d as o3d
    from stanford_full_observed_speck_cleanup import (
        load_poses, camera_rays, render, export_glb,
    )
    source, upstream, output = (p.resolve() for p in (source, upstream, output))
    if output.exists():
        raise FileExistsError("Fresh output directory required")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream,
                               text=True).strip() != UPSTREAM_SHA:
        raise ValueError("Pinned Open3D source changed")
    output.mkdir(parents=True)
    sys.path.insert(0, str(upstream / "examples/python"))
    from open3d_example import get_rgbd_file_lists
    data = o3d.data.LoungeRGBDImages()
    color_paths, depth_paths = get_rgbd_file_lists(data.extract_dir)
    poses = load_poses(source / "scene/trajectory.log")
    if len(poses) != 3000 or len(color_paths) != 3000 or len(depth_paths) != 3000:
        raise ValueError("Full original 3000-frame RGB-D/pose alignment required")
    if any(Path(c).stem != Path(d).stem for c, d in zip(color_paths, depth_paths)):
        raise ValueError("RGB/depth frame identity mismatch")

    original = o3d.io.read_triangle_mesh(str(source / "scene/integrated.ply"))
    if not original.has_triangles() or not original.has_vertex_colors():
        raise ValueError("Missing actual observed baseline")
    views = camera_rays(o3d, original, poses)
    before, before_cov = render(o3d, original, views)
    original_faces = len(original.triangles)
    print("BASELINE", original_faces, "faces; starting full TSDF", flush=True)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    if intrinsic.width != 640 or intrinsic.height != 480:
        raise ValueError("Unexpected camera calibration resolution")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_M, sdf_trunc=SDF_TRUNC_M,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    n_far, n_near = 0, 0
    start = time.monotonic()
    for i, (c, d, pose) in enumerate(zip(color_paths, depth_paths, poses)):
        color = o3d.io.read_image(str(c))
        depth = o3d.io.read_image(str(d))
        if i == 0 and (np.asarray(depth).shape != (480, 640) or
                       np.asarray(color).shape[:2] != (480, 640)):
            raise ValueError("Original RGB-D is not 640x480")
        z = np.asarray(depth)
        n_near += int(((z >= 300) & (z <= 3000)).sum())
        n_far += int(((z > 3000) & (z <= 4500)).sum())
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000., depth_trunc=NEW_DEPTH_M,
            convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
        if (i + 1) % 250 == 0:
            print(f"INTEGRATED {i+1}/3000 far pixels {n_far:,} "
                  f"elapsed {time.monotonic()-start:.1f}s", flush=True)
    if n_far == 0:
        raise RuntimeError("No measured depth beyond 3m; no rationale for candidate")
    candidate = volume.extract_triangle_mesh()
    candidate.compute_vertex_normals()
    if not candidate.has_triangles() or not candidate.has_vertex_colors():
        raise RuntimeError("No valid colored candidate mesh")
    after, after_cov = render(o3d, candidate, views)
    export_glb(candidate, output / "observed-3000-far-depth-45m.glb")
    preview = (candidate.simplify_quadric_decimation(300000)
               if len(candidate.triangles) > 300000 else candidate)
    export_glb(preview, output / "observed-3000-far-depth-45m-preview.glb")
    del preview
    gc.collect()

    canvas = Image.new("RGB", (960, 34 + 5 * 268), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 10), "REAL RGB / free camera", fill="white")
    draw.text((325, 10), "ORIGINAL depth <=3m", fill="white")
    draw.text((645, 10), "NEW measured depth <=4.5m", fill="white")
    for j, (name, _) in enumerate(views):
        y = 34 + j * 268
        draw.text((5, y + 5), name, fill="white")
        if j < 3:
            with Image.open(source / "views" / name) as im:
                if im.size != (640, 268):
                    raise ValueError(f"Unexpected original photo panel: {name}")
                canvas.paste(im.convert("RGB").crop((0, 28, 320, 268)), (0, y + 28))
        canvas.paste(before[name], (320, y + 28))
        canvas.paste(after[name], (640, y + 28))
    canvas.save(output / "five-view-far-depth-ab.png")
    receipt = {
        "scene": "Stanford Lounge, 3000 original 640x480 measured RGB-D frames",
        "source_run": "https://github.com/REDrighthand05/echo4d/actions/runs/40877823227",
        "pose_source": "frozen Open3D estimated 3000-frame camera poses",
        "existing_mesh_prior": False, "predicted_depth": False,
        "inferred_geometry": False,
        "single_changed_parameter": "depth_trunc: 3.0m -> 4.5m",
        "voxel_m_unchanged": VOXEL_M, "sdf_trunc_m_unchanged": SDF_TRUNC_M,
        "newly_eligible_measured_far_depth_pixels": n_far,
        "eligible_baseline_depth_pixels": n_near,
        "baseline_faces": original_faces,
        "candidate_faces": len(candidate.triangles),
        "coverage_before": before_cov, "coverage_after": after_cov,
        "elapsed_seconds": time.monotonic() - start,
        "visual_acceptance": "NOT REVIEWED; #74 remains FAIL",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--upstream", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    run(args.source, args.upstream, args.output)
