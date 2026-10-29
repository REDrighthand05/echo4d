"""Second-scene measured-depth truncation A/B: Redwood Bedroom 3m vs frozen 4.5m.

The baseline is a *previously reconstructed observed-only* 600-frame Mesh,
used exclusively for A/B visualization. It is NOT a geometric prior.
The 3m candidate is independently re-fused from the same original 640x480
RGB-D, logged metric poses, and calibrated intrinsics as the 4.5m baseline.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from open3d_official_room_baseline import read_poses, spaced_real_frames
from redwood_dense_native_review import export_full, raycast
from tum_graph_visual import look_at


EXPECTED_BASELINE_FACES = 553208
VOXEL_M = .025
SDF_TRUNC_M = .1
BASELINE_DEPTH_M = 4.5
CANDIDATE_DEPTH_M = 3.0


def checked_frame_indices(data, trajectory, o3d):
    colors, depths = list(data.color_paths), list(data.depth_paths)
    if len(colors) < 600 or len(colors) != len(depths):
        raise ValueError("Need >=600 aligned real Redwood Bedroom RGB-D pairs")
    if any(Path(a).stem != Path(b).stem for a, b in zip(colors, depths)):
        raise ValueError("Source RGB and measured depth IDs differ")
    k, poses = read_poses(trajectory, colors, data.trajectory_log_path)
    if k[:2] != (640, 480) or len(poses) != len(colors):
        raise ValueError("Must use original 640x480 calibration and logged poses")
    return k, poses, spaced_real_frames(len(colors), 600)


def make_rgbd(o3d, color, depth, k, depth_cutoff_m):
    """Reject invalid calibration/cutoff rather than silently rescale images."""
    if not np.isfinite(depth_cutoff_m) or not 1.0 <= depth_cutoff_m <= 6.0:
        raise ValueError("Invalid depth cutoff")
    if np.asarray(color).shape[:2] != (k[1], k[0]):
        raise ValueError("Original RGB dimensions do not match calibration")
    arr = np.asarray(depth)
    if arr.shape != (k[1], k[0]) or arr.dtype != np.uint16:
        raise ValueError("Expected 16-bit measured millimeter depth at native resolution")
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=1000., depth_trunc=depth_cutoff_m,
        convert_rgb_to_intensity=False)


def fuse(o3d, data, k, poses, indices):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_M, sdf_trunc=SDF_TRUNC_M, volume_unit_resolution=16,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        depth_sampling_stride=1)
    camera = o3d.camera.PinholeCameraIntrinsic(*k)
    far_depth_pixels = 0
    for number, idx in enumerate(indices, 1):
        color = o3d.io.read_image(data.color_paths[idx])
        depth = o3d.io.read_image(data.depth_paths[idx])
        raw = np.asarray(depth)
        far_depth_pixels += int(np.count_nonzero((raw > 3000) & (raw < 4500)))
        frame = make_rgbd(o3d, color, depth, k, CANDIDATE_DEPTH_M)
        volume.integrate(frame, camera, poses[idx])
        if number % 100 == 0:
            print(f"Re-fused {number}/{len(indices)} real RGB-D frames", flush=True)
    mesh = volume.extract_triangle_mesh()
    if not mesh.has_triangles() or not mesh.has_vertex_colors():
        raise RuntimeError("Real measured RGB-D did not produce a colored Mesh")
    mesh.compute_vertex_normals()
    return mesh, far_depth_pixels


def panel(out, photo, baseline, candidate, title):
    parts = ([photo.resize((320, 240))] if photo is not None else []) + [
        Image.fromarray(baseline, "RGB"), Image.fromarray(candidate, "RGB")]
    canvas = Image.new("RGB", (320 * len(parts), 268), (26, 26, 26))
    ImageDraw.Draw(canvas).text((8, 8), title, fill="white")
    for j, part in enumerate(parts):
        canvas.paste(part, (j * 320, 28))
    canvas.save(out)
    return canvas


def compare_five(o3d, data, k, poses, baseline, candidate, out):
    scene_old = o3d.t.geometry.RaycastingScene()
    scene_new = o3d.t.geometry.RaycastingScene()
    scene_old.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(baseline))
    scene_new.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(candidate))
    indices = spaced_real_frames(len(data.color_paths), 360)
    visual = out / "visual"
    visual.mkdir()
    results, panels = [], []
    for idx in (indices[0], indices[len(indices)//2], indices[-1]):
        old, n_old = raycast(o3d, scene_old, baseline, k, poses[idx])
        new, n_new = raycast(o3d, scene_new, candidate, k, poses[idx])
        with Image.open(data.color_paths[idx]) as source:
            photo = source.convert("RGB")
        name = f"recorded-{idx:05d}.png"
        panels.append(panel(visual/name, photo, old, new,
                            "REAL PHOTO / OBSERVED 4.5m / OBSERVED 3.0m"))
        results.append({"type": "recorded", "frame": idx, "file": f"visual/{name}",
                        "baseline_hit_pixels": n_old, "candidate_hit_pixels": n_new})
    for j, (idx, shift) in enumerate(((indices[len(indices)//3], .35),
                                      (indices[2*len(indices)//3], -.35)), 1):
        anchor = np.linalg.inv(poses[idx])
        eye = anchor[:3, 3] + shift * anchor[:3, 0]
        free = look_at(eye, anchor[:3, 3] + 1.8 * anchor[:3, 2])
        camera = np.linalg.inv(free)
        old, n_old = raycast(o3d, scene_old, baseline, k, camera)
        new, n_new = raycast(o3d, scene_new, candidate, k, camera)
        name = f"free-{j}.png"
        panels.append(panel(visual/name, None, old, new,
                            "FREE CAMERA / OBSERVED 4.5m / OBSERVED 3.0m"))
        results.append({"type": "free", "file": f"visual/{name}",
                        "baseline_hit_pixels": n_old, "candidate_hit_pixels": n_new})
    board = Image.new("RGB", (960, 268*len(panels)), (26, 26, 26))
    for i, image in enumerate(panels):
        board.paste(image, (0, i*268))
    board.save(out/"five-view-depth-ab.png")
    return results


def run(source, cache, output):
    import open3d as o3d
    data = o3d.data.BedroomRGBDImages(data_root=str(cache))
    traj = o3d.io.read_pinhole_camera_trajectory(data.trajectory_log_path)
    k, poses, selected = checked_frame_indices(data, traj, o3d)
    baseline = o3d.io.read_triangle_mesh(str(source))
    if (not baseline.has_vertex_colors() or
            len(baseline.triangles) != EXPECTED_BASELINE_FACES):
        raise ValueError("Incorrect frozen Redwood 600-frame 4.5m observed Mesh")
    output.mkdir(parents=True, exist_ok=False)
    candidate, measured_far = fuse(o3d, data, k, poses, selected)
    if not o3d.io.write_triangle_mesh(str(output/"observed-3m-full.ply"), candidate):
        raise RuntimeError("Could not export real 3m Mesh")
    exported = export_full(candidate, output)
    (output/exported["glb"]).rename(output/"observed-3m-full.glb")
    views = compare_five(o3d, data, k, poses, baseline, candidate, output)
    result = {
        "scene": "REAL Redwood Bedroom (second independent room after Stanford Lounge)",
        "input": "600 original 640x480 measured RGB-D frames + recorded poses",
        "dataset": "Open3D BedroomRGBDImages / Redwood",
        "source_4p5m_run": "https://github.com/REDrighthand05/echo4d/actions/runs/40778686280",
        "source_4p5m_mesh_used_only_for_AB": True,
        "preexisting_reference_scan_used": False,
        "baseline_depth_trunc_m": BASELINE_DEPTH_M,
        "candidate_depth_trunc_m": CANDIDATE_DEPTH_M,
        "same_frames_calibration_poses_voxel": True,
        "voxel_m": VOXEL_M, "sdf_trunc_m": SDF_TRUNC_M,
        "baseline_faces": len(baseline.triangles),
        "candidate_faces": len(candidate.triangles),
        "measured_depth_pixels_between_3_and_4p5m": measured_far,
        "candidate_full_color_glb": "observed-3m-full.glb",
        "views": views,
        "human_visual_acceptance": "PENDING",
        "issue_74": "FAIL unchanged",
        "old_desk": "FAIL 16.7779cm > 15cm unchanged",
    }
    (output/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.source_ply, args.cache, args.output)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/"FAILURE.txt").write_text(f"{type(error).__name__}: {error}\n")
        raise


if __name__ == "__main__":
    main()
