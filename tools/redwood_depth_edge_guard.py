"""Redwood measured-depth edge guard before native-resolution Open3D TSDF fusion.

Only original 640x480 registered RGB-D and logged camera poses produce geometry.
The frozen observed mesh is used for A/B rendering, never as a prior.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from open3d_official_room_baseline import read_poses, spaced_real_frames
from redwood_dense_native_review import export_full
from redwood_gap_guided_fusion import compare

def guard_depth_edges(depth, threshold_mm=90):
    """Invalidate both sides of strong depth jumps, leaving other measurements intact."""
    d = np.asarray(depth)
    if d.ndim != 2 or d.dtype != np.uint16:
        raise ValueError("Expected native uint16 millimeter depth")
    if not isinstance(threshold_mm, int) or not 20 <= threshold_mm <= 500:
        raise ValueError("Invalid depth discontinuity threshold")
    valid = (d >= 250) & (d <= 4500)
    discontinuity = np.zeros(d.shape, dtype=bool)
    for axis in (0, 1):
        left = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        right = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        jump = valid[left] & valid[right] & (
            np.abs(d[left].astype(np.int32) - d[right].astype(np.int32)) > threshold_mm
        )
        discontinuity[left] |= jump
        discontinuity[right] |= jump
    cleaned = d.copy()
    cleaned[discontinuity] = 0
    return cleaned, int(discontinuity.sum())

def integrate_edge_guard(o3d, data, indices, poses, k, threshold_mm):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=.025, sdf_trunc=.1, volume_unit_resolution=16,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        depth_sampling_stride=1)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(*k)
    rejected = 0
    for n, index in enumerate(sorted(indices), 1):
        rgb = o3d.io.read_image(data.color_paths[index])
        raw = np.asarray(o3d.io.read_image(data.depth_paths[index]))
        if raw.shape != (480, 640) or np.asarray(rgb).shape[:2] != raw.shape:
            raise ValueError("Native depth/RGB calibration mismatch")
        cleaned, removed = guard_depth_edges(raw, threshold_mm)
        rejected += removed
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, o3d.geometry.Image(np.ascontiguousarray(cleaned)),
            depth_scale=1000., depth_trunc=4.5, convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, poses[index])
        if n % 100 == 0:
            print(f"Measured depth edge-guard fusion {n}/{len(indices)}", flush=True)
    mesh = volume.extract_triangle_mesh()
    if not mesh.has_triangles() or not mesh.has_vertex_colors():
        raise RuntimeError("No real colored Mesh produced")
    mesh.compute_vertex_normals()
    return mesh, rejected

def run(source, cache, output, threshold_mm=90, budget=600):
    import open3d as o3d
    data = o3d.data.BedroomRGBDImages(data_root=str(cache))
    if len(data.color_paths) != len(data.depth_paths) or len(data.color_paths) < budget:
        raise ValueError("Original Redwood RGB-D frames missing")
    if any(Path(c).stem != Path(d).stem for c, d in zip(data.color_paths, data.depth_paths)):
        raise ValueError("Original RGB-D frame IDs do not match")
    traj = o3d.io.read_pinhole_camera_trajectory(data.trajectory_log_path)
    k, poses = read_poses(traj, data.color_paths, data.trajectory_log_path)
    if k[:2] != (640, 480):
        raise ValueError("Expected original 640x480 calibration")
    old = o3d.io.read_triangle_mesh(str(source))
    if not old.has_vertex_colors() or len(old.triangles) != 553208:
        raise ValueError("Frozen 600-frame 4.5m observed-only baseline mismatch")
    selected = spaced_real_frames(len(data.color_paths), budget)
    output.mkdir(parents=True, exist_ok=False)
    new, rejected = integrate_edge_guard(o3d, data, selected, poses, k, threshold_mm)
    if not o3d.io.write_triangle_mesh(str(output / "redwood-edge-guard-observed.ply"), new):
        raise RuntimeError("Cannot save measured-only PLY")
    exported = export_full(new, output)
    (output / exported["glb"]).rename(output / "redwood-edge-guard-observed.glb")
    views = compare(o3d, data, poses, k, old, new, output)
    board = Image.new("RGB", (960, 268 * len(views)), (26, 26, 26))
    for j, entry in enumerate(views):
        path = output / entry["file"]
        with Image.open(path) as image:
            frame = image.convert("RGB").copy()
        ImageDraw.Draw(frame).rectangle((0, 0, frame.width, 27), fill=(26, 26, 26))
        title = "REAL RGB / UNFILTERED / DEPTH EDGE GUARD" if entry["kind"] == "recorded" else "FREE VIEW / UNFILTERED / DEPTH EDGE GUARD"
        ImageDraw.Draw(frame).text((8, 8), title, fill="white")
        frame.save(path)
        board.paste(frame, (0, 268 * j))
    board.save(output / "five-view-depth-edge-guard-ab.png")
    report = {
        "scene": "Real Redwood Bedroom",
        "input": "Original 640x480 registered RGB-D, measured depth, logged metric camera poses",
        "geometry_change": "Invalidate measured depth pixels adjacent to >90mm discontinuities before 2.5cm TSDF fusion",
        "threshold_mm": threshold_mm,
        "frames": len(selected),
        "rejected_depth_pixels": rejected,
        "original_faces": len(old.triangles),
        "new_faces": len(new.triangles),
        "reference_scan_used": False,
        "old_mesh_used_for": "visual A/B only",
        "visual_acceptance": "PENDING independent human review",
        "free_orbit": "FAIL until verified",
        "issue_74": "FAIL unchanged",
        "old_desk": "FAIL 16.7779cm > 15cm unchanged",
        "views": views,
        "glb": "redwood-edge-guard-observed.glb"
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-ply", required=True, type=Path)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--threshold-mm", type=int, default=90)
    p.add_argument("--budget", type=int, default=600)
    a = p.parse_args()
    if not 3 <= a.budget <= 600:
        p.error("budget must be 3..600")
    try:
        run(a.source_ply, a.cache, a.output, a.threshold_mm, a.budget)
    except Exception as exc:
        a.output.mkdir(parents=True, exist_ok=True)
        (a.output / "FAILURE.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise

if __name__ == "__main__":
    main()
