"""Cross-view measured-depth consistency before Redwood Bedroom TSDF fusion.

Reject only pixels independently contradicted by measured frames on BOTH
temporal sides. A nearer depth in another view means occlusion, not disproof.
Neither the source mesh nor the official reference scan is fused as a prior.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import minimum_filter, maximum_filter, convolve

from open3d_official_room_baseline import read_poses, spaced_real_frames
from redwood_dense_native_review import export_full
from redwood_gap_guided_fusion import compare


def depth_patch(depth):
    d = np.asarray(depth)
    if d.ndim != 2 or d.dtype != np.uint16:
        raise ValueError("Expected native uint16 depth in millimeters")
    valid = (d >= 250) & (d <= 4500)
    metres = d.astype(np.float32) / 1000.0
    near = minimum_filter(np.where(valid, metres, np.inf), size=3, mode="constant", cval=np.inf)
    far = maximum_filter(np.where(valid, metres, -np.inf), size=3, mode="constant", cval=-np.inf)
    count = convolve(valid.astype(np.uint8), np.ones((3, 3), dtype=np.uint8),
                     mode="constant", cval=0)
    return near, far, count


def compare_depths(source, source_world_to_camera, target, target_world_to_camera,
                   intrinsics, tolerance_m=.08):
    """Return (supported, contradicted) boolean masks at source resolution.

    A target ray that hits a *closer* surface is occluded, not contradictory.
    Only a patch of >=5 valid pixels ALL farther than the projected source
    surface can contradict it. Unobserved pixels cannot contradict geometry.
    """
    src = np.asarray(source)
    dst = np.asarray(target)
    w, h, fx, fy, cx, cy = intrinsics
    if src.shape != (h, w) or dst.shape != src.shape or src.dtype != np.uint16 or dst.dtype != np.uint16:
        raise ValueError("Depth frames must match the calibrated uint16 image dimensions")
    if not np.isfinite(tolerance_m) or not .02 <= tolerance_m <= .2:
        raise ValueError("Tolerance must be 2-20cm")
    a, b = np.asarray(source_world_to_camera), np.asarray(target_world_to_camera)
    if a.shape != (4, 4) or b.shape != (4, 4) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid measured camera extrinsics")
    valid = (src >= 250) & (src <= 4500)
    y, x = np.nonzero(valid)
    support = np.zeros(src.shape, dtype=bool)
    contradict = np.zeros(src.shape, dtype=bool)
    if not len(x):
        return support, contradict
    z = src[y, x].astype(np.float64) / 1000.0
    xyz = np.column_stack(((x - cx) * z / fx, (y - cy) * z / fy, z))
    relative = b @ np.linalg.inv(a)
    other = xyz @ relative[:3, :3].T + relative[:3, 3]
    zz = other[:, 2]
    front = zz > .15
    uu = np.zeros(len(x), dtype=np.int32)
    vv = np.zeros(len(x), dtype=np.int32)
    uu[front] = np.rint(fx * other[front, 0] / zz[front] + cx).astype(np.int32)
    vv[front] = np.rint(fy * other[front, 1] / zz[front] + cy).astype(np.int32)
    inside = front & (uu >= 1) & (vv >= 1) & (uu < w - 1) & (vv < h - 1)
    if not inside.any():
        return support, contradict
    near, far, count = depth_patch(dst)
    q = np.nonzero(inside)[0]
    near_z, far_z, n = near[vv[q], uu[q]], far[vv[q], uu[q]], count[vv[q], uu[q]]
    enough = n >= 5
    # The range test is conservative at silhouettes: a plausible depth
    # anywhere in the patch prevents a contradiction.
    supported = enough & (zz[q] >= near_z - tolerance_m) & (zz[q] <= far_z + tolerance_m)
    farther = enough & (near_z > zz[q] + tolerance_m)
    support[y[q[supported]], x[q[supported]]] = True
    contradict[y[q[farther]], x[q[farther]]] = True
    return support, contradict


def multiview_guard(source, pose, earlier, later, intrinsics, tolerance_m=.08):
    """Earlier/later contain (uint16 depth, world-to-camera pose) pairs.

    Require at least one contradicting view on EACH side and no supporting
    view anywhere; temporal endpoints are preserved, not guessed.
    """
    depth = np.asarray(source)
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("Expected measured uint16 depth")
    if not earlier or not later:
        return depth.copy(), 0
    supported = np.zeros(depth.shape, dtype=bool)
    before = np.zeros(depth.shape, dtype=bool)
    after = np.zeros(depth.shape, dtype=bool)
    for group, contradictions in ((earlier, before), (later, after)):
        for other_depth, other_pose in group:
            yes, no = compare_depths(depth, pose, other_depth, other_pose, intrinsics, tolerance_m)
            supported |= yes
            contradictions |= no
    rejected = before & after & ~supported
    cleaned = depth.copy()
    cleaned[rejected] = 0
    return cleaned, int(rejected.sum())


def run(source_ply, cache, output, budget=600, tolerance_m=.08):
    import open3d as o3d
    data = o3d.data.BedroomRGBDImages(data_root=str(cache))
    if len(data.color_paths) != len(data.depth_paths) or len(data.color_paths) < budget:
        raise ValueError("Missing actual registered Redwood RGB-D")
    if any(Path(c).stem != Path(d).stem for c, d in zip(data.color_paths, data.depth_paths)):
        raise ValueError("Mismatched RGB-D frame IDs")
    trajectory = o3d.io.read_pinhole_camera_trajectory(data.trajectory_log_path)
    k, poses = read_poses(trajectory, data.color_paths, data.trajectory_log_path)
    if k[:2] != (640, 480):
        raise ValueError("Expected native Redwood 640x480 calibration")
    old = o3d.io.read_triangle_mesh(str(source_ply))
    if not old.has_triangles() or not old.has_vertex_colors():
        raise ValueError("Missing actual measured-only baseline mesh")
    selected = spaced_real_frames(len(data.color_paths), budget)
    output.mkdir(parents=True, exist_ok=False)

    @lru_cache(maxsize=12)
    def depth_at(index):
        d = np.asarray(o3d.io.read_image(data.depth_paths[index]))
        if d.shape != (480, 640) or d.dtype != np.uint16:
            raise ValueError("Invalid native Redwood depth")
        return d

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=.025, sdf_trunc=.1, volume_unit_resolution=16,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        depth_sampling_stride=1)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(*k)
    total_rejected = 0
    changed_frames = 0
    for position, index in enumerate(selected):
        earlier = [(depth_at(selected[j]), poses[selected[j]])
                   for j in range(max(0, position - 2), position)]
        later = [(depth_at(selected[j]), poses[selected[j]])
                 for j in range(position + 1, min(len(selected), position + 3))]
        cleaned, removed = multiview_guard(
            depth_at(index), poses[index], earlier, later, k, tolerance_m)
        total_rejected += removed
        changed_frames += bool(removed)
        color = o3d.io.read_image(data.color_paths[index])
        if np.asarray(color).shape[:2] != cleaned.shape:
            raise ValueError("Color/depth calibration mismatch")
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, o3d.geometry.Image(np.ascontiguousarray(cleaned)),
            depth_scale=1000., depth_trunc=4.5, convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, poses[index])
        if (position + 1) % 100 == 0:
            print(f"Measured cross-view fusion {position + 1}/{len(selected)}, rejected={total_rejected}", flush=True)

    mesh = volume.extract_triangle_mesh()
    if not mesh.has_triangles() or not mesh.has_vertex_colors():
        raise RuntimeError("No real colored mesh produced")
    mesh.compute_vertex_normals()
    ply = output / "redwood-multiview-observed.ply"
    if not o3d.io.write_triangle_mesh(str(ply), mesh):
        raise RuntimeError("Could not export measured-only PLY")
    exported = export_full(mesh, output)
    glb = "redwood-multiview-observed.glb"
    (output / exported["glb"]).rename(output / glb)
    views = compare(o3d, data, poses, k, old, mesh, output)
    board = Image.new("RGB", (960, 268 * len(views)), (26, 26, 26))
    for i, entry in enumerate(views):
        path = output / entry["file"]
        with Image.open(path) as f:
            panel = f.convert("RGB").copy()
        ImageDraw.Draw(panel).rectangle((0, 0, panel.width, 27), fill=(26, 26, 26))
        ImageDraw.Draw(panel).text((8, 8), "REAL RGB / PREVIOUS / MULTIVIEW" if entry["kind"] == "recorded"
                                   else "FREE CAMERA / PREVIOUS / MULTIVIEW", fill="white")
        panel.save(path)
        board.paste(panel, (0, 268 * i))
    board.save(output / "five-view-multiview-ab.png")
    report = {
        "scene": "Real Redwood Bedroom",
        "input": "600 original 640x480 measured RGB-D frames and logged metric poses",
        "geometry": "2.5cm Open3D TSDF after conservative two-sided cross-view depth rejection",
        "frames": len(selected), "changed_frames": changed_frames,
        "rejected_pixels": total_rejected, "tolerance_m": tolerance_m,
        "previous_faces": len(old.triangles), "new_faces": len(mesh.triangles),
        "prior_scan_used": False, "previous_mesh_used_only_for_visual_comparison": True,
        "glb": glb, "views": views,
        "visual_acceptance": "PENDING actual visual inspection",
        "free_orbit": "FAIL until verified", "issue_74": "FAIL unchanged",
        "old_desk": "FAIL 16.7779cm > 15cm unchanged"
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-ply", required=True, type=Path)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--budget", type=int, default=600)
    p.add_argument("--tolerance-m", type=float, default=.08)
    a = p.parse_args()
    if not 3 <= a.budget <= 600:
        p.error("budget must be 3..600")
    try:
        run(a.source_ply, a.cache, a.output, a.budget, a.tolerance_m)
    except Exception as e:
        a.output.mkdir(parents=True, exist_ok=True)
        (a.output / "FAILURE.txt").write_text(f"{type(e).__name__}: {e}\n")
        raise


if __name__ == "__main__":
    main()
