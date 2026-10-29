"""Measured-only Stanford Lounge 25mm TSDF: reject far-side depth-edge bridges.

One isolated geometry intervention against the frozen 480-frame 25mm mesh:
zero ONLY farther depth pixels next to a substantially nearer valid pixel.
Foreground pixels, colors, poses, frame IDs, intrinsics and voxel size stay fixed.
No supplied scan or inferred geometry enters fusion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stanford_sparse_15mm import (
    EXPECTED_BASELINE_FACES, assert_native_intrinsics, frozen_indices,
)


def mask_far_side_depth_edges(
    raw: np.ndarray,
    *,
    absolute_mm: int = 100,
    relative: float = 0.035,
    min_depth_mm: int = 350,
    max_depth_mm: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove background-side mixed depth at 4-connected occlusion edges.

    Only a valid farther pixel can be removed. A valid nearer neighbor must
    differ by more than max(100mm, 3.5% of far depth). Invalid neighbors do
    not create artificial edges. No dilation or depth interpolation is used.
    """
    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise ValueError("Expected native uint16 millimeter depth image")
    if (not 0 < absolute_mm < 2000 or not 0 <= relative < .25
            or not 0 < min_depth_mm < max_depth_mm <= 65535):
        raise ValueError("Invalid metric depth edge parameters")
    z = raw.astype(np.int32)
    valid = (z >= min_depth_mm) & (z < max_depth_mm)
    far_edge = np.zeros(z.shape, dtype=bool)
    limit = np.maximum(absolute_mm, relative * z)
    # Shifted pair views avoid wrapping the opposite image border.
    for dy, dx in ((1, 0), (0, 1)):
        if dy:
            a, b = (slice(1, None), slice(None)), (slice(None, -1), slice(None))
        else:
            a, b = (slice(None), slice(1, None)), (slice(None), slice(None, -1))
        va, vb = valid[a], valid[b]
        pair = va & vb
        za, zb = z[a], z[b]
        far_edge[a] |= pair & (za > zb) & ((za - zb) > limit[a])
        far_edge[b] |= pair & (zb > za) & ((zb - za) > limit[b])
    filtered = raw.copy()
    filtered[far_edge] = 0
    return filtered, far_edge


def run(cache: Path, source: Path, source_report: Path, output: Path):
    import open3d as o3d
    import trimesh

    from open3d_official_room_baseline import spaced_real_frames
    from stanford_lounge_second_room import lounge_recorded_poses
    from stanford_gap_guided_depth import five_views, to_open3d
    from redwood_prior_first_room import export_colored

    if output.exists():
        raise FileExistsError("Use a new output directory")
    frozen = json.loads(source_report.read_text())
    if (frozen.get("after_observed_faces") != EXPECTED_BASELINE_FACES
            or frozen.get("existing_prior_geometry_used") is not False
            or frozen.get("new_frames") != 120):
        raise ValueError("Frozen Stanford 480-frame source changed")
    data = o3d.data.LoungeRGBDImages(data_root=str(cache))
    if (len(data.color_paths) != len(data.depth_paths)
            or any(Path(c).stem != Path(d).stem for c, d in
                   zip(data.color_paths, data.depth_paths))):
        raise ValueError("Stanford RGB/depth synchronization failed")
    trajectory = o3d.io.read_pinhole_camera_trajectory(data.trajectory_log_path)
    k, poses, mapping = lounge_recorded_poses(
        trajectory, list(data.color_paths), data.trajectory_log_path)
    if len(poses) != len(data.color_paths):
        raise ValueError("Missing recorded poses")
    indices = frozen_indices(
        len(data.color_paths),
        spaced_real_frames(len(data.color_paths), 360),
        frozen["selected_frame_indices"])

    before_model = trimesh.load(str(source), force="mesh", process=False)
    if len(before_model.faces) != EXPECTED_BASELINE_FACES:
        raise ValueError("Frozen 25mm source mesh changed")
    before = to_open3d(o3d, before_model)

    engine = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=.025, sdf_trunc=.1, volume_unit_resolution=16,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        depth_sampling_stride=1)
    camera = o3d.camera.PinholeCameraIntrinsic(*k)
    counts = []
    for i in indices:
        color = o3d.io.read_image(data.color_paths[i])
        depth = o3d.io.read_image(data.depth_paths[i])
        rgb = np.asarray(color)
        raw = np.asarray(depth)
        assert_native_intrinsics(k, rgb.shape, raw.shape)
        filtered, mask = mask_far_side_depth_edges(raw)
        valid = (raw >= 350) & (raw < 4000)
        counts.append({
            "frame": int(i),
            "valid_depth_pixels": int(valid.sum()),
            "far_side_pixels_removed": int(mask.sum()),
        })
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, o3d.geometry.Image(np.ascontiguousarray(filtered)),
            depth_scale=1000., depth_trunc=4.,
            convert_rgb_to_intensity=False)
        engine.integrate(rgbd, camera, poses[i])
        if len(counts) % 60 == 0:
            print(f"Fused {len(counts)}/{len(indices)} recorded frames", flush=True)

    valid_total = sum(x["valid_depth_pixels"] for x in counts)
    removed_total = sum(x["far_side_pixels_removed"] for x in counts)
    fraction = removed_total / max(valid_total, 1)
    if not 0 < fraction < .12:
        raise RuntimeError(f"Depth-edge mask removed implausible fraction: {fraction:.4%}")
    after = engine.extract_triangle_mesh()
    if not after.has_triangles() or not after.has_vertex_colors():
        raise RuntimeError("No measured colored TSDF geometry")
    after.compute_vertex_normals()
    vertices = np.asarray(after.vertices)
    triangles = np.asarray(after.triangles)
    colors = np.asarray(after.vertex_colors)
    if len(triangles) < 10000 or not np.isfinite(vertices).all():
        raise RuntimeError("Invalid filtered mesh")
    output.mkdir(parents=True)
    if not o3d.io.write_triangle_mesh(
            str(output / "OBSERVED-25mm-edge-filtered-native.ply"), after):
        raise RuntimeError("Native PLY export failed")
    export_colored(output / "OBSERVED-25mm-edge-filtered-full.glb",
                   vertices, triangles, colors)
    views = five_views(o3d, data, poses, k, before, after, output)
    from PIL import Image, ImageDraw
    for view in views:
        path = output / view["file"]
        with Image.open(path) as im:
            canvas = im.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, canvas.width, 27), fill=(24, 24, 24))
        draw.text((8, 8),
                  ("REAL PHOTO / 480 FRAMES 25mm / EDGE-FILTERED 25mm"
                   if view["kind"] == "recorded" else
                   "FREE CAMERA / 480 FRAMES 25mm / EDGE-FILTERED 25mm"),
                  fill="white")
        canvas.save(path)
    report = {
        "scene": "Stanford Lounge, public real 640x480 RGB-D",
        "input": "480 identical measured frames and recorded poses",
        "recorded_pose_mapping": mapping,
        "intrinsics_640x480": list(k),
        "voxel_m_both": .025,
        "truncation_m_both": .1,
        "frozen_frame_count": len(indices),
        "frozen_before_observed_faces": len(before.triangles),
        "filtered_after_observed_faces": len(triangles),
        "depth_filter": "remove farther of 4-connected valid depth pair if jump > max(100mm,3.5% far depth); no dilation",
        "valid_depth_pixels": valid_total,
        "removed_far_side_depth_pixels": removed_total,
        "removed_fraction": fraction,
        "frame_filter_counts": counts,
        "prior_mesh_used": False,
        "inferred_geometry_used": False,
        "native_colored_glb": "OBSERVED-25mm-edge-filtered-full.glb",
        "three_recorded_two_free": views,
        "visual_acceptance": "NOT REVIEWED; inspect images before claiming improvement",
        "issue_74": "FAIL unchanged",
        "desk_v2": "FAIL 16.7779cm > 15cm unchanged",
    }
    (output / "result.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        k: v for k, v in report.items() if k != "frame_filter_counts"
    }, ensure_ascii=False), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--source-report", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    try:
        run(args.cache, args.source, args.source_report, args.output)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "FAILED.txt").write_text(
            f"{type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    main()
