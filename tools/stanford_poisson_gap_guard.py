"""Measured Stanford Lounge Mesh: conservative screened-Poisson hole candidates.

This is explicitly an INFERRED local completion experiment, not new sensor
observation, not an independent reconstruction, and not an existing scan prior.
The frozen 480-frame 25 mm observed Mesh is preserved byte-for-byte in geometry.
Poisson candidates are limited to measured boundary neighborhoods and vetoed
when they would occlude measured depth in recorded cameras.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MIN_SOURCE_DISTANCE_M = .035
MAX_SOURCE_DISTANCE_M = .35
MAX_BOUNDARY_DISTANCE_M = .40
MAX_TRIANGLE_EDGE_M = .35
OCCLUSION_MARGIN_M = .05


def boundary_vertices(vertices, triangles):
    """Extract real open-edge vertices, not merely low-density interior points."""
    faces = np.asarray(triangles, dtype=np.int64)
    edges = np.sort(np.concatenate((faces[:, [0, 1]],
                                    faces[:, [1, 2]],
                                    faces[:, [2, 0]])), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    return np.unique(boundary.reshape(-1))


def depth_conflict_mask(points_world, frames, k, depth_images, margin=.05):
    """Reject a patch in FRONT of observed metric depth; never infer free space.

    poses are Open3D world->camera. depth_images are raw millimeters.
    Invalid/missing depth is UNKNOWN, not proof of empty space.
    """
    points = np.asarray(points_world, dtype=np.float64)
    conflict = np.zeros(len(points), dtype=bool)
    width, height, fx, fy, cx, cy = k
    for extrinsic, raw_depth in zip(frames, depth_images):
        if conflict.all():
            break
        remaining = np.flatnonzero(~conflict)
        world = points[remaining]
        camera = world @ extrinsic[:3, :3].T + extrinsic[:3, 3]
        z = camera[:, 2]
        valid = z > .05
        u = np.rint(fx * camera[:, 0] / np.maximum(z, .05) + cx).astype(int)
        v = np.rint(fy * camera[:, 1] / np.maximum(z, .05) + cy).astype(int)
        valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
        idx = np.flatnonzero(valid)
        if not len(idx):
            continue
        measured = np.asarray(raw_depth)[v[idx], u[idx]].astype(float) / 1000.
        contradiction = ((measured > .2) & (measured < 4.) &
                         (z[idx] < measured - margin))
        conflict[remaining[idx[contradiction]]] = True
    return conflict


def run(cache: Path, source: Path, output: Path):
    import open3d as o3d
    import trimesh
    from scipy.spatial import cKDTree
    from stanford_lounge_second_room import lounge_recorded_poses
    from stanford_gap_guided_depth import five_views, to_open3d
    from redwood_prior_first_room import export_colored
    from stanford_sparse_15mm import assert_native_intrinsics
    from PIL import Image, ImageDraw

    if output.exists():
        raise FileExistsError("Output must be a fresh directory")
    source_model = trimesh.load(str(source), force="mesh", process=False)
    if len(source_model.faces) != 454009:
        raise ValueError("Frozen 480-frame observed source has changed")
    before = to_open3d(o3d, source_model)
    before.compute_triangle_normals()
    original_vertices = np.asarray(before.vertices)
    original_faces = np.asarray(before.triangles)
    boundary_ids = boundary_vertices(original_vertices, original_faces)
    if len(boundary_ids) < 100:
        raise ValueError("Expected observed open boundaries")

    print("Sampling observed triangles for Poisson; no existing scan", flush=True)
    cloud = before.sample_points_uniformly(
        number_of_points=140000, use_triangle_normal=True)
    if not cloud.has_normals():
        raise ValueError("Cannot reconstruct without oriented observed normals")
    poisson, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=8, width=0, scale=1.05, linear_fit=False, n_threads=2)
    vertices = np.asarray(poisson.vertices)
    triangles = np.asarray(poisson.triangles)
    densities = np.asarray(density)
    centroids = vertices[triangles].mean(axis=1)

    # No replacement of observed furniture; only boundary-adjacent missing
    # patches. Geometric distance uses the ACTUAL original triangle surface.
    ray = o3d.t.geometry.RaycastingScene()
    ray.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(before))
    dist = ray.compute_distance(
        o3d.core.Tensor(centroids.astype(np.float32))).numpy()
    boundary_dist = cKDTree(original_vertices[boundary_ids]).query(
        centroids, workers=2)[0]
    edge = vertices[triangles]
    max_edges = np.maximum.reduce((
        np.linalg.norm(edge[:, 0] - edge[:, 1], axis=1),
        np.linalg.norm(edge[:, 1] - edge[:, 2], axis=1),
        np.linalg.norm(edge[:, 2] - edge[:, 0], axis=1)))
    bbox = before.get_axis_aligned_bounding_box()
    lo, hi = bbox.get_min_bound() - .03, bbox.get_max_bound() + .03
    candidate = (
        (dist >= MIN_SOURCE_DISTANCE_M) &
        (dist <= MAX_SOURCE_DISTANCE_M) &
        (boundary_dist <= MAX_BOUNDARY_DISTANCE_M) &
        (max_edges <= MAX_TRIANGLE_EDGE_M) &
        np.all((centroids >= lo) & (centroids <= hi), axis=1) &
        np.all(densities[triangles] >= np.quantile(densities, .10), axis=1)
    )
    print(f"Poisson triangles {len(triangles)}, local gap candidates "
          f"{int(candidate.sum())}", flush=True)

    data = o3d.data.LoungeRGBDImages(data_root=str(cache))
    trajectory = o3d.io.read_pinhole_camera_trajectory(data.trajectory_log_path)
    k, poses, mapping = lounge_recorded_poses(
        trajectory, list(data.color_paths), data.trajectory_log_path)
    assert_native_intrinsics(
        k, np.asarray(o3d.io.read_image(data.color_paths[0])).shape,
        np.asarray(o3d.io.read_image(data.depth_paths[0])).shape)
    if len(data.depth_paths) != len(poses):
        raise ValueError("Missing synchronized metric depth / poses")
    # Independent measured-depth occlusion veto across the room timeline.
    chosen = np.linspace(0, len(poses) - 1, 90).round().astype(int)
    veto = depth_conflict_mask(
        centroids[candidate], [poses[int(i)] for i in chosen], k,
        [np.asarray(o3d.io.read_image(data.depth_paths[int(i)]))
         for i in chosen], OCCLUSION_MARGIN_M)
    accepted_faces = triangles[candidate][~veto]
    print(f"Depth-veto rejected {int(veto.sum())}; "
          f"retained {len(accepted_faces)} inferred faces", flush=True)

    output.mkdir(parents=True)
    if len(accepted_faces) < 100:
        (output / "FAILED.txt").write_text(
            f"Insufficient safe local Poisson patches: {len(accepted_faces)}\n")
        raise RuntimeError("No significant geometry; negative result preserved")
    used, remapped = np.unique(accepted_faces.reshape(-1), return_inverse=True)
    patch = o3d.geometry.TriangleMesh()
    patch.vertices = o3d.utility.Vector3dVector(vertices[used])
    patch.triangles = o3d.utility.Vector3iVector(remapped.reshape(-1, 3))
    nearest = cKDTree(original_vertices).query(vertices[used], workers=2)[1]
    patch.vertex_colors = o3d.utility.Vector3dVector(
        np.asarray(before.vertex_colors)[nearest])
    patch.compute_vertex_normals()
    combined = before + patch
    combined.compute_vertex_normals()

    # Actual color is nearest measured color, NOT newly measured texture.
    export_colored(output / "OBSERVED-plus-INFERRED-poisson.glb",
                   np.asarray(combined.vertices),
                   np.asarray(combined.triangles),
                   np.asarray(combined.vertex_colors))
    provenance = np.asarray(combined.vertex_colors).copy()
    provenance[len(original_vertices):] = [0.0, .85, 1.0]
    export_colored(output / "OBSERVED-vs-INFERRED-poisson.glb",
                   np.asarray(combined.vertices),
                   np.asarray(combined.triangles), provenance)
    views = five_views(o3d, data, poses, k, before, combined, output)
    for v in views:
        p = output / v["file"]
        with Image.open(p) as img:
            canvas = img.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, canvas.width, 27), fill=(24, 24, 24))
        draw.text((8, 8),
                  ("REAL PHOTO / OBSERVED 480 / OBSERVED + INFERRED POISSON"
                   if v["kind"] == "recorded" else
                   "FREE VIEW / OBSERVED 480 / OBSERVED + INFERRED POISSON"),
                  fill="white")
        canvas.save(p)
    report = {
        "scene": "Stanford Lounge public Open3D measured RGB-D",
        "input": "native 640x480 measured RGB-D and logged camera poses",
        "camera_mapping": mapping,
        "method": "Open3D Poisson depth 8 on observed mesh normals; boundary and measured-depth occlusion guards",
        "observed_faces_preserved": len(original_faces),
        "poisson_total_faces": len(triangles),
        "candidate_faces_before_veto": int(candidate.sum()),
        "depth_conflict_faces_rejected": int(veto.sum()),
        "inferred_faces": len(accepted_faces),
        "combined_faces": len(np.asarray(combined.triangles)),
        "preexisting_scan_prior_used": False,
        "inferred_geometry_used": True,
        "inferred_colors": "nearest measured observed vertex, not new texture",
        "five_comparison_views": views,
        "visual_acceptance": "NOT REVIEWED; requires actual five-view human inspection",
        "issue_74": "FAIL unchanged",
        "desk_v2": "FAIL 16.7779cm > 15cm unchanged",
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.cache, args.source, args.output)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "FAILED.txt").write_text(
            f"{type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    main()
