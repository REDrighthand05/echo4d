"""Redwood Bedroom: Poisson continuous room proxy from a labeled EXISTING SCAN PRIOR.

Input contains registered PRE-EXISTING scan faces and 600-frame measured
RGB-D details. All Poisson geometry is PRIOR-DERIVED, NOT independently
reconstructed from RGB/video. Orange marks unsupported/inferred proxy areas.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import numpy as np
from PIL import Image


def source_support_mask(distances, densities, *, threshold_m=.05):
    d = np.asarray(distances, dtype=np.float64)
    rho = np.asarray(densities, dtype=np.float64)
    if (d.ndim != 1 or d.shape != rho.shape or not len(d)
            or not np.isfinite(d).all() or not np.isfinite(rho).all()
            or np.any(d < 0) or not 0 < threshold_m <= .25):
        raise ValueError("Invalid Poisson support distances/densities")
    return (d <= threshold_m) & (rho >= np.percentile(rho, 5))


def compute_surface_distances(o3d, source, points, *, chunk=60000):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(source))
    xyz = np.asarray(points, dtype=np.float32)
    out = np.empty(len(xyz), dtype=np.float32)
    for start in range(0, len(xyz), chunk):
        tensor = o3d.core.Tensor(np.ascontiguousarray(xyz[start:start + chunk]))
        out[start:start + chunk] = scene.compute_distance(tensor).numpy()
    del scene
    return out


def run(source_glb: Path, cache: Path, output: Path, *, points=180000, depth=9):
    import open3d as o3d
    from scipy.spatial import cKDTree
    from open3d_official_room_baseline import read_poses
    from redwood_original_prior_ab import fixed_view_specs, render_one_mesh
    from redwood_prior_gapfill import load_measured
    from redwood_prior_measured_hybrid import export_glb
    from redwood_gap_guided_fusion import panel

    if output.exists():
        raise FileExistsError(f"Output exists: {output}")
    if not 50000 <= points <= 250000 or not 7 <= depth <= 10:
        raise ValueError("Unsafe Poisson compute budget")
    source = load_measured(source_glb, o3d)
    if not 450000 <= len(source.triangles) <= 600000:
        raise ValueError("Expected 500k registered PRIOR plus measured detail GLB")
    source.compute_vertex_normals()
    source.compute_triangle_normals()
    o3d.utility.random.seed(29)
    cloud = source.sample_points_uniformly(number_of_points=points,
                                           use_triangle_normal=True)
    normals = np.asarray(cloud.normals)
    if len(normals) != points or not np.isfinite(normals).all() or np.mean(
            np.linalg.norm(normals, axis=1) > .9) < .99:
        raise RuntimeError("Poisson needs valid oriented scan normals")
    print(f"Running prior-derived Poisson: {points} points depth={depth}", flush=True)
    proxy, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=depth, width=0, scale=1.05, linear_fit=False, n_threads=2)
    if len(proxy.triangles) < 10000:
        raise RuntimeError("Poisson did not create a usable room proxy")
    proxy.compute_vertex_normals()
    original_xyz = np.asarray(source.vertices)
    candidate_xyz = np.asarray(proxy.vertices)
    _, nearest = cKDTree(original_xyz).query(candidate_xyz, k=1, workers=2)
    proxy.vertex_colors = o3d.utility.Vector3dVector(
        np.asarray(source.vertex_colors)[nearest].copy())
    del cloud
    gc.collect()

    # Triangle distance rather than vertex distance prevents dense tessellation
    # from being incorrectly tagged as invented geometry.
    triangle_distance = compute_surface_distances(o3d, source, candidate_xyz)
    supported = source_support_mask(triangle_distance, np.asarray(density))
    marked = o3d.geometry.TriangleMesh(proxy)
    marked.vertex_colors = o3d.utility.Vector3dVector(np.where(
        supported[:, None], [.14, .77, .83], [1., .35, .04]))
    output.mkdir(parents=True)
    export_glb(output / "PRIOR-derived-poisson-continuous.glb", proxy)
    export_glb(output / "PRIOR-derived-provenance.glb", marked)

    data = o3d.data.BedroomRGBDImages(data_root=str(cache))
    trajectory = o3d.io.read_pinhole_camera_trajectory(data.trajectory_log_path)
    k, poses = read_poses(trajectory, data.color_paths, data.trajectory_log_path)
    specs = fixed_view_specs(data, poses)
    # Separate raycasting acceleration structures to stay within runner RAM.
    old_render = render_one_mesh(o3d, source, specs, k)
    new_render = render_one_mesh(o3d, proxy, specs, k)
    (output / "visual").mkdir()
    board = Image.new("RGB", (960, 268 * len(specs)), (26, 26, 26))
    views = []
    for i, (spec, old, new) in enumerate(zip(specs, old_render, new_render)):
        photo = None
        if spec["kind"] == "recorded":
            with Image.open(data.color_paths[spec["frame"]]) as image:
                photo = image.convert("RGB").copy()
        label = ("REAL RGB / LABELED PRIOR+MEASURED / PRIOR-DERIVED POISSON"
                 if photo is not None else
                 "FREE VIEW / LABELED PRIOR+MEASURED / PRIOR-DERIVED POISSON")
        path = output / spec["file"]
        panel(path, photo, old[0], new[0], label)
        with Image.open(path) as image:
            board.paste(image.convert("RGB"), (0, 268 * i))
        views.append({"kind": spec["kind"], "frame": spec["frame"],
                      "file": spec["file"], "old_pixels": old[1],
                      "new_pixels": new[1]})
    board.save(output / "five-view-prior-poisson-ab.png")
    report = {
        "scene": "Real Redwood Bedroom",
        "input": "Existing registered 500k scan PRIOR + measured RGB-D detail",
        "input_geometry_role": "PRE-EXISTING PRIOR with measured additions",
        "output_geometry_role": "PRIOR-DERIVED Poisson proxy, inferred areas marked",
        "independent_rgb_only_reconstruction": False,
        "input_faces": len(source.triangles),
        "poisson_faces": len(proxy.triangles),
        "poisson_points": points, "poisson_depth": depth,
        "inferred_proxy_vertices": int((~supported).sum()),
        "supported_proxy_vertices": int(supported.sum()),
        "provenance": "cyan=within 5cm of prior triangles and supported density; orange=inferred/uncertain",
        "color_provenance": "Nearest registered prior/observed vertex, not new camera texture",
        "views": views,
        "visual_acceptance": "PENDING actual human review",
        "free_orbit": "FAIL until verified",
        "issue_74": "FAIL unchanged",
        "old_desk": "FAIL 16.7779cm > 15cm unchanged"
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"faces": len(proxy.triangles),
                      "inferred_vertices": int((~supported).sum()),
                      "views": len(views)}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-glb", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--points", type=int, default=180000)
    parser.add_argument("--depth", type=int, default=9)
    args = parser.parse_args()
    try:
        run(args.source_glb, args.cache, args.output,
            points=args.points, depth=args.depth)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "FAILURE.txt").write_text(
            f"{type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    main()
