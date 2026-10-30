"""Real Redwood: retain original-resolution registered scan PRIOR vs 500k decimation.

Both candidates contain identical RGB-D-only detail. Scan geometry is explicitly
PRE-EXISTING PRIOR, not independently reconstructed from the 600-frame video.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def transfer_colors_chunked(prior, measured, max_distance=.09, chunk=50000):
    """Color original multi-million-vertex prior without an enormous Nx8 query."""
    from scipy.spatial import cKDTree
    p = np.asarray(prior.vertices)
    pn = np.asarray(prior.vertex_normals)
    q = np.asarray(measured.vertices)
    qn = np.asarray(measured.vertex_normals)
    rgb = np.asarray(measured.vertex_colors)
    if (not len(p) or not len(q) or pn.shape != p.shape
            or qn.shape != q.shape or rgb.shape != q.shape):
        raise ValueError("Missing aligned prior or observed colors/normals")
    tree = cKDTree(q)
    result = np.full((len(p), 3), (.53, .56, .59), dtype=np.float64)
    supported = np.zeros(len(p), dtype=bool)
    for start in range(0, len(p), chunk):
        stop = min(start + chunk, len(p))
        distance, ids = tree.query(p[start:stop], k=min(4, len(q)), workers=2)
        if distance.ndim == 1:
            distance, ids = distance[:, None], ids[:, None]
        dot = np.einsum("ij,ikj->ik", pn[start:stop], qn[ids])
        good = (distance <= max_distance) & (dot >= .35)
        closest = np.where(good, distance, np.inf)
        selected = np.argmin(closest, axis=1)
        rows = np.arange(stop - start)
        mask = np.isfinite(closest[rows, selected])
        result[start:stop][mask] = rgb[ids[rows[mask], selected[mask]]]
        supported[start:stop] = mask
    return result, supported



def refine_registered_prior(prior, measured, initial, o3d):
    """Conservative point-to-plane refinement of the supplied scan PRIOR.

    Refine a coarse rigid registration against actual measured RGB-D geometry.
    Never deform the source mesh, and reject corrections without improved
    geometric agreement on the same sampled point clouds.
    """
    from redwood_prior_gapfill import as_cloud
    reg = o3d.pipelines.registration
    o3d.utility.random.seed(21)
    source, target = as_cloud(prior, o3d), as_cloud(measured, o3d)
    start = np.asarray(initial, dtype=np.float64)
    baseline = reg.evaluate_registration(source, target, .12, start)
    try:
        estimate = reg.TransformationEstimationPointToPlane(
            reg.TukeyLoss(k=.08))
        fitted = reg.registration_icp(
            source, target, .12, start, estimate,
            reg.ICPConvergenceCriteria(max_iteration=35))
    except (RuntimeError, ValueError):
        return start, {"accepted": False, "reason": "robust ICP unavailable"}
    candidate = np.asarray(fitted.transformation, dtype=np.float64)
    delta = candidate @ np.linalg.inv(start)
    translation = float(np.linalg.norm(delta[:3, 3]))
    angle = float(np.rad2deg(np.arccos(np.clip(
        (np.trace(delta[:3, :3]) - 1) / 2, -1, 1))))
    before_fitness = float(baseline.fitness)
    before_rmse = float(baseline.inlier_rmse)
    after_fitness = float(fitted.fitness)
    after_rmse = float(fitted.inlier_rmse)
    accepted = (
        np.isfinite(candidate).all()
        and np.allclose(candidate[3], [0, 0, 0, 1], atol=1e-5)
        and np.allclose(candidate[:3, :3].T @ candidate[:3, :3],
                        np.eye(3), atol=1e-3)
        and translation <= .08 and angle <= 4.
        and after_fitness >= max(.2, before_fitness - .015)
        and after_rmse + .002 <= before_rmse
    )
    return (candidate if accepted else start), {
        "accepted": bool(accepted), "translation_delta_m": translation,
        "rotation_delta_deg": angle, "before_fitness": before_fitness,
        "after_fitness": after_fitness, "before_rmse_m": before_rmse,
        "after_rmse_m": after_rmse,
    }

def fixed_view_specs(data, poses):
    """Three original camera positions and two fixed independent free views."""
    from open3d_official_room_baseline import spaced_real_frames
    from tum_graph_visual import look_at
    frames = spaced_real_frames(len(data.color_paths), 360)
    specs = []
    for idx in (frames[0], frames[len(frames) // 2], frames[-1]):
        specs.append({"kind": "recorded", "frame": int(idx),
                      "file": f"visual/photo-vs-mesh-{idx:05d}.png",
                      "pose": np.asarray(poses[idx])})
    for j, (idx, dx) in enumerate(((frames[len(frames) // 3], .35),
                                   (frames[2 * len(frames) // 3], -.35)), 1):
        c2w = np.linalg.inv(poses[idx])
        eye = c2w[:3, 3] + dx * c2w[:3, 0]
        free = look_at(eye, c2w[:3, 3] + 1.8 * c2w[:3, 2])
        specs.append({"kind": "free", "frame": int(idx),
                      "file": f"visual/free-view-{j}.png",
                      "pose": np.linalg.inv(free)})
    return specs


def render_one_mesh(o3d, mesh, specs, intrinsics):
    """Render one candidate and release its BVH BEFORE building the next one.

    Original supplied PRIOR can contain millions of faces. Building two Open3D
    raycasting scenes concurrently can exhaust a standard CI runner.
    """
    import gc
    from redwood_dense_native_review import raycast
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rendered = []
    for spec in specs:
        rgb, hits = raycast(o3d, scene, mesh, intrinsics, spec["pose"])
        rendered.append((rgb, hits))
    del scene
    gc.collect()
    return rendered


def save_five_panels(data, specs, previous, current, output):
    """Combine identical cameras only after both independent renders finish."""
    from redwood_gap_guided_fusion import panel
    (output / "visual").mkdir(exist_ok=True)
    views = []
    for spec, (before, old_hits), (after, new_hits) in zip(
            specs, previous, current):
        photo = None
        if spec["kind"] == "recorded":
            with Image.open(data.color_paths[spec["frame"]]) as image:
                photo = image.convert("RGB").copy()
        panel(output / spec["file"], photo, before, after,
              "REAL RGB / 500K PRIOR / ORIGINAL PRIOR" if photo is not None
              else "FREE CAMERA / 500K PRIOR / ORIGINAL PRIOR")
        views.append({"kind": spec["kind"], "frame": spec["frame"],
                      "file": spec["file"], "old_pixels": old_hits,
                      "new_pixels": new_hits})
    return views


def run(observed_path, cache, output):
    import open3d as o3d
    from redwood_prior_gapfill import align_prior, load_measured
    from redwood_prior_measured_hybrid import (
        distance_to_prior_triangles, select_measured_detail_faces, export_glb)
    from open3d_official_room_baseline import read_poses

    if output.exists():
        raise FileExistsError("Use a new output directory")
    data = o3d.data.BedroomRGBDImages(data_root=str(cache))
    observed = load_measured(observed_path, o3d)
    observed.compute_vertex_normals()
    full = o3d.io.read_triangle_mesh(data.reconstruction_path)
    if len(full.triangles) < 2000000:
        raise RuntimeError("Expected original multi-million-face supplied PRIOR")
    original_faces = len(full.triangles)
    transform, registration = align_prior(full, observed, o3d)
    transform, refinement = refine_registered_prior(full, observed, transform, o3d)
    full.transform(transform)
    low = full.simplify_quadric_decimation(500000)
    low.compute_vertex_normals()
    if not full.has_vertex_normals():
        full.compute_vertex_normals()
    low_rgb, low_support = transfer_colors_chunked(low, observed)
    low.vertex_colors = o3d.utility.Vector3dVector(low_rgb)
    # Hold observed detail constant so this A/B isolates prior decimation.
    xyz = np.asarray(observed.vertices)
    triangles = np.asarray(observed.triangles)
    distances = distance_to_prior_triangles(o3d, low, xyz)
    selected = select_measured_detail_faces(xyz, triangles, distances)
    detail = o3d.geometry.TriangleMesh()
    if len(selected):
        used, inv = np.unique(selected, return_inverse=True)
        detail.vertices = o3d.utility.Vector3dVector(xyz[used])
        detail.triangles = o3d.utility.Vector3iVector(inv.reshape(-1, 3))
        detail.vertex_colors = o3d.utility.Vector3dVector(
            np.asarray(observed.vertex_colors)[used])
    old = low + detail
    full_rgb, full_support = transfer_colors_chunked(full, observed)
    full.vertex_colors = o3d.utility.Vector3dVector(full_rgb)
    new = full + detail
    # Export before BVH creation, so a partial failure retains the actual mesh.
    output.mkdir(parents=True)
    export_glb(output / "original-resolution-PRIOR-plus-measured.glb", new)
    export_glb(output / "500k-PRIOR-plus-measured.glb", old)
    # Independent geometry control: the original supplied scan WITHOUT the
    # measured-only detail islands. The scan is explicitly an EXISTING PRIOR.
    # This distinguishes decimation holes from overlapping measured patches.
    export_glb(output / "original-resolution-PRIOR-only.glb", full)
    # Avoid a duplicate multi-hundred-MB source-mask GLB: preserve provenance
    # in metadata; ALL prior faces are the first original_faces triangles.
    trajectory = o3d.io.read_pinhole_camera_trajectory(data.trajectory_log_path)
    k, poses = read_poses(trajectory, list(data.color_paths), data.trajectory_log_path)
    # Export first; release the original scan and its duplicates before
    # allocating the full-resolution raycasting acceleration structure.
    prior_faces = original_faces
    low_faces = len(low.triangles)
    detail_faces = len(selected)
    prior_color_hits = int(full_support.sum())
    low_color_hits = int(low_support.sum())
    unknown_prior_colors = int((~full_support).sum())
    import gc
    del full, low, detail, observed, full_rgb, low_rgb, full_support, low_support
    gc.collect()
    specs = fixed_view_specs(data, poses)
    print("Rendering decimated 500k PRIOR, then releasing its BVH", flush=True)
    old_rendered = render_one_mesh(o3d, old, specs, k)
    del old
    gc.collect()
    print("Rendering original multi-million-face PRIOR separately", flush=True)
    new_rendered = render_one_mesh(o3d, new, specs, k)
    del new
    gc.collect()
    views = save_five_panels(data, specs, old_rendered, new_rendered, output)
    board = Image.new("RGB", (960, 268 * len(views)), (26, 26, 26))
    for i, entry in enumerate(views):
        with Image.open(output / entry["file"]) as image:
            board.paste(image.convert("RGB"), (0, 268 * i))
    board.save(output / "five-view-original-prior-ab.png")
    report = {
        "scene": "Real Redwood Bedroom, 640x480 measured RGB-D",
        "comparison": "500k vs original-resolution EXISTING SCAN PRIOR; identical measured detail",
        "geometry_source": "Open3D BedroomRGBDImages.reconstruction_path",
        "prior_is_independent_video_reconstruction": False,
        "original_prior_faces": prior_faces,
        "decimated_prior_faces": low_faces,
        "measured_detail_faces_in_both": detail_faces,
        "original_prior_vertices_colored_from_measured": prior_color_hits,
        "decimated_prior_vertices_colored_from_measured": low_color_hits,
        "original_prior_vertices_unknown_color": unknown_prior_colors,
        "registration": registration,
        "robust_point_to_plane_refinement": refinement,
        "full_mesh": "original-resolution-PRIOR-plus-measured.glb",
        "small_mesh": "500k-PRIOR-plus-measured.glb",
        "original_prior_only_mesh": "original-resolution-PRIOR-only.glb",
        "prior_only_role": "Supplied pre-existing scan, no measured-only detail islands; NOT independently recovered",
        "provenance": "First original_prior_faces faces in full mesh are supplied prior; remainder measured-only; all first decimated_prior_faces faces in small mesh prior",
        "views": views,
        "human_visual_acceptance": "PENDING",
        "free_orbit": "FAIL until human verified",
        "issue_74": "FAIL unchanged",
        "old_desk": "FAIL 16.7779cm > 15cm unchanged",
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"original_prior_faces": prior_faces,
                      "detail_faces": detail_faces, "views": len(views)}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.observed, args.cache, args.output)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "FAILURE.txt").write_text(
            type(exc).__name__ + ": " + str(exc) + "\n")
        raise
