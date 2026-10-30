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


def run(observed_path, cache, output):
    import open3d as o3d
    from redwood_prior_gapfill import align_prior, load_measured
    from redwood_prior_measured_hybrid import (
        distance_to_prior_triangles, select_measured_detail_faces, export_glb)
    from open3d_official_room_baseline import read_poses
    from redwood_gap_guided_fusion import compare

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
    # Avoid a duplicate multi-hundred-MB source-mask GLB: preserve provenance
    # in metadata; ALL prior faces are the first original_faces triangles.
    trajectory = o3d.io.read_pinhole_camera_trajectory(data.trajectory_log_path)
    k, poses = read_poses(trajectory, list(data.color_paths), data.trajectory_log_path)
    views = compare(o3d, data, poses, k, old, new, output)
    board = Image.new("RGB", (960, 268 * len(views)), (26, 26, 26))
    for i, entry in enumerate(views):
        path = output / entry["file"]
        with Image.open(path) as im:
            panel = im.convert("RGB").copy()
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, panel.width, 27), fill=(26, 26, 26))
        draw.text((8, 8),
                  "REAL RGB / 500K PRIOR / ORIGINAL PRIOR" if entry["kind"] == "recorded"
                  else "FREE VIEW / 500K PRIOR / ORIGINAL PRIOR", fill="white")
        panel.save(path)
        board.paste(panel, (0, 268 * i))
    board.save(output / "five-view-original-prior-ab.png")
    report = {
        "scene": "Real Redwood Bedroom, 640x480 measured RGB-D",
        "comparison": "500k vs original-resolution EXISTING SCAN PRIOR; identical measured detail",
        "geometry_source": "Open3D BedroomRGBDImages.reconstruction_path",
        "prior_is_independent_video_reconstruction": False,
        "original_prior_faces": original_faces,
        "decimated_prior_faces": len(low.triangles),
        "measured_detail_faces_in_both": len(selected),
        "original_prior_vertices_colored_from_measured": int(full_support.sum()),
        "decimated_prior_vertices_colored_from_measured": int(low_support.sum()),
        "original_prior_vertices_unknown_color": int((~full_support).sum()),
        "registration": registration,
        "full_mesh": "original-resolution-PRIOR-plus-measured.glb",
        "small_mesh": "500k-PRIOR-plus-measured.glb",
        "provenance": "First original_prior_faces faces in full mesh are supplied prior; remainder measured-only; all first decimated_prior_faces faces in small mesh prior",
        "views": views,
        "human_visual_acceptance": "PENDING",
        "free_orbit": "FAIL until human verified",
        "issue_74": "FAIL unchanged",
        "old_desk": "FAIL 16.7779cm > 15cm unchanged",
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"original_prior_faces": original_faces,
                      "detail_faces": len(selected), "views": len(views)}), flush=True)
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
