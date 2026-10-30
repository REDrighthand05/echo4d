"""High-information Redwood Bedroom: explicit existing scan PRIOR + measured RGB-D details.

All supplied scan faces remain labeled PRIOR. Added faces come only from the
measured 600-frame RGB-D reconstruction. Not video-only geometry recovery.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def select_measured_detail_faces(vertices, faces, distance_to_prior_m, *,
                                 min_gap_m=.055, max_gap_m=.25,
                                 max_edge_m=.075, min_component_faces=80):
    """Return only connected measured triangles missing from the prior surface."""
    xyz = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(faces)
    distance = np.asarray(distance_to_prior_m, dtype=np.float64)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz)
            or tri.ndim != 2 or tri.shape[1] != 3 or not len(tri)
            or not np.issubdtype(tri.dtype, np.integer)
            or np.any(tri < 0) or np.any(tri >= len(xyz))
            or distance.shape != (len(xyz),)
            or not np.isfinite(xyz).all() or not np.isfinite(distance).all()
            or min_gap_m <= 0 or max_gap_m <= min_gap_m
            or max_edge_m <= 0 or not isinstance(min_component_faces, int)
            or isinstance(min_component_faces, bool) or min_component_faces < 1):
        raise ValueError("Invalid observed geometry, prior distances or thresholds")
    eligible = (distance >= min_gap_m) & (distance <= max_gap_m)
    mask = eligible[tri].all(axis=1)
    edge01 = np.linalg.norm(xyz[tri[:, 0]] - xyz[tri[:, 1]], axis=1)
    edge12 = np.linalg.norm(xyz[tri[:, 1]] - xyz[tri[:, 2]], axis=1)
    edge20 = np.linalg.norm(xyz[tri[:, 2]] - xyz[tri[:, 0]], axis=1)
    mask &= np.maximum.reduce((edge01, edge12, edge20)) <= max_edge_m
    candidate = tri[mask]
    if not len(candidate):
        return np.empty((0, 3), dtype=np.int32)
    rows = np.concatenate((candidate[:, 0], candidate[:, 1], candidate[:, 2]))
    cols = np.concatenate((candidate[:, 1], candidate[:, 2], candidate[:, 0]))
    graph = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)),
                       shape=(len(xyz), len(xyz))).tocsr()
    n, labels = connected_components(graph, directed=False)
    ids = labels[candidate[:, 0]]
    counts = np.bincount(ids, minlength=n)
    return candidate[counts[ids] >= min_component_faces].astype(np.int32)


def distance_to_prior_triangles(o3d, prior, points, chunk=100000):
    """Compute metric distance to aligned PRIOR triangles, not sparse vertices."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(prior))
    xyz = np.asarray(points, dtype=np.float32)
    result = np.empty(len(xyz), dtype=np.float32)
    for i in range(0, len(xyz), chunk):
        query = o3d.core.Tensor(np.ascontiguousarray(xyz[i:i + chunk]))
        result[i:i + chunk] = scene.compute_distance(query).numpy()
    return result


def export_glb(path, mesh):
    import trimesh
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    rgb = np.asarray(mesh.vertex_colors)
    rgba = np.column_stack((np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8),
                            np.full(len(vertices), 255, dtype=np.uint8)))
    trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=rgba,
                    process=False).export(str(path))


def run(observed_path, cache, output, prior_faces=500000):
    import open3d as o3d
    from redwood_prior_gapfill import align_prior, load_measured
    from redwood_prior_first_room import transfer_observed_colors
    from open3d_official_room_baseline import read_poses
    from redwood_gap_guided_fusion import compare

    if output.exists():
        raise FileExistsError("Output directory already exists")
    data = o3d.data.BedroomRGBDImages(data_root=str(cache))
    observed = load_measured(observed_path, o3d)
    supplied = o3d.io.read_triangle_mesh(data.reconstruction_path)
    if not supplied.has_triangles() or len(supplied.triangles) < prior_faces:
        raise RuntimeError("No usable supplied high-information prior scan")
    transform, registration = align_prior(supplied, observed, o3d)
    supplied.transform(transform)
    prior = supplied.simplify_quadric_decimation(prior_faces)
    prior.compute_vertex_normals()
    observed.compute_vertex_normals()
    prior_colors, prior_color_source = transfer_observed_colors(
        prior, observed, max_distance_m=.09)
    prior.vertex_colors = o3d.utility.Vector3dVector(prior_colors)

    xyz, triangles = np.asarray(observed.vertices), np.asarray(observed.triangles)
    distances = distance_to_prior_triangles(o3d, prior, xyz)
    selected = select_measured_detail_faces(xyz, triangles, distances)
    detail = o3d.geometry.TriangleMesh()
    if len(selected):
        used, inv = np.unique(selected, return_inverse=True)
        detail.vertices = o3d.utility.Vector3dVector(xyz[used])
        detail.triangles = o3d.utility.Vector3iVector(inv.reshape(-1, 3))
        detail.vertex_colors = o3d.utility.Vector3dVector(
            np.asarray(observed.vertex_colors)[used])
        detail.compute_vertex_normals()
    hybrid = prior + detail
    hybrid.compute_vertex_normals()
    if not hybrid.has_vertex_colors() or not hybrid.has_triangles():
        raise RuntimeError("Hybrid has no colored triangles")
    output.mkdir(parents=True)
    export_glb(output / "PRIOR-plus-measured-detail.glb", hybrid)
    if not o3d.io.write_triangle_mesh(str(output / "PRIOR-plus-measured-detail.ply"), hybrid):
        raise RuntimeError("Cannot save colored hybrid PLY")

    provenance = o3d.geometry.TriangleMesh(hybrid)
    colors = np.zeros((len(hybrid.vertices), 3), dtype=float)
    colors[:len(prior.vertices)] = np.where(prior_color_source[:, None] == 1,
                                           [.15, .65, .75], [.95, .48, .12])
    colors[len(prior.vertices):] = [.30, .95, .35]
    provenance.vertex_colors = o3d.utility.Vector3dVector(colors)
    export_glb(output / "PRIOR-vs-measured-source.glb", provenance)

    trajectory = o3d.io.read_pinhole_camera_trajectory(data.trajectory_log_path)
    k, poses = read_poses(trajectory, list(data.color_paths), data.trajectory_log_path)
    views = compare(o3d, data, poses, k, observed, hybrid, output)
    board = Image.new("RGB", (960, 268 * len(views)), (26, 26, 26))
    for i, entry in enumerate(views):
        path = output / entry["file"]
        with Image.open(path) as f:
            panel = f.convert("RGB").copy()
        ImageDraw.Draw(panel).rectangle((0, 0, panel.width, 27), fill=(26, 26, 26))
        ImageDraw.Draw(panel).text((8, 8),
            "REAL RGB / MEASURED ONLY / LABELED PRIOR+MEASURED"
            if entry["kind"] == "recorded" else
            "FREE VIEW / MEASURED ONLY / LABELED PRIOR+MEASURED", fill="white")
        panel.save(path)
        board.paste(panel, (0, 268 * i))
    board.save(output / "five-view-prior-measured-hybrid.png")
    report = {
        "scene": "Real Redwood Bedroom",
        "input": "original 640x480 RGB-D with metric poses",
        "geometry_role": "Existing room scan is PRIOR; added details are measured RGB-D-only",
        "prior_source": "Open3D BedroomRGBDImages.reconstruction_path",
        "prior_faces_original": len(supplied.triangles),
        "prior_faces_used": len(prior.triangles),
        "observed_faces_input": len(observed.triangles),
        "measured_detail_faces_added": len(selected),
        "hybrid_faces": len(hybrid.triangles),
        "prior_to_observed_registration": registration,
        "prior_vertices_with_measured_color": int(np.count_nonzero(prior_color_source == 1)),
        "prior_vertices_with_unknown_color": int(np.count_nonzero(prior_color_source == 2)),
        "provenance": "cyan=prior with measured color; orange=prior unknown color; green=measured-only geometry",
        "prior_is_independent_reconstruction": False,
        "visual_acceptance": "PENDING human five-view review",
        "free_orbit": "FAIL until directly verified", "issue_74": "FAIL unchanged",
        "old_desk": "FAIL 16.7779cm > 15cm unchanged", "views": views,
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prior-faces", type=int, default=500000)
    args = parser.parse_args()
    if not 100000 <= args.prior_faces <= 2000000:
        parser.error("prior-faces must be 100000..2000000")
    run(args.observed, args.cache, args.output, args.prior_faces)


if __name__ == "__main__":
    main()
