"""Remove tiny disconnected observed TSDF fragments without inventing geometry.

Real input: measured Redwood Bedroom RGB-D and logged camera poses, fused by
the existing Open3D 600-frame branch. The prior 3D scan is not used.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def component_face_mask(faces, vertex_count, min_faces=100):
    """Return mask of faces belonging to vertex-connected components.

    Only removes existing triangles; never joins disconnected surfaces,
    fills missing regions, or changes observed coordinates.
    """
    f = np.asarray(faces)
    if (f.ndim != 2 or f.shape[1] != 3 or len(f) == 0
            or not np.issubdtype(f.dtype, np.integer)
            or np.any(f < 0) or np.any(f >= vertex_count)):
        raise ValueError("Invalid observed triangle indices")
    if (type(vertex_count) is not int or vertex_count < 3
            or type(min_faces) is not int or not 1 <= min_faces <= 100000):
        raise ValueError("Invalid component filter settings")
    edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    graph = coo_matrix(
        (np.ones(len(edges), dtype=np.uint8), (edges[:, 0], edges[:, 1])),
        shape=(vertex_count, vertex_count)).tocsr()
    count, labels = connected_components(graph, directed=False)
    face_labels = labels[f[:, 0]]
    sizes = np.bincount(face_labels, minlength=count)
    keep = sizes[face_labels] >= min_faces
    return keep, sizes[sizes > 0]


def remove_tiny_observed_components(mesh, min_faces=100):
    import open3d as o3d

    if not mesh.has_triangles() or not mesh.has_vertex_colors():
        raise ValueError("Input must be a real colored observed mesh")
    original_faces = np.asarray(mesh.triangles)
    original_vertices = np.asarray(mesh.vertices)
    keep, sizes = component_face_mask(original_faces, len(original_vertices), min_faces)
    if not np.any(keep):
        raise ValueError("Refusing to erase all observed geometry")
    cleaned = o3d.geometry.TriangleMesh(mesh)
    cleaned.remove_triangles_by_mask(np.asarray(~keep))
    cleaned.remove_unreferenced_vertices()
    cleaned.compute_vertex_normals()
    if not cleaned.has_vertex_colors() or len(cleaned.triangles) != int(keep.sum()):
        raise RuntimeError("Observed colors or triangle topology were lost")
    return cleaned, {
        "original_faces": int(len(original_faces)),
        "new_faces": int(keep.sum()),
        "removed_faces": int((~keep).sum()),
        "original_components": int(len(sizes)),
        "removed_components": int(np.count_nonzero(sizes < min_faces)),
        "retained_components": int(np.count_nonzero(sizes >= min_faces)),
        "threshold_faces": min_faces,
    }


def run(source, cache, output, min_faces=100):
    import open3d as o3d
    from open3d_official_room_baseline import read_poses
    from redwood_dense_native_review import export_full
    from redwood_gap_guided_fusion import compare

    old = o3d.io.read_triangle_mesh(str(source))
    if not old.has_triangles() or not old.has_vertex_colors():
        raise ValueError("Missing measured 600-frame colored source PLY")
    cleaned, stats = remove_tiny_observed_components(old, min_faces)
    output.mkdir(parents=True, exist_ok=False)
    ply = output / "redwood-observed-components-cleaned.ply"
    if not o3d.io.write_triangle_mesh(str(ply), cleaned):
        raise RuntimeError("Could not write observed-only PLY")
    exported = export_full(cleaned, output)
    glb = "redwood-observed-components-cleaned.glb"
    (output / exported["glb"]).rename(output / glb)

    data = o3d.data.BedroomRGBDImages(data_root=str(cache))
    if len(data.color_paths) != len(data.depth_paths):
        raise ValueError("Mismatched measured RGB-D source")
    trajectory = o3d.io.read_pinhole_camera_trajectory(data.trajectory_log_path)
    k, poses = read_poses(trajectory, data.color_paths, data.trajectory_log_path)
    if k[:2] != (640, 480):
        raise ValueError("Original Redwood calibration must be 640x480")
    views = compare(o3d, data, poses, k, old, cleaned, output)
    board = Image.new("RGB", (960, 268 * len(views)), (26, 26, 26))
    for i, entry in enumerate(views):
        path = output / entry["file"]
        with Image.open(path) as file:
            panel = file.convert("RGB").copy()
        label = ("REAL RGB / FAR-SIDE / CLEANED" if entry["kind"] == "recorded"
                 else "FREE VIEW / FAR-SIDE / CLEANED")
        ImageDraw.Draw(panel).rectangle((0, 0, panel.width, 27), fill=(26, 26, 26))
        ImageDraw.Draw(panel).text((8, 8), label, fill="white")
        panel.save(path)
        board.paste(panel, (0, i * 268))
    board.save(output / "five-view-component-cleanup-ab.png")
    report = {
        "scene": "Real Redwood Bedroom",
        "source": "600 measured 640x480 RGB-D frames, calibrated depth and logged poses",
        "source_run": "https://github.com/REDrighthand05/echo4d/actions/runs/40969878515",
        "geometry_change": "Remove only observed mesh components with fewer than 100 triangles",
        **stats,
        "reference_prior_used": False,
        "new_glb": glb,
        "views": views,
        "visual_acceptance": "PENDING human comparison",
        "free_orbit": "FAIL until verified",
        "issue_74": "FAIL unchanged",
        "old_desk": "FAIL 16.7779cm > 15cm unchanged",
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-ply", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-faces", type=int, default=100)
    a = p.parse_args()
    try:
        run(a.source_ply, a.cache, a.output, a.min_faces)
    except Exception as error:
        a.output.mkdir(parents=True, exist_ok=True)
        (a.output / "FAILURE.txt").write_text(
            f"{type(error).__name__}: {error}\n")
        raise


if __name__ == "__main__":
    main()
