"""Postprocess an immutable real 158-view TUM Room TSDF artifact.

Opt-in removal of tiny disconnected islands only; no invented geometry, no
hole fill, no claim of improved heldout sensor depth. Both meshes are retained.
The source is the actual GitHub Actions 40759653811 158-view result, not a
synthetic test or a downloaded model prior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SOURCE_RUN = 40759653811
SOURCE_SHA = "9132850c895e722cb287c35e22c2c458080b4326"
SOURCE_MESH_SHA = "1dd1b4944affbde11d17b0d8428a98482cd078d377fe146ddcce5383dd4cc727"


def micro_island_mask(vertices: np.ndarray, faces: np.ndarray, *,
                      max_faces: int = 9, max_extent_m: float = .08):
    """Return a face deletion mask; require BOTH tiny face count and 3D extent.

    Uses shared-vertex connectivity, like existing Echo4D face_components.
    A long narrow real surface is preserved even when it has few triangles.
    """
    v, f = np.asarray(vertices), np.asarray(faces)
    if (v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3
            or len(v) < 3 or not len(f) or not np.isfinite(v).all()
            or np.any(f < 0) or np.any(f >= len(v))):
        raise ValueError("Invalid observed mesh")
    if (type(max_faces) is not int or not 1 <= max_faces <= 100
            or not isinstance(max_extent_m, (int, float))
            or isinstance(max_extent_m, bool) or not np.isfinite(max_extent_m)
            or not .01 <= max_extent_m <= .5):
        raise ValueError("Invalid tiny-component thresholds")
    a = np.concatenate((f[:, 0], f[:, 1], f[:, 2]))
    b = np.concatenate((f[:, 1], f[:, 2], f[:, 0]))
    graph = coo_matrix((np.ones(len(a), dtype=np.uint8), (a, b)),
                       shape=(len(v), len(v))).tocsr()
    n, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels[f[:, 0]], minlength=n)
    lo = np.full((n, 3), np.inf)
    hi = np.full((n, 3), -np.inf)
    np.minimum.at(lo, labels, v)
    np.maximum.at(hi, labels, v)
    extent = np.linalg.norm(hi - lo, axis=1)
    small = (sizes > 0) & (sizes <= max_faces) & (extent < max_extent_m)
    mask = small[labels[f[:, 0]]]
    return mask, {
        "removed_components": int(np.count_nonzero(small)),
        "removed_faces": int(mask.sum()),
        "retained_faces": int(len(f) - mask.sum()),
        "threshold_max_faces": max_faces,
        "threshold_max_extent_m": max_extent_m,
        "small_real_islands_may_also_be_removed": True,
        "unknown_surface_fill": False,
    }


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(source: Path, output: Path):
    import trimesh
    from echo4d.model import Camera
    from tum_graph_visual import rasterize_vertex_rgb
    from tum_sparse_training_density import sheet

    report_path = source / "room-sparse-quality-report.json"
    mesh_path = source / "room-158-fine-wide" / "mesh.ply"
    evidence = json.loads(report_path.read_text(encoding="utf8"))
    if (evidence.get("checkout_sha") != SOURCE_SHA
            or evidence.get("status") != "real_room_120_vs_158_real_views_visual_review_pending"
            or evidence["fusion"]["room-158-fine-wide"]["frames_integrated"] != 158
            or sha(mesh_path) != SOURCE_MESH_SHA):
        raise ValueError("Real 158-view source artifact or original Mesh SHA mismatch")
    raw = trimesh.load(str(mesh_path), process=False)
    if not isinstance(raw, trimesh.Trimesh):
        raise ValueError("Expected one observed triangle Mesh")
    mask, pruning = micro_island_mask(raw.vertices, raw.faces)
    if not mask.any():
        raise ValueError("Source has no eligible tiny islands; candidate not generated")
    clean = raw.submesh([np.flatnonzero(~mask)], append=True, repair=False)
    output.mkdir(parents=True, exist_ok=True)
    raw.export(output / "room-158-original.glb")
    clean.export(output / "room-158-micropruned.glb")
    clean.export(output / "room-158-micropruned.ply")

    camera = Camera(320, 240, 262.5, 262.5, 159.75, 119.75)
    visual = output / "visual"
    visual.mkdir(exist_ok=True)
    pictures = []
    raw_rgb = np.asarray(raw.visual.vertex_colors)[:, :3].astype(np.uint8)
    clean_rgb = np.asarray(clean.visual.vertex_colors)[:, :3].astype(np.uint8)
    for i, entry in enumerate((v for v in evidence["visual"]
                               if v["kind"] == "free_camera"), 1):
        pose = np.asarray(entry["camera_to_world"]).reshape(4, 4, order="F")
        before, before_pixels = rasterize_vertex_rgb(
            np.asarray(raw.vertices), np.asarray(raw.faces), raw_rgb,
            camera, pose, near=.15, far=10.)
        after, after_pixels = rasterize_vertex_rgb(
            np.asarray(clean.vertices), np.asarray(clean.faces), clean_rgb,
            camera, pose, near=.15, far=10.)
        name = f"free-{i}-room-158-microprune-ab.png"
        sheet(None, before, after, visual / name,
              f"Real TUM room fixed free camera {i}",
              ("158-view raw TSDF", "158-view small-island removal"))
        pictures.append({"name": name, "original_pixels": int(before_pixels),
                         "candidate_pixels": int(after_pixels),
                         "pose_from_original_real_run": True})
    # These three are original source/120-vs-158 reference sheets, NOT new
    # heldout camera renders of the pruned candidate (poses unavailable).
    baseline_refs = []
    for entry in evidence["visual"]:
        if entry["kind"] == "heldout_camera":
            old = source / "visual" / entry["name"]
            dest = visual / ("reference-only-" + entry["name"])
            shutil.copyfile(old, dest)
            baseline_refs.append(dest.name)
    result = {
        "source_real_run": SOURCE_RUN, "source_checkout_sha": SOURCE_SHA,
        "source_original_mesh_sha256": SOURCE_MESH_SHA,
        "original_scene": "TUM Freiburg1 Room, real 640x480 RGB-D, 158 measured known-pose training views",
        "geometry_operation": "remove disconnected components with <=9 faces AND <8cm 3D extent",
        "pruning": pruning, "original_faces": int(len(raw.faces)),
        "candidate_faces": int(len(clean.faces)),
        "original_vertices": int(len(raw.vertices)),
        "candidate_vertices": int(len(clean.vertices)),
        "candidate_mesh_sha256": sha(output / "room-158-micropruned.ply"),
        "new_candidate_free_view_comparisons": pictures,
        "three_heldout_source_reference_sheets_not_candidate_renders": baseline_refs,
        "same_sensor_holdout_candidate": "not rerun; must not claim depth improvement",
        "human_visual_acceptance": "pending",
        "issue_74": "FAIL / NOT ACCEPTED",
        "historical_desk_v2": "FAIL 16.7779cm > 15cm; unchanged",
        "limitations": "Lossy removal may erase tiny real objects; no missing surfaces reconstructed",
    }
    (output / "microprune-report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source, args.output)
    print(json.dumps({k: result[k] for k in (
        "source_real_run", "pruning", "original_faces", "candidate_faces",
        "candidate_mesh_sha256", "issue_74")}, indent=2))


if __name__ == "__main__":
    main()
