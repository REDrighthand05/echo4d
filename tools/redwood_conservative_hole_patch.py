"""Close only small, nearly planar boundary loops in an existing colored room GLB.

Existing scan PRIOR and measured faces stay untouched. Every appended triangle
is explicitly INFERRED; this is not independent RGB-D reconstruction.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def triangulate_polygon(xy):
    """Ear clipping in a local 2D plane, without inventing new vertices."""
    xy = np.asarray(xy, dtype=np.float64)
    area2 = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) -
                   xy[:, 1] * np.roll(xy[:, 0], -1))
    if abs(area2) < 1e-10:
        return []
    sign = 1 if area2 > 0 else -1
    indices = list(range(len(xy)))
    result = []
    while len(indices) > 3:
        found = False
        for j in range(len(indices)):
            a, b, c = (indices[(j - 1) % len(indices)], indices[j],
                       indices[(j + 1) % len(indices)])
            ab, ac = xy[b] - xy[a], xy[c] - xy[a]
            cross = np.linalg.det(np.stack((ab, ac))) * sign
            if cross <= 1e-10:
                continue
            inside = False
            for q in indices:
                if q in (a, b, c):
                    continue
                pt = xy[q]
                def side(u, v):
                    x, y = xy[v] - xy[u], pt - xy[u]
                    return sign * (x[0] * y[1] - x[1] * y[0])
                if min(side(a, b), side(b, c), side(c, a)) >= -1e-10:
                    inside = True
                    break
            if inside:
                continue
            result.append((a, b, c))
            indices.pop(j)
            found = True
            break
        if not found:
            return []
    result.append(tuple(indices))
    return result


def infer_small_hole_faces(vertices, faces, max_perimeter=.65,
                           max_diameter=.25, max_planarity=.012,
                           max_edges=100):
    """Return inferred caps and statistics; never alter the original triangles."""
    xyz = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or tri.ndim != 2
            or tri.shape[1] != 3 or len(tri) == 0 or len(xyz) == 0
            or not np.isfinite(xyz).all() or np.any(tri < 0)
            or np.any(tri >= len(xyz)) or max_perimeter <= 0
            or max_diameter <= 0 or max_planarity <= 0 or max_edges < 3):
        raise ValueError("Invalid mesh or conservative hole limits")
    directed = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]],
                               tri[:, [2, 0]]))
    _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                   return_inverse=True, return_counts=True)
    boundary = directed[counts[inverse] == 1]
    outgoing, incoming = defaultdict(list), defaultdict(list)
    for u, v in boundary:
        outgoing[int(u)].append(int(v))
        incoming[int(v)].append(int(u))
    regular = {u for u in outgoing if len(outgoing[u]) == 1
               and len(incoming.get(u, [])) == 1}
    visited = set()
    patches = []
    patch_loop_ids = []
    loop_count = 0
    for start in regular:
        if start in visited:
            continue
        loop, local, current = [], set(), start
        while current in regular and current not in local and len(loop) <= max_edges:
            local.add(current)
            loop.append(current)
            current = outgoing[current][0]
        visited.update(local)
        if current != start or len(loop) < 3 or len(loop) > max_edges:
            continue
        points = xyz[loop]
        perimeter = np.linalg.norm(points - np.roll(points, -1, axis=0),
                                   axis=1).sum()
        if perimeter > max_perimeter:
            continue
        center = points.mean(axis=0)
        diameter = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        if diameter > max_diameter:
            continue
        _, _, basis = np.linalg.svd(points - center, full_matrices=False)
        if np.max(np.abs((points - center) @ basis[-1])) > max_planarity:
            continue
        uv = (points - center) @ basis[:2].T
        area = abs(np.sum(uv[:, 0] * np.roll(uv[:, 1], -1) -
                          uv[:, 1] * np.roll(uv[:, 0], -1))) / 2
        if area < 1e-7 or perimeter ** 2 / (4 * np.pi * area) > 8:
            continue
        ears = triangulate_polygon(uv)
        if len(ears) != len(loop) - 2:
            continue
        # Opposite winding to the original face's boundary u->v.
        patches.extend((loop[a], loop[c], loop[b]) for a, b, c in ears)
        patch_loop_ids.extend([loop_count] * len(ears))
        loop_count += 1
    patches = np.asarray(patches, dtype=np.int64).reshape(-1, 3)
    patch_loop_ids = np.asarray(patch_loop_ids, dtype=np.int64)
    # Reject entire loops whose diagonals would introduce non-manifold edges.
    for _ in range(5):
        if not len(patches):
            break
        added = np.concatenate((patches[:, [0, 1]], patches[:, [1, 2]],
                                patches[:, [2, 0]]))
        combined = np.concatenate((directed, added))
        _, edge_ids, multiplicity = np.unique(
            np.sort(combined, axis=1), axis=0,
            return_inverse=True, return_counts=True)
        bad_added = multiplicity[edge_ids[len(directed):]] > 2
        if not bad_added.any():
            break
        bad_face_ids = np.tile(np.arange(len(patches)), 3)[bad_added]
        bad_loops = np.unique(patch_loop_ids[bad_face_ids])
        keep = ~np.isin(patch_loop_ids, bad_loops)
        patches, patch_loop_ids = patches[keep], patch_loop_ids[keep]
    loop_count = len(np.unique(patch_loop_ids))
    return patches, {
        "boundary_edges": int(len(boundary)),
        "inferred_small_holes": loop_count,
        "inferred_faces": len(patches),
    }


def main():
    import trimesh
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; refusing to overwrite")
    mesh = trimesh.load(str(args.input), force="mesh", process=False)
    patches, report = infer_small_hole_faces(mesh.vertices, mesh.faces)
    if not len(patches):
        raise RuntimeError("No conservative real-scene holes qualified")
    original_faces = len(mesh.faces)
    mesh.faces = np.concatenate((mesh.faces, patches))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(args.output))
    report.update({
        "input": str(args.input), "output": str(args.output),
        "geometry_provenance": "All appended faces are INFERRED, not measured",
        "source_face_count": original_faces,
        "inferred_face_range_start": original_faces,
        "visual_acceptance": "PENDING human review",
        "issue_74": "FAIL unchanged", "old_desk": "FAIL unchanged",
    })
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
