"""Opt-in RGB-inferred edge support and short topological bridges for TSDF faces.

Only known-pose RGB-derived depth maps vote. Weak faces can close one/two-face
gaps between supported components, but cannot grow dangling sheets.
Sensor depth is never consulted. The original TSDF faces remain the A/B control.
"""
from __future__ import annotations

import numpy as np

from .known_pose_edge_crossview import _face_keys, filter_crossview_faces
from .known_pose_edge_extraction import _attached_faces


def _adjacency(faces: np.ndarray) -> list[set[int]]:
    adjacency: list[set[int]] = [set() for _ in range(len(faces))]
    first: dict[tuple[int, int], int] = {}
    for i, (a, b, c) in enumerate(faces):
        for x, y in ((a, b), (b, c), (c, a)):
            edge = (min(int(x), int(y)), max(int(x), int(y)))
            if edge in first:
                j = first[edge]
                adjacency[i].add(j)
                adjacency[j].add(i)
            else:
                first[edge] = i
    return adjacency


def select_short_bridges(faces: np.ndarray, baseline: np.ndarray,
                         strong: np.ndarray, weak: np.ndarray,
                         *, max_weak_chain: int = 2) -> tuple[np.ndarray, dict]:
    """Select weak 1/2-face paths between different preexisting strong components.

    Component identities are frozen BEFORE choosing bridges. A weak face
    touching only one component is never allowed to extend a dangling sheet.
    """
    faces = np.asarray(faces)
    baseline = np.asarray(baseline, dtype=bool)
    strong = np.asarray(strong, dtype=bool)
    weak = np.asarray(weak, dtype=bool)
    if (faces.ndim != 2 or faces.shape[1:] != (3,)
            or baseline.shape != (len(faces),) or strong.shape != baseline.shape
            or weak.shape != baseline.shape or max_weak_chain not in (1, 2)
            or np.any(baseline & ~strong) or np.any(strong & ~weak)):
        raise ValueError("Invalid face support masks or bridge length")
    adjacency = _adjacency(faces)
    parent = np.arange(len(faces), dtype=np.int64)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for i in np.flatnonzero(strong):
        for j in adjacency[int(i)]:
            if strong[j]:
                parent[root(int(i))] = root(j)
    component = np.full(len(faces), -1, dtype=np.int64)
    for i in np.flatnonzero(strong):
        component[i] = root(int(i))
    eligible = weak & ~strong
    near = [set(component[j] for j in adjacency[i] if component[j] >= 0)
            for i in range(len(faces))]
    bridges = np.zeros(len(faces), dtype=bool)
    for i in np.flatnonzero(eligible):
        i = int(i)
        if len(near[i]) >= 2:
            bridges[i] = True
        if max_weak_chain == 2 and near[i]:
            for j in adjacency[i]:
                if j <= i or not eligible[j] or not near[j]:
                    continue
                if len(near[i] | near[j]) >= 2:
                    bridges[i] = True
                    bridges[j] = True
    selected = strong | bridges
    attached = _attached_faces(faces[selected], baseline[selected])
    selected[np.flatnonzero(selected)[~attached]] = False
    if not np.all(selected[baseline]):
        raise ValueError("Baseline faces unexpectedly removed")
    return selected, {
        "bridge_faces_selected_before_attachment": int(np.count_nonzero(bridges)),
        "bridge_faces_retained": int(np.count_nonzero(selected & bridges)),
        "strong_extra_faces_retained": int(np.count_nonzero(selected & strong & ~baseline)),
        "total_extra_faces_retained": int(np.count_nonzero(selected & ~baseline)),
        "max_weak_chain_faces": max_weak_chain,
    }


def _independent_midpoint_votes(vertices: np.ndarray, faces: np.ndarray,
                                depths: list[np.ndarray], poses: list[np.ndarray],
                                camera, distinct_baseline_m: float) -> np.ndarray:
    """Three edge-midpoint votes, each requiring two separated inferred-Z views."""
    triangles = vertices[faces]
    midpoints = (triangles + np.roll(triangles, -1, axis=1)) / 2
    xyz_world = midpoints.reshape(-1, 3)
    vote = np.zeros((len(poses), len(faces), 3), dtype=bool)
    for j, (depth, pose) in enumerate(zip(depths, poses)):
        xyz = (xyz_world - pose[:3, 3]) @ pose[:3, :3]
        z = xyz[:, 2]
        safe = np.maximum(z, 1e-8)
        u = np.rint(camera.fx * xyz[:, 0] / safe + camera.cx).astype(np.int64)
        v = np.rint(camera.fy * xyz[:, 1] / safe + camera.cy).astype(np.int64)
        inside = ((z >= .15) & (z <= 4.) & (u >= 0) & (u < camera.width)
                  & (v >= 0) & (v < camera.height))
        sampled = np.zeros(len(z), dtype=np.float32)
        sampled[inside] = depth[v[inside], u[inside]]
        consistent = (inside & (sampled > 0) &
                      (np.abs(sampled - z) <= np.maximum(.10, .04 * z)))
        vote[j] = consistent.reshape(len(faces), 3)
    independent = np.zeros((len(faces), 3), dtype=bool)
    for j in range(len(poses)):
        for k in range(j):
            if np.linalg.norm(poses[j][:3, 3] - poses[k][:3, 3]) >= distinct_baseline_m:
                independent |= vote[j] & vote[k]
    return independent


def filter_bridge_faces(original_vertices: np.ndarray, original_faces: np.ndarray,
                        candidate_vertices: np.ndarray, candidate_faces: np.ndarray,
                        inferred_depths: list[np.ndarray], camera_to_world: list[np.ndarray],
                        camera) -> tuple[np.ndarray, dict]:
    """Extend the independent-centroid filter using supported edges and short bridges.

    The previous centroid filter validates all geometry, depth maps, and rigid
    camera poses. The old Mesh only labels baseline triangles from the same
    frozen TSDF replay; it is never an external reconstruction prior.
    """
    centroid_mask, previous = filter_crossview_faces(
        original_vertices, original_faces, candidate_vertices, candidate_faces,
        inferred_depths, camera_to_world, camera)
    candidate_faces = np.asarray(candidate_faces)
    candidate_vertices = np.asarray(candidate_vertices)
    original_keys = set(_face_keys(original_vertices, original_faces))
    baseline = np.array([key in original_keys for key in
                         _face_keys(candidate_vertices, candidate_faces)], dtype=bool)
    extension = np.flatnonzero(~baseline)
    midpoint = np.zeros((len(candidate_faces), 3), dtype=bool)
    if len(extension):
        midpoint[extension] = _independent_midpoint_votes(
            candidate_vertices, candidate_faces[extension],
            inferred_depths, camera_to_world, camera, .04)
    strong = centroid_mask | (midpoint.sum(axis=1) >= 2) | baseline
    weak = strong | midpoint.any(axis=1)
    selected, stats = select_short_bridges(candidate_faces, baseline, strong, weak)
    stats.update({
        "centroid_extra_faces_after_old_attachment": int(np.count_nonzero(centroid_mask & ~baseline)),
        "two_or_more_independent_midpoints": int(np.count_nonzero((midpoint.sum(axis=1) >= 2) & ~baseline)),
        "midpoint_weak_candidates": int(np.count_nonzero(weak & ~strong)),
        "baseline_faces_preserved": int(np.count_nonzero(baseline)),
        "candidate_extra_faces": int(len(extension)),
        "sensor_depth_used": False,
        "prior_mesh_used_for_reconstruction": False,
        "baseline_mesh_used_for_same_volume_face_label_only": True,
        "centroid_filter": previous,
    })
    return selected, stats
