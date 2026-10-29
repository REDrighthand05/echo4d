"""Conservative Stanford room-shell extrapolation from measured RGB-D TSDF.

Added geometry is inferred, never a sensor observation or pre-existing scan.
Run on the native 480-frame measured-only Stanford GLB.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def fit_room_planes(vertices, normals, *, seed=20261029, max_planes=7):
    """Normal-aware RANSAC; reject small or sparse furniture-like planes."""
    xyz = np.asarray(vertices, dtype=np.float64)
    n = np.asarray(normals, dtype=np.float64)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or n.shape != xyz.shape
            or len(xyz) < 1500 or not np.isfinite(xyz).all()
            or not np.isfinite(n).all()):
        raise ValueError("Expected finite measured vertices and normals")
    rng = np.random.default_rng(seed)
    active = np.ones(len(xyz), dtype=bool)
    planes = []
    for _ in range(max_planes):
        ids = np.flatnonzero(active)
        if len(ids) < 1000:
            break
        sample = rng.choice(ids, size=min(len(ids), 20000), replace=False)
        pts = xyz[sample]
        dirs = n[sample]
        best = (0, None, None)
        for j in rng.choice(len(sample), size=min(250, len(sample)), replace=False):
            normal = dirs[j]
            if np.max(np.abs(normal)) < .83:
                continue
            count = int(np.count_nonzero(
                (np.abs((pts - pts[j]) @ normal) < .04)
                & (np.abs(dirs @ normal) > .80)))
            if count > best[0]:
                best = (count, normal, pts[j])
        if best[0] < 500:
            break
        normal, center = best[1:]
        support = np.empty(0, dtype=int)
        for _ in range(3):
            local = xyz[ids] - center
            good = ((np.abs(local @ normal) < .045)
                    & (np.abs(n[ids] @ normal) > .75))
            support = ids[good]
            if len(support) < 500:
                break
            center = xyz[support].mean(axis=0)
            _, _, basis = np.linalg.svd(xyz[support] - center, full_matrices=False)
            normal = basis[-1]
        if len(support) < 500:
            break
        active[support] = False
        uv = (xyz[support] - center) @ basis[:2].T
        span = np.quantile(uv, .98, axis=0) - np.quantile(uv, .02, axis=0)
        area = float(np.prod(span))
        density = len(support) / max(area, .001)
        if len(support) >= 4000 and min(span) > 1.2 and area > 5.0 and density > 220:
            planes.append((center, normal, basis[:2], support, span))
    return planes


def extend_plane(vertices, colors, plane, tree, *,
                 grid=.10, close_m=.75, edge_m=.55, near_m=1.20):
    """Small external strips + interior holes, constrained to measured support."""
    center, normal, basis, ids, span = plane
    uv = (vertices[ids] - center) @ basis.T
    pad = edge_m + grid * 2
    lo = uv.min(axis=0) - pad
    shape = np.ceil((uv.max(axis=0) - uv.min(axis=0) + 2 * pad) / grid).astype(int) + 1
    if np.prod(shape) > 350000:
        raise ValueError("Room grid budget exceeded")
    cell = np.floor((uv - lo) / grid).astype(int)
    cell = np.clip(cell, 0, shape - 1)
    occupied = np.zeros(tuple(shape), dtype=bool)
    occupied[cell[:, 0], cell[:, 1]] = True

    def disk(radius):
        r = int(np.ceil(radius / grid))
        x, y = np.ogrid[-r:r+1, -r:r+1]
        return (x*x + y*y) * grid * grid <= radius*radius + 1e-10

    closed = ndimage.binary_closing(occupied, structure=disk(close_m))
    envelope = ndimage.binary_dilation(closed | occupied, structure=disk(edge_m))
    candidate = np.argwhere(envelope & ~occupied)
    if not len(candidate):
        return np.empty((0, 3)), np.empty((0, 3), int), np.empty((0, 3)), 0
    candidate_uv = lo + (candidate + .5) * grid
    planar_tree = cKDTree(uv)
    distance, _ = planar_tree.query(candidate_uv)
    world = center + candidate_uv @ basis
    other_distance, _ = tree.query(world)
    keep = (distance <= near_m) & (other_distance >= grid * .70)
    candidate = candidate[keep]
    if not len(candidate):
        return np.empty((0, 3)), np.empty((0, 3), int), np.empty((0, 3)), 0
    mask = np.zeros_like(occupied)
    mask[candidate[:, 0], candidate[:, 1]] = True
    neighbours = ndimage.convolve(
        mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant")
    candidate = candidate[neighbours[candidate[:, 0], candidate[:, 1]] >= 3]
    if not len(candidate):
        return np.empty((0, 3)), np.empty((0, 3), int), np.empty((0, 3)), 0
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=int)
    unique, inv = np.unique(
        (candidate[:, None, :] + corners).reshape(-1, 2),
        axis=0, return_inverse=True)
    verts = center + (lo + unique * grid) @ basis
    quads = inv.reshape(-1, 4)
    faces = np.concatenate(
        (quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]), axis=0)
    _, closest = planar_tree.query(lo + unique * grid)
    rgb = colors[ids][closest]
    return verts, faces, rgb, len(candidate)



def depth_conflict_mask(vertices, faces, samples, k, *, tolerance=.10,
                        votes=2, min_valid_views=0):
    """Carve inferred triangles against distinct measured-depth camera views.

    Check seven spatial locations per triangle, not only its centroid: a thin
    foreground false wall may cover a chair at its corner while its centroid
    happens to project onto the correct background. A contradiction must be
    repeated in independent recorded frames. Optionally reject triangles
    without two recorded views with usable depth (unknown is not safe).
    """
    w, h, fx, fy, cx, cy = k
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or
            faces.ndim != 2 or faces.shape[1] != 3 or
            not np.isfinite(vertices).all() or
            np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError("Expected finite triangle geometry")
    if votes < 1 or min_valid_views < 0:
        raise ValueError("Invalid recorded-view thresholds")
    weights = np.array([
        [1/3, 1/3, 1/3],
        [.75, .125, .125], [.125, .75, .125], [.125, .125, .75],
        [.5, .5, 0], [0, .5, .5], [.5, 0, .5]], dtype=np.float64)
    points = np.einsum("fvc,sv->fsc", vertices[faces], weights)
    face_count, sample_count = points.shape[:2]
    points = points.reshape(-1, 3)
    conflicts = np.zeros((face_count, sample_count), dtype=np.uint16)
    valid_views = np.zeros(face_count, dtype=np.uint16)
    tested = 0
    duplicates = 0
    identities = {}
    observations = set()
    for source_id, pose, raw in samples:
        pose = np.asarray(pose, dtype=np.float64)
        if not isinstance(source_id, (str, int)) or isinstance(source_id, bool):
            raise ValueError("Recorded source frame identity is required")
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("Recorded camera pose must be finite 4x4")
        if raw.shape != (h, w) or raw.dtype != np.uint16:
            raise ValueError("Recorded depth must be native uint16")
        digest = hashlib.sha256(pose.tobytes() + raw.tobytes()).hexdigest()
        if source_id in identities and identities[source_id] != digest:
            raise ValueError("One recorded frame identity has conflicting data")
        identities[source_id] = digest
        if digest in observations:
            duplicates += 1
            continue
        observations.add(digest)
        camera = points @ pose[:3, :3].T + pose[:3, 3]
        z = camera[:, 2]
        good = (z > .35) & (z < 3.95)
        safe = np.where(good, z, 1.)
        u = np.rint(fx * camera[:, 0] / safe + cx).astype(int)
        v = np.rint(fy * camera[:, 1] / safe + cy).astype(int)
        good &= (u >= 2) & (u < w-2) & (v >= 2) & (v < h-2)
        u = np.clip(u, 0, w-1)
        v = np.clip(v, 0, h-1)
        observed = raw[v, u].astype(float) / 1000.
        good &= (observed > .35) & (observed < 3.95)
        visible = good & (z <= observed + tolerance)
        valid_views += np.any(visible.reshape(face_count, sample_count), axis=1).astype(np.uint16)
        conflicts += (good & (z + tolerance < observed)).reshape(
            face_count, sample_count).astype(np.uint16)
        tested += 1
    contradicts_depth = np.any(conflicts >= votes, axis=1)
    lacks_visibility = valid_views < min_valid_views
    rejected = contradicts_depth | lacks_visibility
    return rejected, {
        "tested_recorded_frames": tested,
        "duplicate_recorded_frames": duplicates,
        "triangle_depth_samples": sample_count,
        "min_valid_recorded_views": min_valid_views,
        "vetoed_conflicting_faces": int(contradicts_depth.sum()),
        "vetoed_unobserved_faces": int(lacks_visibility.sum()),
        "vetoed_faces": int(rejected.sum())}


def recorded_depth_samples(cache):
    import open3d as o3d
    from stanford_lounge_second_room import lounge_recorded_poses
    from open3d_official_room_baseline import spaced_real_frames
    data = o3d.data.LoungeRGBDImages(data_root=str(cache))
    trajectory = o3d.io.read_pinhole_camera_trajectory(data.trajectory_log_path)
    k, poses, _ = lounge_recorded_poses(
        trajectory, list(data.color_paths), data.trajectory_log_path)
    indices = spaced_real_frames(
        len(data.color_paths), min(240, len(data.color_paths)))
    return k, [
        (f"stanford-lounge-depth:{i}", np.asarray(poses[i], dtype=float),
         np.asarray(o3d.io.read_image(data.depth_paths[i])))
        for i in indices]

def reconstruct(source: Path, output: Path, cache: Path | None = None):
    import trimesh
    if output.exists():
        raise FileExistsError("Fresh output folder required")
    observed = trimesh.load(str(source), force="mesh", process=False)
    xyz = np.asarray(observed.vertices)
    rgba = np.asarray(observed.visual.vertex_colors)
    if len(observed.faces) < 100000 or rgba.shape != (len(xyz), 4):
        raise ValueError("Requires native observed-only colored Stanford GLB")
    rgb = rgba[:, :3].astype(float) / 255.
    planes = fit_room_planes(xyz, np.asarray(observed.vertex_normals))
    tree = cKDTree(xyz)
    verts, faces, colors, records, offset = [], [], [], [], 0
    for center, normal, basis, ids, span in planes:
        v, f, c, count = extend_plane(
            xyz, rgb, (center, normal, basis, ids, span), tree)
        records.append({
            "support_vertices": len(ids), "span_m": span.tolist(),
            "normal": normal.tolist(), "inferred_cells": count})
        if len(f):
            verts.append(v)
            faces.append(f + offset)
            colors.append(c)
            offset += len(v)
    if not faces:
        raise RuntimeError("No supported room-shell extension; retain measured Mesh")
    v = np.concatenate(verts)
    f = np.concatenate(faces)
    c = np.concatenate(colors)
    depth_veto = {"state": "NOT RUN"}
    if cache is not None:
        k, samples = recorded_depth_samples(cache)
        rejected, depth_veto = depth_conflict_mask(v, f, samples, k, min_valid_views=2)
        f = f[~rejected]
        if not len(f):
            raise RuntimeError("All inferred faces conflict with recorded depth")
    rgba_patch = np.column_stack((
        np.rint(np.clip(c, 0, 1) * 255).astype(np.uint8),
        np.full(len(v), 255, dtype=np.uint8)))
    patch = trimesh.Trimesh(
        vertices=v, faces=f, vertex_colors=rgba_patch, process=False)
    combined = trimesh.util.concatenate([observed, patch])
    provenance = trimesh.util.concatenate([
        trimesh.Trimesh(
            vertices=xyz, faces=observed.faces,
            vertex_colors=np.tile([24, 184, 190, 255], (len(xyz), 1)),
            process=False),
        trimesh.Trimesh(
            vertices=v, faces=f,
            vertex_colors=np.tile([255, 145, 35, 255], (len(v), 1)),
            process=False)])
    output.mkdir(parents=True)
    patch.export(output / "INFERRED-room-envelope-only.glb")
    combined.export(output / "OBSERVED-plus-INFERRED-room-envelope.glb")
    provenance.export(output / "OBSERVED-vs-INFERRED-room-envelope.glb")
    result = {
        "scene": "Stanford Lounge measured 480-frame 640x480 RGB-D TSDF",
        "observed_faces": int(len(observed.faces)),
        "observed_triangles": int(len(observed.faces)),
        "inferred_faces": int(len(f)),
        "combined_faces": int(len(combined.faces)),
        "planes": records,
        "recorded_depth_conflict_veto": depth_veto,
        "preexisting_mesh_prior_used": False,
        "inferred_surfaces_are_not_measured": True,
        "visual_acceptance": "NOT REVIEWED",
        "issue_74": "FAIL unchanged",
        "desk_v2": "FAIL 16.7779cm > 15cm unchanged"}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path)
    a = p.parse_args()
    reconstruct(a.source, a.output, a.cache)


if __name__ == "__main__":
    main()

