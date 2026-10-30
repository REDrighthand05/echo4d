"""Real Lounge: retain supplied scan PRIOR unless measured camera rays justify replacement.

Comparison: 144-frame measured TSDF, old 6 cm Euclidean carving, and
visibility-aware PRIOR carving. The supplied lounge.ply is an EXISTING scan,
never independent reconstruction. All geometry is from measured RGB-D or
explicitly labeled supplied scan; unknown regions remain unknown.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback

import numpy as np
from PIL import Image

from echo4d.model import Camera, DataError
from lounge_spatial_prior_real import VOXEL_M, read_frames, select_views
from lounge_measured_first_carve_real import carve_scan_around_measurements
from tum_xyz_open3d_320 import export_glb, free_view, fuse, panel, render


def classify_scan_faces(prior_vertices, prior_triangles, measured_vertices,
                        frames, camera, *, agreement_m=.065,
                        free_margin_m=.12, near_m=.085):
    """Classify each PRIOR triangle using real depth observations.

    A triangle is redundant only when its centroid is near measured geometry
    AND two camera rays agree with its depth. It is inconsistent free-space
    geometry only when three distinct views put it clearly in front of valid
    observed depth. Occluded, unobserved, or depth-edge pixels are preserved.

    Projected depth is camera z, NOT Euclidean range. Frames contain
    camera_to_world transforms and metric depth at camera resolution.
    """
    from scipy.ndimage import maximum_filter, minimum_filter
    from scipy.spatial import cKDTree

    if not len(prior_triangles) or not len(measured_vertices):
        raise DataError("Empty PRIOR faces or measured geometry")
    centroids = np.asarray(prior_vertices, dtype=np.float64)[
        np.asarray(prior_triangles, dtype=np.int64)].mean(axis=1)
    dist, _ = cKDTree(np.asarray(measured_vertices)).query(
        centroids, workers=2)
    near = dist < near_m
    agreement = np.zeros(len(centroids), np.uint16)
    freespace = np.zeros(len(centroids), np.uint16)
    occluded = np.zeros(len(centroids), np.uint16)
    seen = np.zeros(len(centroids), np.uint16)

    for frame in frames:
        depth = np.asarray(frame.depth, dtype=np.float32)
        if depth.shape != (camera.height, camera.width):
            raise DataError("Depth shape differs from scaled camera intrinsics")
        valid = np.isfinite(depth) & (depth > .2) & (depth < 4.)
        # Reject depth discontinuities and invalid neighboring pixels; otherwise
        # foreground/background pixel mixing would erase legitimate scan walls.
        lo = minimum_filter(np.where(valid, depth, np.inf), size=3)
        hi = maximum_filter(np.where(valid, depth, -np.inf), size=3)
        stable = valid & np.isfinite(lo) & np.isfinite(hi) & ((hi - lo) < .10)
        world_to_camera = np.linalg.inv(np.asarray(frame.camera_to_world))
        cam = centroids @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        z = cam[:, 2]
        positive = (z > .2) & (z < 4.)
        if not positive.any():
            continue
        inds = np.flatnonzero(positive)
        p = cam[inds]
        u = np.rint(camera.fx * p[:, 0] / p[:, 2] + camera.cx).astype(np.int32)
        v = np.rint(camera.fy * p[:, 1] / p[:, 2] + camera.cy).astype(np.int32)
        inside = ((u >= 1) & (u < camera.width - 1)
                  & (v >= 1) & (v < camera.height - 1))
        inds, u, v = inds[inside], u[inside], v[inside]
        stable_hit = stable[v, u]
        inds, u, v = inds[stable_hit], u[stable_hit], v[stable_hit]
        if not len(inds):
            continue
        measured_z = depth[v, u]
        delta = z[inds] - measured_z
        seen[inds] += 1
        agreement[inds] += (np.abs(delta) <= agreement_m)
        freespace[inds] += (delta < -free_margin_m)
        occluded[inds] += (delta > free_margin_m)

    redundant = near & (agreement >= 2)
    # A single conflicting depth observation cannot delete a PRIOR surface.
    # Require more front-of-depth evidence than behind-object occlusions.
    impossible = (freespace >= 3) & (freespace > occluded)
    keep = ~(redundant | impossible)
    return keep, {
        "faces": int(len(keep)),
        "redundant_near_measured": int(redundant.sum()),
        "contradicted_free_space": int((impossible & ~redundant).sum()),
        "preserved_unseen": int((seen == 0).sum()),
        "preserved_occluded": int((keep & (occluded > 0)).sum()),
        "preserved_prior_faces": int(keep.sum()),
        "camera_views": len(frames),
    }


def visibility_hybrid(o3d, measured, supplied_scan, frames, camera):
    """Measured-first Mesh plus only justified surviving scan PRIOR faces."""
    from scipy.spatial import cKDTree

    scan = o3d.geometry.TriangleMesh(supplied_scan)
    if len(scan.triangles) > 350_000:
        scan = scan.simplify_quadric_decimation(350_000)
    scan.remove_unreferenced_vertices()
    p = np.asarray(scan.vertices)
    m = np.asarray(measured.vertices)
    if not len(p) or not len(m):
        raise DataError("Empty measured or scan mesh")
    m_to_p, _ = cKDTree(p).query(m, workers=2)
    median = float(np.median(m_to_p))
    if not np.isfinite(median) or median > .40:
        raise DataError(f"Measured/scan coordinates disagree: {median:.3f}m")

    faces = np.asarray(scan.triangles).copy()
    keep, stats = classify_scan_faces(p, faces, m, frames, camera)
    # Reject extreme pre-existing scan triangles, not normal hole-filling faces.
    xyz = p[faces]
    longest = np.maximum.reduce([
        np.linalg.norm(xyz[:, 0] - xyz[:, 1], axis=1),
        np.linalg.norm(xyz[:, 1] - xyz[:, 2], axis=1),
        np.linalg.norm(xyz[:, 2] - xyz[:, 0], axis=1)])
    keep &= longest < .30

    # A missing color is not evidence of missing geometry. Unobserved PRIOR
    # stays neutral gray; nearby real measured RGB colors the scan.
    d, nearest = cKDTree(m).query(p, workers=2)
    colors = np.tile([.58, .59, .60], (len(p), 1))
    if measured.has_vertex_colors():
        near = d < .12
        colors[near] = np.asarray(measured.vertex_colors)[nearest[near]]
    scan.vertex_colors = o3d.utility.Vector3dVector(colors)
    scan.triangles = o3d.utility.Vector3iVector(faces[keep])
    scan.remove_unreferenced_vertices()

    hybrid = o3d.geometry.TriangleMesh(measured) + scan
    provenance = o3d.geometry.TriangleMesh(hybrid)
    origin_colors = np.empty((len(hybrid.vertices), 3))
    origin_colors[:len(measured.vertices)] = [.10, .85, .35]
    origin_colors[len(measured.vertices):] = [.95, .43, .13]
    provenance.vertex_colors = o3d.utility.Vector3dVector(origin_colors)
    stats.update(
        retained_after_edge_guard=int(keep.sum()),
        measured_triangles=len(measured.triangles),
        hybrid_triangles=len(hybrid.triangles),
        alignment_median_m=median,
        provenance="green MEASURED RGB-D; orange EXISTING SCAN PRIOR")
    return hybrid, provenance, stats


def experiment(work, output, receipt):
    import open3d as o3d

    work.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    data = o3d.data.LoungeRGBDImages(data_root=str(work))
    params = o3d.io.read_pinhole_camera_trajectory(
        data.trajectory_log_path).parameters
    _, selected, poses, pose_indices = select_views(params, data.color_paths)
    intr = params[pose_indices[0]].intrinsic
    k = intr.intrinsic_matrix
    full = Camera(intr.width, intr.height, float(k[0, 0]), float(k[1, 1]),
                  float(k[0, 2]), float(k[1, 2]))
    if full.width % 2 or full.height % 2:
        raise DataError("Even image dimensions required")
    half = full.scaled(2)
    for i in selected:
        q = params[pose_indices[i]].intrinsic
        if (q.width != full.width or q.height != full.height
                or not np.allclose(q.intrinsic_matrix, k, atol=1e-3)):
            raise DataError("Camera intrinsics differ between frames")
    frames, originals, photos = read_frames(data, selected, poses, full, half)
    measured, _ = fuse(o3d, [frames[i] for i in selected], half,
                       pixel_stride=1, voxel=VOXEL_M)
    supplied = o3d.io.read_triangle_mesh(data.reconstruction_path)
    old, _, old_stats = carve_scan_around_measurements(
        o3d, measured, supplied, .06)
    # These are already spatially diverse RGB-D views, subsampled to cap CPU.
    support_ids = selected[::4]
    if selected[-1] not in support_ids:
        support_ids.append(selected[-1])
    new, provenance, stats = visibility_hybrid(
        o3d, measured, supplied, [frames[i] for i in support_ids], half)
    for name, mesh in (
        ("MEASURED-144-only.glb", measured),
        ("OLD-MEASURED-first-6cm.glb", old),
        ("NEW-VISIBILITY-MEASURED-plus-PRIOR.glb", new),
        ("ORIGIN-green-MEASURED-orange-PRIOR.glb", provenance),
    ):
        export_glb(mesh, output / name)

    labels = ["REAL / FREE", "MEASURED 144", "OLD 6cm CARVE",
              "NEW RAY CARVE", "SOURCE MAP"]
    boards, views = [], []
    for i in photos:
        Image.fromarray(originals[i]).save(output / f"REAL-lounge-{i:04d}.png")
        photo = np.asarray(Image.fromarray(originals[i]).resize(
            (half.width, half.height), Image.Resampling.LANCZOS))
        imgs = [photo] + [render(o3d, mesh, half, poses[i])
                          for mesh in (measured, old, new, provenance)]
        name = f"recorded-{i:04d}.png"
        boards.append(panel(output / name, imgs, labels))
        views.append({"kind": "recorded", "frame": i, "file": name})
    for i, side in ((photos[0], -1.), (photos[-1], 1.)):
        pose = free_view(poses[i], side)
        imgs = [render(o3d, mesh, half, pose)
                for mesh in (measured, old, new, provenance)]
        name = f"free-{i:04d}.png"
        boards.append(panel(output / name,
                            [np.full_like(imgs[0], 20)] + imgs, labels))
        views.append({"kind": "free", "frame": i, "file": name})
    board = Image.new("RGB", (half.width * 5, (half.height + 27) * 5))
    for j, view in enumerate(boards):
        board.paste(view, (0, j * (half.height + 27)))
    board.save(output / "lounge-five-view-ray-carve-vs-6cm.png")
    receipt.update(
        status="real_five_view_exported_review_pending",
        source="Open3D Stanford Lounge real RGB-D plus existing scan PRIOR",
        original_resolution=[full.width, full.height],
        fusion_resolution=[half.width, half.height],
        measured_frames=len(selected), depth_support_frames=len(support_ids),
        measured_triangles=len(measured.triangles),
        old_carve=old_stats, ray_carve=stats, photos=photos, views=views,
        visual_verdict="PENDING HUMAN REVIEW; no face-count proxy",
        free_orbit="FAIL until visually verified",
        issue_74="FAIL unchanged",
        old_desk="FAIL 16.7779cm > 15cm unchanged")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    receipt = {
        "scene": "Stanford Lounge real RGB-D interior",
        "status": "running",
        "run_url": (
            f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/"
            f"{os.getenv('GITHUB_RUN_ID')}" if os.getenv("GITHUB_RUN_ID") else None),
    }
    try:
        experiment(args.work, args.output, receipt)
        code = 0
    except Exception as exc:
        traceback.print_exc()
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        code = 1
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "result.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"status": receipt["status"],
                          "error": receipt.get("error")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
