"""Real Stanford Lounge: wide-trajectory measured RGB-D TSDF and labeled supplied scan PRIOR.

The supplied lounge.ply is an EXISTING reconstruction, not recovered by Echo4D.
48 early-sequence measured frames vs 144 spatially diverse measured frames.
Requires Open3D's ~1.84 GB public LoungeRGBDImages download.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback

import numpy as np
from PIL import Image

from echo4d.model import Camera, DataError, Frame
from tum_xyz_open3d_320 import export_glb, free_view, fuse, panel, render

BASE_FRAMES = 48
EXTRA_FRAMES = 96
VOXEL_M = 0.020


def select_views(params, image_paths):
    """Keep 48 early trajectory views, then add 96 maximally different poses."""
    n = len(image_paths)
    if len(params) not in (n, n + 1):
        raise DataError(f"Camera/image count mismatch: {len(params)} poses, {n} RGB images")
    # Open3D's current Lounge archive names color files 000001..002999.
    # Some trajectory logs include frame zero; align by filename when possible.
    numbers = [int(Path(path).stem) for path in image_paths]
    if len(params) > max(numbers):
        pose_indices = numbers
    else:
        pose_indices = list(range(n))
    poses = [np.linalg.inv(np.asarray(params[i].extrinsic)) for i in pose_indices]
    valid = [i for i, p in enumerate(poses) if np.isfinite(p).all()
             and abs(np.linalg.det(p[:3, :3]) - 1) < .05
             and np.linalg.norm(p[:3, 3]) < 100]
    if len(valid) < BASE_FRAMES + EXTRA_FRAMES:
        raise DataError("Too few reliable Lounge camera poses")
    # First third is deliberately limited: the experiment tests the added
    # spatial coverage, not just more adjacent depth samples.
    early = [i for i in valid if i < n // 3]
    if len(early) < BASE_FRAMES:
        raise DataError("Insufficient first-third views")
    base = [early[i] for i in np.linspace(0, len(early)-1, BASE_FRAMES, dtype=int)]
    candidate = valid[::5]
    selected = list(dict.fromkeys(base))
    positions = np.asarray([p[:3, 3] for p in poses])
    forward = np.asarray([p[:3, 2] for p in poses])
    candidate_pos, candidate_dir = positions[candidate], forward[candidate]
    score = np.full(len(candidate), np.inf)
    for i in selected:
        d = np.sum((candidate_pos - positions[i])**2, axis=1)
        d += .25 * np.sum((candidate_dir - forward[i])**2, axis=1)
        score = np.minimum(score, d)
    while len(selected) < BASE_FRAMES + EXTRA_FRAMES:
        score[np.isin(candidate, selected)] = -1.
        chosen = int(candidate[int(np.argmax(score))])
        if chosen in selected:
            raise DataError("Spatial candidate pool exhausted")
        selected.append(chosen)
        d = np.sum((candidate_pos - positions[chosen])**2, axis=1)
        d += .25 * np.sum((candidate_dir - forward[chosen])**2, axis=1)
        score = np.minimum(score, d)
    return sorted(base), sorted(selected), poses, pose_indices


def read_frames(dataset, chosen, poses, full, half):
    frames = {}
    originals = {}
    photo_indices = [chosen[0], chosen[len(chosen)//2], chosen[-1]]
    for count, i in enumerate(chosen):
        with Image.open(dataset.color_paths[i]) as img:
            color = np.asarray(img.convert("RGB"), dtype=np.uint8)
        with Image.open(dataset.depth_paths[i]) as img:
            depth = np.asarray(img)
        if color.shape != (full.height, full.width, 3) or depth.shape != (full.height, full.width):
            raise DataError(f"Original RGB-D image {i} mismatches camera calibration")
        if depth.dtype.kind not in "ui":
            raise DataError("Expected integer sensor depth in millimeters")
        metric = np.ascontiguousarray(depth[::2, ::2].astype(np.float32) / 1000.)
        metric[(metric < .2) | (metric > 4.)] = 0.
        if np.count_nonzero(metric) < 1500:
            raise DataError(f"Insufficient valid measured depth at frame {i}")
        frames[i] = Frame(t=float(i), camera_to_world=poses[i],
                          depth=metric,
                          color=np.ascontiguousarray(color[::2, ::2])).validate(half)
        if i in photo_indices:
            originals[i] = color
        if (count + 1) % 36 == 0:
            print(f"Read {count+1}/{len(chosen)} real Lounge RGB-D frames", flush=True)
    return frames, originals, photo_indices


def prior_assisted(o3d, measured, scan):
    """Keep supplied geometry as PRIOR; add only well-separated measured faces."""
    from scipy.spatial import cKDTree
    if len(scan.triangles) < 1000:
        raise DataError("Missing pre-existing Lounge scan mesh")
    if len(scan.triangles) > 350_000:
        scan = scan.simplify_quadric_decimation(350_000)
    scan.remove_unreferenced_vertices()
    p = np.asarray(scan.vertices)
    m = np.asarray(measured.vertices)
    if not len(p) or not len(m):
        raise DataError("Empty prior or measured mesh")
    # Both meshes should share the dataset's supplied trajectory coordinates.
    dist, nearest = cKDTree(p).query(m, workers=2)
    median = float(np.median(dist))
    if median > .40:
        raise DataError(f"Supplied scan and measured poses appear misaligned: {median:.3f} m")
    if not scan.has_vertex_colors():
        scan_color = np.tile([.58, .59, .60], (len(p), 1))
        p_dist, p_nearest = cKDTree(m).query(p, workers=2)
        good = p_dist < .09
        scan_color[good] = np.asarray(measured.vertex_colors)[p_nearest[good]]
        scan.vertex_colors = o3d.utility.Vector3dVector(scan_color)
    faces = np.asarray(measured.triangles)
    xyz = m[faces]
    edges = np.stack([np.linalg.norm(xyz[:,0]-xyz[:,1], axis=1),
                      np.linalg.norm(xyz[:,1]-xyz[:,2], axis=1),
                      np.linalg.norm(xyz[:,2]-xyz[:,0], axis=1)])
    keep = np.all(dist[faces] > .07, axis=1) & (edges.max(axis=0) < .16)
    used, inverse = np.unique(faces[keep], return_inverse=True)
    supplement = o3d.geometry.TriangleMesh()
    if len(used):
        supplement.vertices = o3d.utility.Vector3dVector(m[used])
        supplement.triangles = o3d.utility.Vector3iVector(inverse.reshape(-1, 3))
        supplement.vertex_colors = o3d.utility.Vector3dVector(
            np.asarray(measured.vertex_colors)[used])
    hybrid = scan + supplement
    provenance = o3d.geometry.TriangleMesh(hybrid)
    colors = np.empty((len(hybrid.vertices), 3))
    colors[:len(scan.vertices)] = [.95, .43, .13]  # PRIOR
    colors[len(scan.vertices):] = [.10, .85, .35]  # MEASURED
    provenance.vertex_colors = o3d.utility.Vector3dVector(colors)
    return scan, hybrid, provenance, dict(
        measured_to_prior_median_m=median,
        prior_triangles=len(scan.triangles),
        added_measured_triangles=int(keep.sum()),
        hybrid_triangles=len(hybrid.triangles))


def experiment(work, output, receipt):
    import open3d as o3d
    work.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    print("Downloading real Stanford Lounge RGB-D (~1.84GB archive)", flush=True)
    data = o3d.data.LoungeRGBDImages(data_root=str(work))
    trajectory = o3d.io.read_pinhole_camera_trajectory(data.trajectory_log_path)
    params = trajectory.parameters
    base, selected, poses, pose_indices = select_views(params, data.color_paths)
    intrinsic = params[pose_indices[0]].intrinsic
    k = intrinsic.intrinsic_matrix
    full = Camera(intrinsic.width, intrinsic.height,
                  float(k[0,0]), float(k[1,1]), float(k[0,2]), float(k[1,2]))
    if full.width % 2 or full.height % 2:
        raise DataError("Lounge camera dimensions must be even")
    half = full.scaled(2)
    for i in selected:
        q = params[pose_indices[i]].intrinsic
        if q.width != full.width or q.height != full.height or not np.allclose(
                q.intrinsic_matrix, k, atol=1e-3):
            raise DataError("Variable per-frame intrinsics require a different fusion path")
    frames, originals, photos = read_frames(data, selected, poses, full, half)
    early, _ = fuse(o3d, [frames[i] for i in base], half,
                    pixel_stride=1, voxel=VOXEL_M)
    print("Early-only measured triangles:", len(early.triangles), flush=True)
    wide, _ = fuse(o3d, [frames[i] for i in selected], half,
                   pixel_stride=1, voxel=VOXEL_M)
    print("Wide-coverage measured triangles:", len(wide.triangles), flush=True)
    scan = o3d.io.read_triangle_mesh(data.reconstruction_path)
    prior, hybrid, provenance, counts = prior_assisted(o3d, wide, scan)
    for name, mesh in (("EARLY-measured-only.glb", early),
                       ("WIDE-measured-only.glb", wide),
                       ("EXISTING-PRIOR-only.glb", prior),
                       ("PRIOR-assisted-with-measured.glb", hybrid),
                       ("ORIGIN-orange-PRIOR-green-MEASURED.glb", provenance)):
        export_glb(mesh, output / name)
    boards, views = [], []
    for i in photos:
        Image.fromarray(originals[i]).save(output / f"REAL-lounge-{i:04d}.png")
        photo = np.asarray(Image.fromarray(originals[i]).resize(
            (half.width, half.height), Image.Resampling.LANCZOS))
        images = [photo, render(o3d, early, half, poses[i]),
                  render(o3d, wide, half, poses[i]),
                  render(o3d, hybrid, half, poses[i])]
        name = f"recorded-{i:04d}.png"
        boards.append(panel(output / name, images,
                            ["REAL RGB", "EARLY 48", "WIDE 144", "WIDE + PRIOR"]))
        views.append({"kind": "recorded", "frame": i, "file": name})
    for i, side in ((photos[0], -1.), (photos[-1], 1.)):
        pose = free_view(poses[i], side)
        a = render(o3d, early, half, pose)
        b = render(o3d, wide, half, pose)
        c = render(o3d, hybrid, half, pose)
        name = f"free-{i:04d}.png"
        boards.append(panel(output / name, [np.full_like(a, 20), a, b, c],
                            ["FREE / NO PHOTO", "EARLY 48", "WIDE 144", "WIDE + PRIOR"]))
        views.append({"kind": "free", "frame": i, "file": name})
    board = Image.new("RGB", (half.width * 4, (half.height + 27) * 5))
    for j, view in enumerate(boards):
        board.paste(view, (0, j * (half.height + 27)))
    board.save(output / "lounge-five-view-spatial-vs-PRIOR.png")
    receipt.update(
        status="real_five_view_exported_review_pending",
        original_resolution=[full.width, full.height],
        fusion_resolution=[half.width, half.height],
        base_frames=len(base), wide_frames=len(selected),
        early_measured_triangles=len(early.triangles),
        wide_measured_triangles=len(wide.triangles),
        prior=counts,
        provenance="PRIOR = supplied lounge.ply; green supplemental geometry = measured RGB-D",
        photo_indices=photos, views=views,
        visual_verdict="PENDING HUMAN REVIEW; face counts are not quality",
        free_orbit="FAIL until human-reviewed",
        issue_74="FAIL unchanged", old_desk="FAIL 16.7779cm > 15cm unchanged")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = {
        "scene": "Real Stanford Lounge RGB-D interior",
        "source": "Open3D LoungeRGBDImages: 2999/3000 real RGB-D frames, poses, supplied scan",
        "status": "running",
        "run_url": (f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/"
                    f"{os.getenv('GITHUB_RUN_ID')}" if os.getenv("GITHUB_RUN_ID") else None)}
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
                          "early": receipt.get("early_measured_triangles"),
                          "wide": receipt.get("wide_measured_triangles"),
                          "error": receipt.get("error")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
