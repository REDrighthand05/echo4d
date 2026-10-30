"""Stanford Lounge: add REAL measured-depth cameras that see gaps in the 144-view Mesh.

The new cameras are chosen by raycasting the measured-only baseline into
unseen RGB-D camera poses, scoring valid measured depth pixels for which the
baseline has no corresponding surface. No supplied scan is used in either
measured-only reconstruction. The separately exported hybrid explicitly
uses the supplied EXISTING scan as PRIOR.
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
from lounge_visibility_prior_real import visibility_hybrid
from tum_xyz_open3d_320 import camera_matrix, export_glb, free_view, fuse, panel, render


def gap_pixels(depth, rendered_z, *, tolerance_m=.12):
    """Count stable, measured surface pixels absent from the baseline Mesh."""
    from scipy.ndimage import maximum_filter, minimum_filter

    d = np.asarray(depth, dtype=np.float32)
    z = np.asarray(rendered_z, dtype=np.float32)
    if d.shape != z.shape or d.ndim != 2:
        raise ValueError("Measured and rendered depth shapes differ")
    valid = np.isfinite(d) & (d > .2) & (d < 4.)
    lo = minimum_filter(np.where(valid, d, np.inf), size=3)
    hi = maximum_filter(np.where(valid, d, -np.inf), size=3)
    stable = valid & np.isfinite(lo) & np.isfinite(hi) & ((hi - lo) < .18)
    missing = stable & (~np.isfinite(z) | (np.abs(z - d) > tolerance_m))
    return int(missing.sum()), int(stable.sum())


def rank_missing_views(o3d, dataset, poses, pose_indices, params, baseline,
                       full, measured, *, candidate_step=12, extra_count=72):
    """Raycast baseline Mesh against real held-out depth, then select gap views.

    The scan PRIOR is intentionally absent from this selection process.
    Candidate poses and depth are from the supplied real RGB-D dataset.
    """
    probe = full.scaled(8)
    if (probe.width, probe.height) != (80, 60):
        raise DataError("Expected calibrated 640x480 Lounge camera")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(measured))
    intrinsic = o3d.core.Tensor(camera_matrix(probe).astype(np.float32))
    base = set(baseline)
    scored = []
    for i in range(0, len(dataset.depth_paths), candidate_step):
        if i in base:
            continue
        pose = np.asarray(poses[i], dtype=np.float64)
        if not np.isfinite(pose).all() or abs(np.linalg.det(pose[:3, :3]) - 1) > .05:
            continue
        intr = params[pose_indices[i]].intrinsic
        if (intr.width != full.width or intr.height != full.height
                or abs(intr.intrinsic_matrix[0, 0] - full.fx) > .001):
            continue
        with Image.open(dataset.depth_paths[i]) as im:
            depth = np.asarray(im)[::8, ::8].astype(np.float32) / 1000.
        if depth.shape != (probe.height, probe.width):
            raise DataError("Depth probe is not aligned with scaled intrinsics")
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic, o3d.core.Tensor(np.linalg.inv(pose).astype(np.float32)),
            probe.width, probe.height)
        hits = scene.cast_rays(rays)["t_hit"].numpy()
        # t_hit is ray parameter, not necessarily Euclidean distance.
        # Project ray displacement onto the camera forward axis for metric z.
        directions = rays.numpy()[..., 3:6]
        z = hits * np.einsum("hwi,i->hw", directions, pose[:3, 2])
        missing, stable = gap_pixels(depth, z)
        if stable >= 450 and missing >= 40:
            scored.append((i, missing / stable, missing, stable))
        if len(scored) % 60 == 0 and len(scored) > 0 and i % 120 == 0:
            print(f"Gap-scored {i+1}/{len(dataset.depth_paths)} candidate positions", flush=True)
    if len(scored) < extra_count:
        raise DataError(f"Only {len(scored)} valid gap candidates for {extra_count} additions")
    ids = np.asarray([x[0] for x in scored], dtype=np.int32)
    ratios = np.asarray([x[1] for x in scored], dtype=np.float64)
    positions = np.asarray([poses[i][:3, 3] for i in ids])
    forwards = np.asarray([poses[i][:3, 2] for i in ids])
    chosen = []
    used = np.zeros(len(ids), dtype=bool)
    novelty = np.zeros(len(ids), dtype=np.float64)
    for _ in range(extra_count):
        objective = ratios + .12 * np.minimum(np.sqrt(novelty), 1.)
        objective[used] = -np.inf
        j = int(np.argmax(objective))
        if not np.isfinite(objective[j]):
            raise DataError("No remaining gap candidate")
        chosen.append(int(ids[j]))
        used[j] = True
        d = np.sum((positions - positions[j]) ** 2, axis=1)
        d += .25 * np.sum((forwards - forwards[j]) ** 2, axis=1)
        if len(chosen) == 1:
            novelty = d
        else:
            novelty = np.minimum(novelty, d)
        # Suppress virtually identical additional camera poses.
        used |= d < .0008
    report = sorted(scored, key=lambda x: x[1], reverse=True)[:12]
    return sorted(chosen), report, len(scored)


def experiment(work, output, receipt):
    import open3d as o3d

    work.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    print("Loading official real Stanford Lounge RGB-D", flush=True)
    dataset = o3d.data.LoungeRGBDImages(data_root=str(work))
    params = o3d.io.read_pinhole_camera_trajectory(
        dataset.trajectory_log_path).parameters
    _, baseline, poses, pose_indices = select_views(params, dataset.color_paths)
    intr = params[pose_indices[0]].intrinsic
    k = intr.intrinsic_matrix
    full = Camera(intr.width, intr.height, float(k[0, 0]), float(k[1, 1]),
                  float(k[0, 2]), float(k[1, 2]))
    if (full.width, full.height) != (640, 480):
        raise DataError("Unexpected Lounge original resolution")
    half = full.scaled(2)
    frames, _, _ = read_frames(dataset, baseline, poses, full, half)
    baseline_mesh, _ = fuse(o3d, [frames[i] for i in baseline], half,
                            pixel_stride=1, voxel=VOXEL_M)
    print("Baseline measured-only Mesh complete; scoring missing depth", flush=True)
    extra, gap_rank, candidate_count = rank_missing_views(
        o3d, dataset, poses, pose_indices, params, baseline, full, baseline_mesh)
    selected = sorted(set(baseline) | set(extra))
    for i in selected:
        q = params[pose_indices[i]].intrinsic
        if (q.width != full.width or q.height != full.height
                or not np.allclose(q.intrinsic_matrix, k, atol=1e-3)):
            raise DataError("Variable camera intrinsics in added depth views")
    added_frames, _, _ = read_frames(dataset, extra, poses, full, half)
    frames.update(added_frames)
    expanded, _ = fuse(o3d, [frames[i] for i in selected], half,
                       pixel_stride=1, voxel=VOXEL_M)
    supplied = o3d.io.read_triangle_mesh(dataset.reconstruction_path)
    old_support = baseline[::4]
    new_support = selected[::4]
    old_hybrid, _, old_stats = visibility_hybrid(
        o3d, baseline_mesh, supplied, [frames[i] for i in old_support], half)
    new_hybrid, provenance, new_stats = visibility_hybrid(
        o3d, expanded, supplied, [frames[i] for i in new_support], half)
    meshes = (
        ("MEASURED-144-baseline.glb", baseline_mesh),
        ("MEASURED-216-gap-targeted.glb", expanded),
        ("PRIOR-hybrid-144.glb", old_hybrid),
        ("PRIOR-hybrid-216-gap-targeted.glb", new_hybrid),
        ("ORIGIN-216-green-MEASURED-orange-PRIOR.glb", provenance),
    )
    for name, mesh in meshes:
        export_glb(mesh, output / name)
        print(f"Exported {name}: {len(mesh.triangles)} triangles", flush=True)
    photos = [baseline[0], baseline[len(baseline) // 2], baseline[-1]]
    labels = ["REAL / FREE", "MEASURED 144", "MEASURED 216",
              "PRIOR 144", "PRIOR 216"]
    boards, views = [], []
    for i in photos:
        with Image.open(dataset.color_paths[i]) as im:
            original = np.asarray(im.convert("RGB"))
        Image.fromarray(original).save(output / f"REAL-lounge-{i:04d}.png")
        small = np.asarray(Image.fromarray(original).resize(
            (half.width, half.height), Image.Resampling.LANCZOS))
        imgs = [small] + [render(o3d, m, half, poses[i])
                          for m in (baseline_mesh, expanded, old_hybrid, new_hybrid)]
        name = f"recorded-{i:04d}.png"
        boards.append(panel(output / name, imgs, labels))
        views.append({"kind": "recorded", "frame": i, "file": name})
    for i, side in ((photos[0], -1.), (photos[-1], 1.)):
        pose = free_view(poses[i], side)
        imgs = [render(o3d, m, half, pose)
                for m in (baseline_mesh, expanded, old_hybrid, new_hybrid)]
        name = f"free-{i:04d}.png"
        boards.append(panel(output / name, [np.full_like(imgs[0], 20)] + imgs, labels))
        views.append({"kind": "free", "reference_frame": i, "file": name})
    board = Image.new("RGB", (half.width * 5, (half.height + 27) * 5))
    for j, view in enumerate(boards):
        board.paste(view, (0, j * (half.height + 27)))
    board.save(output / "lounge-five-view-gap-targeted-144-vs-216.png")
    receipt.update(
        status="real_meshes_and_five_views_exported_review_pending",
        original_resolution=[full.width, full.height],
        fusion_resolution=[half.width, half.height],
        measured_baseline_frames=len(baseline),
        measured_gap_targeted_frames=len(selected),
        added_depth_frame_ids=extra,
        eligible_gap_candidates=candidate_count,
        largest_baseline_depth_gaps=gap_rank,
        baseline_measured_triangles=len(baseline_mesh.triangles),
        gap_measured_triangles=len(expanded.triangles),
        old_prior=old_stats, new_prior=new_stats,
        photos=photos, views=views,
        provenance="MEASURED=real RGB-D; PRIOR=pre-existing supplied lounge.ply",
        visual_verdict="PENDING HUMAN REVIEW; triangle counts are not quality",
        free_orbit="FAIL until human-reviewed",
        issue_74="FAIL unchanged",
        old_desk="FAIL 16.7779cm > 15cm unchanged")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = {
        "scene": "Real Stanford Lounge RGB-D",
        "source": "Open3D LoungeRGBDImages, calibrated RGB-D and poses; scan separately PRIOR",
        "status": "running",
        "run_url": (f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/"
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
