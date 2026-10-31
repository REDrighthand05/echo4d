"""Intersection-only trim fusion (experiment, user-originated method).

Reproduces echo_scene.fuse.stage_fuse's back-projection exactly (stride 4,
per-frame (a, b) anchor fits, validity + 0.18 depth-edge rejection, per-axis
1-99 percentile trim, deterministic voxel_pick at extent/220), keeping a
source-frame id per point, and inserts ONE extra selection rule before
voxel_pick:

    keep a point only if its voxel -- sized by the spread yardstick
    (extent/110, passed in as --voxel) -- contains points from at least
    K distinct frames.

This deletes unproven points instead of moving them: single-view fringes
and fly-away sheets have no independent corroboration and are cut; walls
and the desktop, seen from many route positions, survive. It is an
evidence threshold, not an average -- no point value is ever blended.

Usage:
    python experiments/trim_fuse.py <job_dir> <K> <yardstick_voxel>
        [--save-baseline-fids <path>]

With K=0 the corroboration filter is skipped (pure replication of
stage_fuse; must reproduce the job's existing points.bin). The final
cloud's per-point frame ids are written to <job_dir>/points_frame_ids.npy,
and a JSON summary is printed.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echo_scene.fuse import (  # noqa: E402
    _collect_anchor_pairs,
    _fit_mad,
    _quat_xyzw_to_matrix,
    _robust_affine_fit,
    _write_binary_ply,
    _write_points_bin,
)

STRIDE = 4


def back_project(job_dir: Path):
    """stage_fuse's dense pass, verbatim except per-point frame ids."""
    poses = json.loads((job_dir / "poses.json").read_text())
    depth_dir = job_dir / "depth"
    frames_dir = job_dir / "frames"

    rel_vals, inv_z, per_frame = _collect_anchor_pairs(job_dir, depth_dir)
    aligned = len(rel_vals) >= 30
    if aligned:
        a_glob, b_glob = _robust_affine_fit(inv_z, rel_vals)
    else:
        a_glob, b_glob = 1.0, 0.0
    frame_fits = {}
    if aligned:
        for stem, (f_rel, f_inv) in per_frame.items():
            if len(f_rel) < 60:
                continue
            a_i, b_i = _robust_affine_fit(f_inv, f_rel)
            if a_i <= 0:
                continue
            if _fit_mad(f_inv, f_rel, a_i, b_i) <= _fit_mad(f_inv, f_rel, a_glob, b_glob):
                frame_fits[stem] = (a_i, b_i)

    all_points, all_colors, all_fids = [], [], []
    frame_index = {}
    used = 0
    for name, pose in sorted(poses.items()):
        stem = Path(name).stem
        npy = depth_dir / f"{stem}.npy"
        img_path = frames_dir / name
        if not npy.exists() or not img_path.exists():
            continue
        a, b = frame_fits.get(stem, (a_glob, b_glob))
        rel = np.load(npy).astype(np.float64)
        if aligned:
            denom = rel - b
            valid = denom > 1e-4
            z_map = np.where(valid, a / np.where(valid, denom, 1.0), np.nan)
            z_hi = np.nanpercentile(z_map[valid], 99) if valid.any() else np.nan
            valid &= z_map < z_hi * 1.5
        else:
            z_map = 1.0 / np.maximum(rel, 1e-3)
            med = np.median(z_map)
            valid = (z_map > med * 0.2) & (z_map < med * 5.0)
        cam = pose["camera"]["params"]
        f, cx, cy = float(cam[0]), float(cam[1]), float(cam[2])
        rot = _quat_xyzw_to_matrix(pose["cam_from_world_quat_xyzw"])
        t = np.array(pose["cam_from_world_t"], dtype=np.float64)

        ys, xs = np.mgrid[0:rel.shape[0]:STRIDE, 0:rel.shape[1]:STRIDE]
        zs = z_map[::STRIDE, ::STRIDE]
        ok = valid[::STRIDE, ::STRIDE] & np.isfinite(zs)
        if zs.shape[0] > 1 and zs.shape[1] > 1:
            jump = np.zeros_like(zs)
            jump[:, :-1] = np.maximum(jump[:, :-1], np.abs(zs[:, 1:] - zs[:, :-1]))
            jump[:-1, :] = np.maximum(jump[:-1, :], np.abs(zs[1:, :] - zs[:-1, :]))
            ok &= jump <= 0.18 * np.maximum(zs, 1e-6)
        if not ok.any():
            continue
        xs_f, ys_f, zs_f = xs[ok].astype(np.float64), ys[ok].astype(np.float64), zs[ok]
        xc = (xs_f - cx) / f * zs_f
        yc = (ys_f - cy) / f * zs_f
        cam_pts = np.stack([xc, yc, zs_f], axis=1)
        world = (cam_pts - t) @ rot
        img = np.asarray(Image.open(img_path).convert("RGB"))
        colors = img[::STRIDE, ::STRIDE][ok]
        fi = frame_index.setdefault(stem, len(frame_index))
        all_points.append(world)
        all_colors.append(colors)
        all_fids.append(np.full(len(world), fi, dtype=np.int32))
        used += 1

    points = np.concatenate(all_points)
    colors = np.concatenate(all_colors)
    fids = np.concatenate(all_fids)
    return points, colors, fids, used, frame_index


def percentile_trim(points, colors, fids):
    """Production per-axis 1-99 percentile trim, verbatim."""
    lo = np.percentile(points, 1, axis=0)
    hi = np.percentile(points, 99, axis=0)
    keep = np.all((points >= lo) & (points <= hi), axis=1)
    if keep.sum() >= 1000:
        return points[keep], colors[keep], fids[keep], lo, hi
    return points, colors, fids, lo, hi


def corroboration_filter(points, colors, fids, voxel, k, lo):
    """Keep points whose yardstick voxel holds >= k distinct frames."""
    keys = np.floor((points - lo) / voxel).astype(np.int64)
    lin = (keys[:, 0] * 73856093) ^ (keys[:, 1] * 19349663) ^ (keys[:, 2] * 83492791)
    vox_id, inverse = np.unique(lin, return_inverse=True)
    n_frames_code = 1024  # > number of frames (230)
    pair_code = inverse.astype(np.int64) * n_frames_code + fids.astype(np.int64)
    uniq_pairs = np.unique(pair_code)
    distinct = np.bincount(uniq_pairs // n_frames_code, minlength=len(vox_id))
    keep = distinct[inverse] >= k
    stats = {
        "yardstick_voxels": int(len(vox_id)),
        "voxels_with_ge_k_frames": int((distinct >= k).sum()),
        "points_before_filter": int(len(points)),
        "points_after_filter": int(keep.sum()),
    }
    return points[keep], colors[keep], fids[keep], stats


def voxel_pick(points, colors, fids):
    """Production deterministic voxel_pick, verbatim (extent/220)."""
    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    voxel = max(extent / 220.0, 1e-6)
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    centroid = np.empty((len(counts), 3), dtype=np.float64)
    for k in range(3):
        centroid[:, k] = np.bincount(inverse, weights=points[:, k]) / counts
    d2 = np.sum((points - centroid[inverse]) ** 2, axis=1)
    order = np.lexsort((np.arange(len(points)), d2, inverse))
    inv_sorted = inverse[order]
    first = np.ones(len(points), dtype=bool)
    first[1:] = inv_sorted[1:] != inv_sorted[:-1]
    pick = np.sort(order[first])
    assert len(pick) == len(counts), (len(pick), len(counts))
    return points[pick], colors[pick], fids[pick]


def main() -> None:
    job_dir = Path(sys.argv[1])
    k_req = int(sys.argv[2])
    yardstick_voxel = float(sys.argv[3])
    save_fids = None
    if "--save-baseline-fids" in sys.argv:
        save_fids = Path(sys.argv[sys.argv.index("--save-baseline-fids") + 1])

    points, colors, fids, used, frame_index = back_project(job_dir)
    n_dense = len(points)
    points, colors, fids, lo, _hi = percentile_trim(points, colors, fids)
    n_trimmed = len(points)

    summary = {
        "job": str(job_dir),
        "frames_used": used,
        "dense_points": int(n_dense),
        "after_percentile_trim": int(n_trimmed),
        "yardstick_voxel": yardstick_voxel,
        "k": k_req,
    }

    if k_req <= 0:
        out_pts, out_cols, out_fids = voxel_pick(points, colors, fids)
        summary["filter"] = None
    else:
        f_pts, f_cols, f_fids, stats = corroboration_filter(
            points, colors, fids, yardstick_voxel, k_req, lo)
        summary["filter"] = stats
        out_pts, out_cols, out_fids = voxel_pick(f_pts, f_cols, f_fids)

        # Baseline (unfiltered) frame ids for the delivered-cloud spread
        # comparison, aligned with production voxel_pick on the same
        # pre-filter cloud.
        if save_fids is not None:
            _bp, _bc, base_fids = voxel_pick(points, colors, fids)
            np.save(save_fids, base_fids)
            summary["baseline_fids_saved"] = str(save_fids)

    _write_binary_ply(out_pts, out_cols, job_dir / "scene_points.ply")
    _write_points_bin(out_pts, out_cols, job_dir / "points.bin")
    np.save(job_dir / "points_frame_ids.npy", out_fids)
    summary["final_points"] = int(len(out_pts))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
