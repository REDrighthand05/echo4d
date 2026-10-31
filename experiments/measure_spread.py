"""Cross-frame spread measurement for an arbitrary job (diagnostic).

Adapted from experiments/prep_refine.py's BEFORE-spread computation,
parameterized by job dir: sample every frame's depth map (stride 8,
fuse's validity + edge rules), back-project with the job's CURRENT
per-frame (a, b) fits and poses, group into voxels at extent/110, and
report the RMS distance to the voxel centroid over points in voxels
containing >= 2 distinct frames. Same code + params for both jobs, so
room2 vs room2_ba numbers are directly comparable.

Usage: python experiments/measure_spread.py <job_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_refine import _collect_anchors_with_pixels  # noqa: E402
from echo_scene.fuse import (  # noqa: E402
    _fit_mad,
    _quat_xyzw_to_matrix,
    _robust_affine_fit,
)

STRIDE = 8


def measure(job: Path) -> dict:
    poses = json.loads((job / "poses.json").read_text())
    depth_dir = job / "depth"
    frames_dir = job / "frames"

    per_frame_px = _collect_anchors_with_pixels(job, depth_dir)
    per_frame = {k: (v[0], v[1]) for k, v in per_frame_px.items()}
    rel_vals = np.concatenate([v[0] for v in per_frame.values()])
    inv_z = np.concatenate([v[1] for v in per_frame.values()])
    a_glob, b_glob = _robust_affine_fit(inv_z, rel_vals)
    frame_fits = {}
    for stem, (f_rel, f_inv) in per_frame.items():
        if len(f_rel) < 60:
            continue
        a_i, b_i = _robust_affine_fit(f_inv, f_rel)
        if a_i <= 0:
            continue
        if _fit_mad(f_inv, f_rel, a_i, b_i) <= _fit_mad(f_inv, f_rel, a_glob, b_glob):
            frame_fits[stem] = (a_i, b_i)

    ray_list, rel_list, off_list, fit_list, idx_list = [], [], [], [], []
    frame_stems = []
    for name, pose in sorted(poses.items()):
        stem = Path(name).stem
        npy = depth_dir / f"{stem}.npy"
        img_path = frames_dir / name
        if not npy.exists() or not img_path.exists():
            continue
        a, b = frame_fits.get(stem, (a_glob, b_glob))
        rel = np.load(npy).astype(np.float64)
        denom = rel - b
        valid = denom > 1e-4
        z_map = np.where(valid, a / np.where(valid, denom, 1.0), np.nan)
        if valid.any():
            z_hi = np.nanpercentile(z_map[valid], 99)
            valid &= z_map < z_hi * 1.5
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
        xs_f = xs[ok].astype(np.float64)
        ys_f = ys[ok].astype(np.float64)
        ray_cam = np.stack([(xs_f - cx) / f, (ys_f - cy) / f, np.ones_like(xs_f)], axis=1)
        fi = len(frame_stems)
        frame_stems.append(stem)
        ray_list.append(ray_cam @ rot)
        rel_list.append(rel[::STRIDE, ::STRIDE][ok])
        off_list.append(-t @ rot)
        fit_list.append((a, b))
        idx_list.append(np.full(int(ok.sum()), fi, dtype=np.int32))

    rays = np.concatenate(ray_list)
    rels = np.concatenate(rel_list)
    offs = np.stack(off_list)
    fits = np.array(fit_list)
    fidx = np.concatenate(idx_list)
    pts = (fits[fidx, 0] / (rels - fits[fidx, 1]))[:, None] * rays + offs[fidx]

    lo_t = np.percentile(pts, 1, axis=0)
    hi_t = np.percentile(pts, 99, axis=0)
    keep = np.all((pts >= lo_t) & (pts <= hi_t), axis=1)
    rays, rels, fidx, pts = rays[keep], rels[keep], fidx[keep], pts[keep]

    lo = np.percentile(pts, 1, axis=0)
    hi = np.percentile(pts, 99, axis=0)
    extent = float(np.linalg.norm(hi - lo))
    voxel = extent / 110.0
    keys = np.floor((pts - lo) / voxel).astype(np.int64)
    lin = (keys[:, 0] * 73856093) ^ (keys[:, 1] * 19349663) ^ (keys[:, 2] * 83492791)
    vox_id, inverse = np.unique(lin, return_inverse=True)
    counts = np.bincount(inverse)
    order = np.argsort(inverse, kind="stable")
    inv_s = inverse[order]
    fi_s = fidx[order]
    starts = np.flatnonzero(np.r_[True, inv_s[1:] != inv_s[:-1]])
    multi = np.zeros(len(vox_id), dtype=bool)
    for s_i, s in enumerate(starts):
        e = starts[s_i + 1] if s_i + 1 < len(starts) else len(inv_s)
        if e - s >= 2 and len(np.unique(fi_s[s:e])) >= 2:
            multi[inv_s[s]] = True
    in_multi = multi[inverse]
    cent = np.zeros((len(vox_id), 3))
    for k in range(3):
        cent[:, k] = np.bincount(inverse, weights=pts[:, k]) / counts
    d = pts[in_multi] - cent[inverse[in_multi]]
    spread = float(np.sqrt(np.mean(np.sum(d * d, axis=1))))
    return {
        "job": str(job),
        "frames": len(frame_stems),
        "sampled_points": int(len(pts)),
        "voxels": int(len(vox_id)),
        "cross_frame_voxels": int(multi.sum()),
        "points_in_cross_frame_voxels": int(in_multi.sum()),
        "voxel_size": voxel,
        "extent": extent,
        "cross_frame_spread_rms": spread,
    }


if __name__ == "__main__":
    result = measure(Path(sys.argv[1]))
    print(json.dumps(result, indent=2))
