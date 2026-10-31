"""Global least-squares refinement, step A: data prep + diagnostic.

Samples every frame's depth map once (stride 8, fuse.py's own validity and
edge rules), back-projects with the CURRENT per-frame fits, groups points
into fixed voxels, and measures how far apart different frames' surfaces
sit inside the same voxel (cross-frame spread). That spread is what
jointly refining all frames' (a, b) depth-alignment params can reduce:
anchors pin each frame to the sparse points, voxel terms pin frames to
each other. The optimizer (step B) consumes refine_data.npz.
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
)


def _collect_anchors_with_pixels(job_dir, depth_dir):
    """Same sampling as echo_scene.fuse._collect_anchor_pairs, plus the
    anchor pixel coordinates (needed to evaluate a per-frame field)."""
    import pycolmap
    from echo_scene.fuse import _quat_xyzw_to_matrix as q2m
    sfm_dir = job_dir / "sfm"
    models = [d for d in sfm_dir.iterdir()
              if d.is_dir() and d.name.isdigit() and (d / "cameras.bin").exists()]
    rec = max((pycolmap.Reconstruction(str(d)) for d in models),
              key=lambda r: r.num_reg_images())
    per_frame = {}
    depth_cache = {}
    for image in rec.images.values():
        has_pose = image.has_pose() if callable(image.has_pose) else image.has_pose
        if not has_pose:
            continue
        stem = Path(image.name).stem
        npy = depth_dir / f"{stem}.npy"
        if not npy.exists():
            continue
        if stem not in depth_cache:
            depth_cache[stem] = np.load(npy)
        depth = depth_cache[stem]
        h, w = depth.shape
        cfw = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
        rot = q2m(cfw.rotation.quat)
        t = np.array([float(v) for v in cfw.translation], dtype=np.float64)
        f_rel, f_inv, f_px, f_py = [], [], [], []
        for p2d in image.points2D:
            if p2d.point3D_id < 0 or p2d.point3D_id not in rec.points3D:
                continue
            xyz = np.array(rec.points3D[p2d.point3D_id].xyz, dtype=np.float64)
            xc = rot @ xyz + t
            z = xc[2]
            if z <= 0.05:
                continue
            px, py = int(round(p2d.xy[0])), int(round(p2d.xy[1]))
            if 0 <= px < w and 0 <= py < h:
                f_rel.append(float(depth[py, px]))
                f_inv.append(1.0 / z)
                f_px.append(px / max(w - 1, 1))
                f_py.append(py / max(h - 1, 1))
        if f_rel:
            per_frame[stem] = (np.array(f_rel), np.array(f_inv),
                               np.array(f_px), np.array(f_py))
    return per_frame

JOB = Path("jobs/room1_big")
STRIDE = 8
OUT = JOB / "refine"


def main():
    OUT.mkdir(exist_ok=True)
    poses = json.loads((JOB / "poses.json").read_text())
    depth_dir = JOB / "depth"
    frames_dir = JOB / "frames"

    per_frame_px = _collect_anchors_with_pixels(JOB, depth_dir)
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
    print(f"anchors {len(rel_vals)}, global a={a_glob:.4f} b={b_glob:.4f}, "
          f"per-frame fits {len(frame_fits)}", flush=True)

    ray_list, rel_list, col_list, off_list, fit_list, idx_list = [], [], [], [], [], []
    uv_list = []
    anchors = {}  # stem -> (rel array, z_colmap array)
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
        uv_list.append(np.stack([xs_f / max(rel.shape[1] - 1, 1),
                                 ys_f / max(rel.shape[0] - 1, 1)], axis=1))
        rel_list.append(rel[::STRIDE, ::STRIDE][ok])
        img = np.asarray(Image.open(img_path).convert("RGB"))
        col_list.append(img[::STRIDE, ::STRIDE][ok])
        off_list.append(-t @ rot)
        fit_list.append((a, b))
        idx_list.append(np.full(int(ok.sum()), fi, dtype=np.int32))
        if stem in per_frame:
            f_rel, f_inv = per_frame[stem]
            anchors[stem] = (f_rel, 1.0 / f_inv)

    rays = np.concatenate(ray_list)
    uvs = np.concatenate(uv_list)
    rels = np.concatenate(rel_list)
    cols = np.concatenate(col_list)
    offs = np.stack(off_list)
    fits = np.array(fit_list)
    fidx = np.concatenate(idx_list)
    pts = (fits[fidx, 0] / (rels - fits[fidx, 1]))[:, None] * rays + offs[fidx]
    print(f"sampled points {len(pts)} from {len(frame_stems)} frames", flush=True)
    # Outlier trim BEFORE anything is saved (same 1-99 percentile box fuse
    # itself applies): a few % of back-projected points sit hundreds of
    # units away and would dominate a squared-loss objective.
    lo_t = np.percentile(pts, 1, axis=0)
    hi_t = np.percentile(pts, 99, axis=0)
    keep = np.all((pts >= lo_t) & (pts <= hi_t), axis=1)
    print(f"outlier trim keeps {int(keep.sum())}/{len(pts)}", flush=True)
    rays, rels, cols, fidx, pts = rays[keep], rels[keep], cols[keep], fidx[keep], pts[keep]
    uvs = uvs[keep]

    # robust extent (same trim philosophy as fuse)
    lo = np.percentile(pts, 1, axis=0)
    hi = np.percentile(pts, 99, axis=0)
    extent = float(np.linalg.norm(hi - lo))
    voxel = extent / 110.0
    keys = np.floor((pts - lo) / voxel).astype(np.int64)
    lin = (keys[:, 0] * 73856093) ^ (keys[:, 1] * 19349663) ^ (keys[:, 2] * 83492791)
    vox_id, inverse = np.unique(lin, return_inverse=True)
    counts = np.bincount(inverse)
    # voxels containing >= 2 distinct frames
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
    print(f"voxels {len(vox_id)}, cross-frame voxels {int(multi.sum())}, "
          f"points in them {int(in_multi.sum())}", flush=True)
    print(f"BEFORE cross-frame spread (RMS dist to voxel centroid): {spread:.4f} "
          f"(voxel {voxel:.4f})", flush=True)

    anch_frame, anch_rel, anch_z, anch_uv = [], [], [], []
    stem_to_fi = {s: i for i, s in enumerate(frame_stems)}
    for stem, (r, z) in anchors.items():
        px_uv = per_frame_px[stem][2:]
        _uv = np.stack(px_uv, axis=1)
        if stem not in stem_to_fi:
            continue
        fi = stem_to_fi[stem]
        a_i, b_i = fits[fi]
        # Inlier filter with the frame's own robust fit (same trimming the
        # fit itself uses): a few anchors have rel ~ b (predicted Z explodes)
        # or sit on far-out sparse points; squared loss must never see them.
        pred_rel = a_i * (1.0 / z) + b_i
        mad = _fit_mad(1.0 / z, r, a_i, b_i)
        keep_a = np.abs(r - pred_rel) <= max(3.0 * mad, 1e-6)
        anch_frame.append(np.full(int(keep_a.sum()), fi, dtype=np.int32))
        anch_rel.append(r[keep_a])
        anch_z.append(z[keep_a])
        anch_uv.append(_uv[keep_a])
    anch_z_all = np.concatenate(anch_z)
    z_cap = np.percentile(anch_z_all, 99.5)
    print(f"anchor inliers {len(anch_z_all)}, z cap {z_cap:.2f}", flush=True)
    np.savez_compressed(
        OUT / "refine_data.npz",
        rays=rays, rels=rels, cols=cols, offs=offs, fits=fits, fidx=fidx,
        inverse=inverse.astype(np.int64), multi=multi, lo=lo, voxel=np.array([voxel]),
        anch_frame=np.concatenate(anch_frame), anch_rel=np.concatenate(anch_rel),
        anch_z=np.concatenate(anch_z), anch_uv=np.concatenate(anch_uv),
        uvs=uvs,
        stems=np.array(frame_stems),
    )
    print("saved", OUT / "refine_data.npz", flush=True)


if __name__ == "__main__":
    main()
