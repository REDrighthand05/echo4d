"""Joint route+depth refinement, step A: data prep.

Base job: jobs/room2_joint (scratch copy of jobs/room2_sc — one shared
camera, 230 registered frames). Depth-side (a, b) fits are computed
exactly as production fuse / measure_spread compute them and stay FIXED;
the joint optimizer (step B) moves POSES only.

What is prepared here (all deterministic: fixed strides, fixed gates,
evenly-spaced subsampling — no randomness, no RANSAC, no learned parts):

1. Per-frame sampled source points: stride-10 subsample of each frame's
   depth map with fuse's validity + depth-edge rules, back-projected to
   camera coordinates Pi_c = Z * ray with the frame's fixed (a, b).
2. Co-visibility pairs from the COLMAP model itself: pairs of frames
   sharing >= 25 sparse points (the literal "intersection" of the
   routes). For each ordered pair (i -> j), frame i's points are
   projected into j with the ORIGINAL poses; samples survive if they
   land in-bounds, j's bilinearly sampled depth is valid, and the two
   depth readings roughly agree (loose gate, hygiene only).
3. Sparse-anchor observations for the reprojection guard: (frame,
   world point, observed pixel), deterministically subsampled.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_refine import _collect_anchors_with_pixels  # noqa: E402
from echo_scene.fuse import (  # noqa: E402
    _fit_mad,
    _quat_xyzw_to_matrix,
    _robust_affine_fit,
)

JOB = Path("jobs/room2_joint")
OUT = JOB / "joint"
STRIDE = 10
MIN_SHARED = 25
MAX_PAIRS = 1600          # unordered pairs, both directions used
PER_PAIR_CAP = 62         # samples per ordered pair
MAX_ANCH = 45_000
GATE_ABS, GATE_REL = 0.6, 0.35


def bilinear_rel(rel_map, denom_ok, u, v):
    """Bilinear sample of a relative-depth map; NaN where any corner is
    invalid (denominator test identical to fuse's)."""
    h, w = rel_map.shape
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    ok = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    x0c = np.clip(x0, 0, w - 2)
    y0c = np.clip(y0, 0, h - 2)
    fx = np.clip(u - x0c, 0.0, 1.0)
    fy = np.clip(v - y0c, 0.0, 1.0)
    corners = [(y0c, x0c), (y0c, x0c + 1), (y0c + 1, x0c), (y0c + 1, x0c + 1)]
    vals = []
    valid = ok.copy()
    for yy, xx in corners:
        valid &= denom_ok[yy, xx]
        vals.append(rel_map[yy, xx].astype(np.float64))
    out = ((vals[0] * (1 - fx) + vals[1] * fx) * (1 - fy)
           + (vals[2] * (1 - fx) + vals[3] * fx) * fy)
    return np.where(valid, out, np.nan)


def main():
    OUT.mkdir(exist_ok=True)
    poses = json.loads((JOB / "poses.json").read_text())
    names = sorted(poses)
    stems = [Path(n).stem for n in names]
    name_to_fi = {n: i for i, n in enumerate(names)}
    stem_to_fi = {s: i for i, s in enumerate(stems)}
    NF = len(names)
    depth_dir = JOB / "depth"

    # ---- fixed per-frame (a, b) fits, identical recipe to fuse/spread ----
    per_frame_px = _collect_anchors_with_pixels(JOB, depth_dir)
    per_frame = {k: (v[0], v[1]) for k, v in per_frame_px.items()}
    rel_vals = np.concatenate([v[0] for v in per_frame.values()])
    inv_z = np.concatenate([v[1] for v in per_frame.values()])
    a_glob, b_glob = _robust_affine_fit(inv_z, rel_vals)
    fits = np.tile(np.array([a_glob, b_glob]), (NF, 1))
    n_own = 0
    for stem, (f_rel, f_inv) in per_frame.items():
        if len(f_rel) < 60:
            continue
        a_i, b_i = _robust_affine_fit(f_inv, f_rel)
        if a_i <= 0:
            continue
        if _fit_mad(f_inv, f_rel, a_i, b_i) <= _fit_mad(f_inv, f_rel, a_glob, b_glob):
            fits[stem_to_fi[stem]] = (a_i, b_i)
            n_own += 1
    print(f"fits: global a={a_glob:.4f} b={b_glob:.4f}; own fits {n_own}/{NF}", flush=True)

    # ---- base poses + cameras from poses.json ----
    R0 = np.zeros((NF, 3, 3))
    t0 = np.zeros((NF, 3))
    cam = np.zeros((NF, 4))
    for i, n in enumerate(names):
        p = poses[n]
        R0[i] = _quat_xyzw_to_matrix(p["cam_from_world_quat_xyzw"])
        t0[i] = np.array(p["cam_from_world_t"], dtype=np.float64)
        prm = p["camera"]["params"]
        cam[i] = (float(prm[0]), float(prm[1]), float(prm[2]), float(prm[3]))
    H = int(poses[names[0]]["camera"]["height"])
    W = int(poses[names[0]]["camera"]["width"])

    # ---- per-frame sampled source points (camera coords) + z caps ----
    src_pts = []      # per frame: (n,3) Pi_c
    zcap = np.full(NF, np.inf)
    for i, stem in enumerate(stems):
        npy = depth_dir / f"{stem}.npy"
        if not npy.exists():
            # frame_00163: registered in the sc model but its depth map
            # was never produced (room2's set differed by one frame).
            # fuse/measure_spread skip it the same way.
            src_pts.append(np.zeros((0, 3)))
            print(f"note: {stem} has no depth map; excluded from term 1", flush=True)
            continue
        rel = np.load(npy).astype(np.float64)
        a, b = fits[i]
        denom = rel - b
        ok_den = denom > 1e-4
        z_map = np.where(ok_den, a / np.where(ok_den, denom, 1.0), np.nan)
        valid = ok_den.copy()
        if valid.any():
            z_hi = np.nanpercentile(z_map[valid], 99)
            zcap[i] = z_hi * 1.5
            valid &= z_map < zcap[i]
        f, cx, cy = cam[i, 0], cam[i, 1], cam[i, 2]
        ys, xs = np.mgrid[0:rel.shape[0]:STRIDE, 0:rel.shape[1]:STRIDE]
        zs = z_map[::STRIDE, ::STRIDE]
        ok = valid[::STRIDE, ::STRIDE] & np.isfinite(zs)
        if zs.shape[0] > 1 and zs.shape[1] > 1:
            jump = np.zeros_like(zs)
            jump[:, :-1] = np.maximum(jump[:, :-1], np.abs(zs[:, 1:] - zs[:, :-1]))
            jump[:-1, :] = np.maximum(jump[:-1, :], np.abs(zs[1:, :] - zs[:-1, :]))
            ok &= jump <= 0.18 * np.maximum(zs, 1e-6)
        xs_f = xs[ok].astype(np.float64)
        ys_f = ys[ok].astype(np.float64)
        zs_f = zs[ok]
        pc = np.stack([(xs_f - cx) / f * zs_f, (ys_f - cy) / f * zs_f, zs_f], axis=1)
        src_pts.append(pc)
    total_src = sum(len(p) for p in src_pts)
    print(f"source points per frame: total {total_src}, "
          f"median {int(np.median([len(p) for p in src_pts]))}", flush=True)
    # world points under ORIGINAL poses (for pair projection in prep)
    world_pts = [src_pts[i] @ R0[i] - t0[i] @ R0[i] for i in range(NF)]
    # note: rows convention: Xw = R^T (Xc - t)  ==  (Xc - t) @ R

    # ---- sparse model: co-visibility pairs + anchor observations ----
    import pycolmap
    sfm_dir = JOB / "sfm"
    models = [d for d in sfm_dir.iterdir()
              if d.is_dir() and d.name.isdigit() and (d / "cameras.bin").exists()]
    rec = max((pycolmap.Reconstruction(str(d)) for d in models),
              key=lambda r: r.num_reg_images())
    img_fi = {}   # rec image id -> frame index
    for image in rec.images.values():
        if image.name in name_to_fi:
            img_fi[image.image_id] = name_to_fi[image.name]
    pair_counts = defaultdict(int)
    anch_f, anch_X, anch_uv = [], [], []
    for p3 in rec.points3D.values():
        xyz = np.array(p3.xyz, dtype=np.float64)
        obs = []
        for el in p3.track.elements:
            fi = img_fi.get(el.image_id)
            if fi is None:
                continue
            img = rec.images[el.image_id]
            xy = img.points2D[el.point2D_idx].xy
            obs.append(fi)
            anch_f.append(fi)
            anch_X.append(xyz)
            anch_uv.append((float(xy[0]), float(xy[1])))
        for a_i in range(len(obs)):
            for b_i in range(a_i + 1, len(obs)):
                if obs[a_i] != obs[b_i]:
                    pair_counts[(min(obs[a_i], obs[b_i]), max(obs[a_i], obs[b_i]))] += 1
    anch_f = np.array(anch_f, dtype=np.int32)
    anch_X = np.array(anch_X)
    anch_uv = np.array(anch_uv)
    print(f"anchor observations total {len(anch_f)}", flush=True)
    if len(anch_f) > MAX_ANCH:
        sel = np.arange(0, len(anch_f), int(np.ceil(len(anch_f) / MAX_ANCH)))[:MAX_ANCH]
        anch_f, anch_X, anch_uv = anch_f[sel], anch_X[sel], anch_uv[sel]
    print(f"anchor observations kept {len(anch_f)}", flush=True)

    pairs = sorted(((c, ij) for ij, c in pair_counts.items() if c >= MIN_SHARED),
                   key=lambda z: -z[0])[:MAX_PAIRS]
    print(f"co-visibility pairs kept: {len(pairs)} unordered "
          f"(shared counts {pairs[-1][0]}..{pairs[0][0]})", flush=True)

    # ---- ordered-pair samples with ORIGINAL poses, grouped by target
    # frame so each target depth map is loaded only once ----
    ordered = []
    for _, (i, j) in pairs:
        ordered.append((i, j))
        ordered.append((j, i))
    by_target = defaultdict(list)
    for a_f, b_f in ordered:
        by_target[b_f].append(a_f)
    s_src, s_tgt, s_pc = [], [], []
    for b_f, a_list in sorted(by_target.items()):
        npy = depth_dir / f"{stems[b_f]}.npy"
        if not npy.exists():
            continue
        rel_map = np.load(npy).astype(np.float64)
        a_j, b_j = fits[b_f]
        denom_ok = (rel_map - b_j) > 1e-4
        f, cx, cy = cam[b_f, 0], cam[b_f, 1], cam[b_f, 2]
        for a_f in a_list:
            pc = src_pts[a_f]
            if len(pc) == 0:
                continue
            Pj = world_pts[a_f] @ R0[b_f].T + t0[b_f]
            z = Pj[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = f * Pj[:, 0] / z + cx
                v = f * Pj[:, 1] / z + cy
            cand = (z > 0.3) & (u >= 2) & (u < W - 2) & (v >= 2) & (v < H - 2)
            if not cand.any():
                continue
            idx = np.flatnonzero(cand)
            rel_j = bilinear_rel(rel_map, denom_ok, u[idx], v[idx])
            with np.errstate(divide="ignore", invalid="ignore"):
                Zj = a_j / (rel_j - b_j)
            good = np.isfinite(Zj) & (Zj > 0.05) & (Zj < zcap[b_f]) \
                & (np.abs(z[idx] - Zj) <= np.maximum(GATE_ABS, GATE_REL * Zj))
            idx = idx[good]
            if len(idx) == 0:
                continue
            if len(idx) > PER_PAIR_CAP:
                pick = np.linspace(0, len(idx) - 1, PER_PAIR_CAP).astype(int)
                idx = idx[pick]
            s_src.append(np.full(len(idx), a_f, dtype=np.int32))
            s_tgt.append(np.full(len(idx), b_f, dtype=np.int32))
            s_pc.append(pc[idx])
        del rel_map, denom_ok
    s_src = np.concatenate(s_src)
    s_tgt = np.concatenate(s_tgt)
    s_pc = np.concatenate(s_pc).astype(np.float64)
    print(f"term-1 samples: {len(s_src)}", flush=True)

    np.savez_compressed(
        OUT / "joint_data.npz",
        stems=np.array(stems), fits=fits, R0=R0, t0=t0, cam=cam,
        zcap=zcap, hw=np.array([H, W]),
        s_src=s_src, s_tgt=s_tgt, s_pc=s_pc,
        anch_f=anch_f, anch_X=anch_X, anch_uv=anch_uv,
        n_pairs=np.array([len(pairs)]),
    )
    print("saved", OUT / "joint_data.npz", flush=True)


if __name__ == "__main__":
    main()
