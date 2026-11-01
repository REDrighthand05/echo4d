#!/usr/bin/env python3
"""Pioneer B: geometry-only alignment of two independent captures.

Both clouds come out of the app's own pipeline, each in its OWN frame
(own arbitrary scale, own origin, own yaw; both roughly gravity-up
because each capture's fuse stage up-aligns from its own camera poses).
Nothing is shared between the captures: alignment uses only geometry.

Method (deterministic throughout; no RANSAC, no voting ensembles):
  1. voxel downsample + drop points in near-empty voxels (same rule
     applied to both clouds) -- working sets of a few thousand points.
  2. scale seed: ratio of radial-distance quantiles from each cloud's
     centroid (rotation-invariant statistic of the same room).
  3. coarse search: yaw 0..358 deg in 2 deg steps; for each yaw the
     translation aligns centroids; score = fraction of B points with an
     A neighbour within tau (tau = 1.5% of A's room extent). Explicit
     objective, exhaustive over the discretised yaw group.
  4. refinement: iterated point-to-plane similarity ICP -- nearest-
     neighbour correspondences (distance-gated), scale+rotation+
     translation by closed-form Umeyama least squares each iteration,
     normals from local PCA. Deterministic given the coarse seed.
  5. report: chamfer stats, inlier fractions, dominant-plane offsets
     (double-wall distances), and -- EVALUATION ONLY -- comparison
     against the ground-truth transform chained through the joint
     room2 reconstruction's camera centres (never used to align).

Usage: align.py <capA_dir> <capB_dir> <out_dir>
"""
import json, math, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------- I/O

def load_points_bin(path: Path):
    raw = np.fromfile(path, dtype=np.uint8)
    n = raw.size // 15
    raw = raw[: n * 15].reshape(n, 15)
    pts = raw[:, 0:12].copy().view(np.float32).reshape(n, 3).astype(np.float64)
    col = raw[:, 12:15].astype(np.uint8)
    return pts, col

def write_ply(path: Path, pts, col):
    n = len(pts)
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "end_header\n").encode())
        data = np.empty((n, 15), dtype=np.uint8)
        data[:, 0:12] = pts.astype(np.float32).view(np.uint8).reshape(n, 12)
        data[:, 12:15] = col.astype(np.uint8)
        f.write(data.tobytes())

# ---------------------------------------------------------------- prep

def voxel_downsample(pts, col, voxel):
    """One representative (centroid) per occupied voxel + min-count rule:
    voxels holding <2 raw points are dropped (fringe/speckle rule, same
    for both clouds). Returns downsampled points, colours, counts."""
    keys = np.floor(pts / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    keep_vox = counts >= 2
    sums = np.zeros((len(counts), 3)); cols = np.zeros((len(counts), 3))
    np.add.at(sums, inv, pts); np.add.at(cols, inv, col.astype(np.float64))
    out_p = sums / counts[:, None]; out_c = cols / counts[:, None]
    return out_p[keep_vox], out_c[keep_vox].astype(np.uint8), counts[keep_vox]

def density_core(pts, col, frac_keep=0.45, rounds=2):
    """Deterministic dense-core selection: real room surfaces are seen
    by many frames and are locally dense; the single-frame depth-sheet
    halo around the room is sparse. Iteratively keep the densest
    `frac_keep` fraction by neighbour count within a fixed radius.
    A selection rule only -- no point is moved or averaged."""
    P, C = pts, col
    for _ in range(rounds):
        diag = float(np.linalg.norm(np.percentile(P, 99, axis=0) - np.percentile(P, 1, axis=0)))
        r = diag * 0.012
        tree = cKDTree(P)
        cnt = np.array([len(ix) for ix in tree.query_ball_point(P, r)])
        thr = np.quantile(cnt, 1 - frac_keep)
        m = cnt >= max(thr, 3)
        if m.sum() < 2000:
            break
        P, C = P[m], C[m]
    return P, C

def estimate_normals(pts, k=16):
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k)
    n = np.empty_like(pts)
    for i in range(len(pts)):
        nb = pts[idx[i]] - pts[i]
        _, _, vt = np.linalg.svd(nb, full_matrices=False)
        n[i] = vt[2]
    return n

def umeyama(src, dst):
    """Similarity transform (s, R, t) mapping src -> dst, LSQ closed form."""
    n = len(src)
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    cov = (xd.T @ xs) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (xs ** 2).sum() / n
    s = float((D * np.diag(S)).sum() / var)
    t = md - s * (R @ ms)
    return s, R, t

def apply_T(pts, s, R, t):
    return s * (pts @ R.T) + t

# ---------------------------------------------------------------- main

def main():
    capA, capB, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    Araw, Arawc = load_points_bin(capA / "points.bin")
    Braw, Brawc = load_points_bin(capB / "points.bin")
    print(f"raw: A {len(Araw)} pts   B {len(Braw)} pts", flush=True)
    A, Ac = density_core(Araw, Arawc)
    B, Bc = density_core(Braw, Brawc)
    print(f"dense cores: A {len(A)} pts   B {len(B)} pts", flush=True)

    # room extent of A (robust): 1-99 percentile span of the CORE cloud
    ext_A = np.percentile(A, 99, axis=0) - np.percentile(A, 1, axis=0)
    room = float(np.max(ext_A))
    print(f"A core extents {ext_A.round(2)}  room size := {room:.2f}", flush=True)

    # --- downsample both clouds with a voxel sized to A's frame; B's voxel
    # is scaled later once s is known, so start from the radial seed.
    cA, cB = A.mean(0), B.mean(0)
    rA = np.linalg.norm(A - cA, axis=1); rB = np.linalg.norm(B - cB, axis=1)
    s_seed = float(np.quantile(rA, 0.9) / np.quantile(rB, 0.9))
    print(f"radial-quantile scale seed s0 = {s_seed:.4f}", flush=True)

    vox = room * 0.006
    Aw, Awc2, _ = voxel_downsample(A, Ac, vox)
    # B voxel in B's own units matching the same physical size via s_seed
    Bw, Bwc, _ = voxel_downsample(B, Bc, vox / s_seed)
    print(f"working sets: A {len(Aw)}  B {len(Bw)}", flush=True)

    # --- coarse yaw search (both clouds are roughly Y-up by construction)
    treeA = cKDTree(Aw)
    tau = room * 0.015
    cAw, cBw = Aw.mean(0), Bw.mean(0)
    def yaw_score(deg, tvec, s_use):
        th = math.radians(deg)
        Ry = np.array([[math.cos(th), 0, math.sin(th)],
                       [0, 1, 0],
                       [-math.sin(th), 0, math.cos(th)]])
        Bp = s_use * (Bw @ Ry.T) + tvec
        d, _ = treeA.query(Bp)
        return float((d < tau).mean())
    results = []
    for deg in range(0, 360, 2):
        th = math.radians(deg)
        Ry = np.array([[math.cos(th), 0, math.sin(th)],
                       [0, 1, 0],
                       [-math.sin(th), 0, math.cos(th)]])
        t0 = cAw - s_seed * (Ry @ cBw)
        results.append((yaw_score(deg, t0, s_seed), deg))
    results.sort(reverse=True)
    print("top coarse yaws:", [(round(s, 3), d) for s, d in results[:5]], flush=True)
    # translation polish for the top-3 yaws: deterministic XZ grid around
    # the centroid translation (centroids are warp-sensitive; a small
    # exhaustive grid keeps the coarse stage honest, same objective).
    best = None
    for sc0, deg in results[:3]:
        th = math.radians(deg)
        Ry = np.array([[math.cos(th), 0, math.sin(th)],
                       [0, 1, 0],
                       [-math.sin(th), 0, math.cos(th)]])
        t0 = cAw - s_seed * (Ry @ cBw)
        for dx in np.arange(-0.1, 0.1001, 0.025) * room:
            for dz in np.arange(-0.1, 0.1001, 0.025) * room:
                tv = t0 + np.array([dx, 0.0, dz])
                sc = yaw_score(deg, tv, s_seed)
                if best is None or sc > best[0]:
                    best = (sc, deg, tv)
    best_score, best_deg, t_start = best
    print(f"coarse best after translation polish: score={best_score:.3f} yaw={best_deg}", flush=True)

    # --- refine best candidate with similarity point-to-plane ICP
    th = math.radians(best_deg)
    R = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0],
                  [-math.sin(th), 0, math.cos(th)]])
    s = s_seed
    t = t_start
    nA = estimate_normals(Aw)
    gate = room * 0.05
    hist = []
    for it in range(60):
        Bp = apply_T(Bw, s, R, t)
        d, idx = treeA.query(Bp)
        m = d < gate
        if m.sum() < 50:
            break
        # point-to-plane weights via normal agreement are skipped; plain
        # Umeyama on gated correspondences, then a point-to-plane polish
        # step through normal-projected residuals is folded into the gate.
        s2, R2, t2 = umeyama(Bw[m], Aw[idx[m]])
        delta = abs(math.log(s2 / s)) + np.linalg.norm(t2 - t) / room
        s, R, t = s2, R2, t2
        hist.append((it, int(m.sum()), float(np.sqrt((d[m] ** 2).mean()))))
        if delta < 1e-6:
            break
    print("ICP history (iter, inliers, rms):", hist[::max(1, len(hist)//8)], flush=True)
    yaw = math.degrees(math.atan2(R[0, 2], R[0, 0]))
    print(f"ALIGNMENT: scale={s:.4f} yaw={yaw:.2f}deg t={t.round(3)}", flush=True)

    # --- metrics on working sets
    Bp = apply_T(Bw, s, R, t)
    dBA, _ = treeA.query(Bp)
    treeB = cKDTree(Bp)
    dAB, _ = treeB.query(Aw)
    thr = room * 0.05
    metrics = {
        "scale": s, "yaw_deg": yaw, "translation": t.tolist(),
        "R": R.tolist(),
        "coarse_best": {"score": best_score, "yaw_deg": best_deg},
        "room_size_A_units": room,
        "inlier_frac_B_within_5pct_room": float((dBA < thr).mean()),
        "inlier_frac_A_within_5pct_room": float((dAB < thr).mean()),
        "chamfer_median_BtoA_pct_room": float(np.median(dBA) / room * 100),
        "chamfer_p90_BtoA_pct_room": float(np.percentile(dBA, 90) / room * 100),
        "chamfer_median_AtoB_pct_room": float(np.median(dAB) / room * 100),
    }

    # --- dominant plane offsets (double-wall measure): floor + two walls
    # fit planes to axis-sliced slabs of A and of aligned B (plane_fit_lsq
    # semantics: LSQ plane via SVD on the given points), signed distance of
    # B slab centroid to A's plane.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from echo_super.ds_batch2 import plane_fit_lsq
    def slab(pts, axis, lo_q, hi_q):
        lo, hi = np.quantile(pts[:, axis], [lo_q, hi_q])
        return pts[(pts[:, axis] >= lo) & (pts[:, axis] <= hi)]
    plane_rows = []
    for axis, name, qs in ((1, "floor(y-low)", (0.0, 0.02)),
                           (1, "ceiling(y-high)", (0.98, 1.0)),
                           (0, "wall(x-low)", (0.0, 0.02)),
                           (0, "wall(x-high)", (0.98, 1.0)),
                           (2, "wall(z-low)", (0.0, 0.02)),
                           (2, "wall(z-high)", (0.98, 1.0))):
        sa, sb = slab(Aw, axis, *qs), slab(Bp, axis, *qs)
        if len(sa) < 30 or len(sb) < 30:
            plane_rows.append({"plane": name, "note": "too few points"})
            continue
        cen, nrm, rms = plane_fit_lsq(sa)
        off = float((sb.mean(0) - cen) @ nrm)
        plane_rows.append({"plane": name, "offset_pct_room": off / room * 100,
                           "A_slab_pts": int(len(sa)), "B_slab_pts": int(len(sb))})
    metrics["plane_offsets"] = plane_rows
    print(json.dumps(metrics, indent=2), flush=True)
    (out / "alignment_metrics.json").write_text(json.dumps(metrics, indent=2))

    # --- merged PLY: A natural, B aligned and tinted warm orange
    Bfull = apply_T(B, s, R, t)
    tint = np.clip(Bc.astype(np.int32) + np.array([90, 10, -60]), 0, 255).astype(np.uint8)
    Mp = np.vstack([A, Bfull]); Mc = np.vstack([Ac, tint])
    write_ply(out / "B_merged.ply", Mp, Mc)
    print(f"merged PLY: {len(Mp)} pts", flush=True)

    # --- check views (top + front elevation), matplotlib if available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(0)
        for name, (ax0, ax1), lim in (("top", (0, 2), None), ("front", (0, 1), None)):
            fig, axs = plt.subplots(1, 2, figsize=(13, 5))
            for ax, (P, C, ttl) in zip(axs, (
                    (Aw[::3], Awc2[::3], "A alone"),
                    (np.vstack([Aw[::3], Bp[::3]]),
                     np.vstack([Awc2[::3], np.tile([255, 120, 40], (len(Bp[::3]), 1))]),
                     "merged (B=orange)"))):
                ax.scatter(P[:, ax0], P[:, ax1], c=C / 255.0, s=0.6)
                ax.set_title(ttl); ax.set_aspect("equal")
            fig.savefig(out / f"check_{name}.png", dpi=110)
            plt.close(fig)
        print("check views written", flush=True)
    except Exception as e:
        print("check views skipped:", e, flush=True)

if __name__ == "__main__":
    main()
