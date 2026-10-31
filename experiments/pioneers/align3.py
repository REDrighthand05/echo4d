#!/usr/bin/env python3
"""Pioneer B, attempt 3: scaffold alignment on parallax-filtered sparse
points. Diagnosis from attempts 1-2 + the GT ceiling test:
  * capA's sparse cloud is a near-isotropic ~35-unit ball -- most of its
    triangulated points are near-zero-parallax junk flung far away;
    capB's is a clean anisotropic room [15.8, 9.0, 8.7].
  * Points near a capture's own camera path are the reliable scaffold.
Filter (own data only): keep sparse points whose distance to the
nearest own camera centre is within [0.02, 0.35] x path span. Then the
same deterministic machinery: PCA-24 coarse + gated similarity ICP,
convert to fused frames, score, merge. Also renders the GT-transform
merge as a third panel (the ceiling any rigid method could reach).
Usage: align3.py <capA_dir> <capB_dir> <out_dir>
"""
import itertools, json, math, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import (load_points_bin, voxel_downsample, density_core,
                   umeyama, apply_T, write_ply)
from align2 import load_sparse, pca_frame
from gt_eval import centers

def filtered_scaffold(capdir: Path):
    S, _ = load_sparse(capdir / "sparse_points.ply")
    poses = json.loads((capdir / "poses.json").read_text())
    C = np.array(list(centers(poses).values()))
    span = float(np.max(np.percentile(C, 99, axis=0) - np.percentile(C, 1, axis=0)))
    d, _ = cKDTree(C).query(S)
    m = (d >= 0.02 * span) & (d <= 0.35 * span)
    print(f"{capdir.name}: scaffold {len(S)} -> {int(m.sum())} pts (path span {span:.2f})", flush=True)
    return S[m]

def main():
    capA, capB, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    sA = filtered_scaffold(capA); sB = filtered_scaffold(capB)
    ext = float(np.max(np.percentile(sA, 99, axis=0) - np.percentile(sA, 1, axis=0)))
    print(f"filtered A extents {(np.percentile(sA,99,axis=0)-np.percentile(sA,1,axis=0)).round(2)}")
    print(f"filtered B extents {(np.percentile(sB,99,axis=0)-np.percentile(sB,1,axis=0)).round(2)}")
    rA = np.linalg.norm(sA - sA.mean(0), axis=1); rB = np.linalg.norm(sB - sB.mean(0), axis=1)
    s_seed = float(np.quantile(rA, 0.9) / np.quantile(rB, 0.9))
    Aw, _, _ = voxel_downsample(sA, np.zeros((len(sA), 3), np.uint8), ext * 0.004)
    Bw, _, _ = voxel_downsample(sB, np.zeros((len(sB), 3), np.uint8), ext * 0.004 / s_seed)
    print(f"working sets A {len(Aw)} B {len(Bw)}; scale seed {s_seed:.4f}; ext {ext:.2f}", flush=True)
    cA, VA = pca_frame(Aw); cB, VB = pca_frame(Bw)
    treeA = cKDTree(Aw); tau = ext * 0.01
    cands = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for i, (p, sg) in enumerate(zip(perm, signs)):
                M[i, p] = sg
            if round(np.linalg.det(M)) != 1:
                continue
            R = VA.T @ M @ VB
            t = cA - s_seed * (R @ cB)
            d, _ = treeA.query(apply_T(Bw, s_seed, R, t))
            cands.append((float((d < tau).mean()), R, t))
    cands.sort(key=lambda x: -x[0])
    print("PCA scores:", [round(c[0], 3) for c in cands[:6]], flush=True)
    _, R, t = cands[0]; s = s_seed
    for it in range(100):
        gate = ext * max(0.01, 0.03 * (0.92 ** it))
        Bp = apply_T(Bw, s, R, t)
        d, idx = treeA.query(Bp)
        m = d < gate
        if m.sum() < 50:
            print("ICP starved", it); break
        s2, R2, t2 = umeyama(Bw[m], Aw[idx[m]])
        delta = abs(math.log(s2 / s)) + np.linalg.norm(t2 - t) / ext
        s, R, t = s2, R2, t2
        if it % 15 == 0 or delta < 1e-7:
            print(f"ICP it{it}: gate {gate:.3f} inliers {int(m.sum())} rms {float(np.sqrt((d[m]**2).mean())):.4f} s {s:.4f}", flush=True)
        if delta < 1e-7:
            break
    Bp = apply_T(Bw, s, R, t)
    dBA, _ = treeA.query(Bp)
    print(f"sparse result: scale {s:.4f} inliers<1%ext {float((dBA < ext*0.01).mean()):.3f} median {float(np.median(dBA))/ext*100:.2f}%ext", flush=True)

    RA = np.array(json.loads((capA / "alignment.json").read_text())["rotation"])
    RB = np.array(json.loads((capB / "alignment.json").read_text())["rotation"])
    Rf = RA @ R @ RB.T; tf = RA @ t; sf = s

    Araw, Arawc = load_points_bin(capA / "points.bin")
    Braw, Brawc = load_points_bin(capB / "points.bin")
    A, Ac = density_core(Araw, Arawc); B, Bc = density_core(Braw, Brawc)
    room = float(np.max(np.percentile(A, 99, axis=0) - np.percentile(A, 1, axis=0)))
    Awf, Awcf, _ = voxel_downsample(A, Ac, room * 0.006)
    Bwf, Bwcf, _ = voxel_downsample(B, Bc, room * 0.006 / sf)
    BpF = apply_T(Bwf, sf, Rf, tf)
    tA, tB = cKDTree(Awf), cKDTree(BpF)
    d1, _ = tA.query(BpF); d2, _ = tB.query(Awf)
    thr = room * 0.05
    metrics = {"method": "parallax-filtered sparse scaffold PCA24 + ICP, applied to fused",
               "fused_scale": float(sf), "fused_R": Rf.tolist(), "fused_t": tf.tolist(),
               "room_size_A_units": room,
               "inlier_frac_B_within_5pct_room": float((d1 < thr).mean()),
               "inlier_frac_A_within_5pct_room": float((d2 < thr).mean()),
               "chamfer_median_BtoA_pct_room": float(np.median(d1) / room * 100),
               "chamfer_p90_BtoA_pct_room": float(np.percentile(d1, 90) / room * 100),
               "chamfer_median_AtoB_pct_room": float(np.median(d2) / room * 100),
               "chamfer_p90_AtoB_pct_room": float(np.percentile(d2, 90) / room * 100)}
    print(json.dumps(metrics, indent=2), flush=True)
    (out / "alignment3_metrics.json").write_text(json.dumps(metrics, indent=2))
    Bfull = apply_T(B, sf, Rf, tf)
    tint = np.clip(Bc.astype(np.int32) + np.array([90, 10, -60]), 0, 255).astype(np.uint8)
    write_ply(out / "B_merged.ply", np.vstack([A, Bfull]), np.vstack([Ac, tint]))
    print(f"merged PLY (attempt 3): {len(A) + len(Bfull)} pts", flush=True)

    # GT comparison for this transform (evaluation only)
    from gt_eval import load_capture, umeyama as um, compose, invert
    room2 = centers(json.loads((capA.parents[3] / "jobs" / "room2" / "poses.json").read_text()))
    fits = {}
    for cap, cd in (("capA", capA), ("capB", capB)):
        cf = load_capture(cd)
        common = sorted(set(cf) & set(room2))
        src = np.array([cf[k] for k in common]); dst = np.array([room2[k] for k in common])
        fits[cap] = um(src, dst)
    gt = compose(*fits["capB"], *invert(*fits["capA"]))
    cosang = (np.trace(Rf.T @ gt[1]) - 1) / 2
    ang = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
    step = max(1, len(B) // 20000)
    disp = np.linalg.norm(apply_T(B[::step], *gt) - apply_T(B[::step], sf, Rf, tf), axis=1) / room * 100
    gt_metrics = {"gt_scale": gt[0], "est_scale": float(sf),
                  "scale_error_pct": float((sf / gt[0] - 1) * 100),
                  "rotation_error_deg": ang,
                  "disp_median_pct_room": float(np.median(disp)),
                  "disp_p90_pct_room": float(np.percentile(disp, 90))}
    print("GT comparison:", json.dumps(gt_metrics, indent=2), flush=True)
    (out / "gt_eval3.json").write_text(json.dumps(gt_metrics, indent=2))

    # check views incl. GT ceiling panel
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        BpGT = apply_T(Bwf, *gt)
        for name, (a0, a1) in (("top", (0, 2)), ("front", (0, 1))):
            fig, axs = plt.subplots(1, 3, figsize=(17, 5))
            panels = ((Awf[::3], Awcf[::3], "A alone"),
                      (np.vstack([Awf[::3], BpF[::3]]),
                       np.vstack([Awcf[::3], np.tile([255, 120, 40], (len(BpF[::3]), 1))]),
                       "merged attempt3 (B=orange)"),
                      (np.vstack([Awf[::3], BpGT[::3]]),
                       np.vstack([Awcf[::3], np.tile([255, 120, 40], (len(BpGT[::3]), 1))]),
                       "merged under GT (ceiling)"))
            for ax, (P, C, ttl) in zip(axs, panels):
                ax.scatter(P[:, a0], P[:, a1], c=C / 255.0, s=0.6)
                ax.set_title(ttl); ax.set_aspect("equal")
            fig.savefig(out / f"check3_{name}.png", dpi=110); plt.close(fig)
        print("check3 views written", flush=True)
    except Exception as e:
        print("views skipped:", e, flush=True)

if __name__ == "__main__":
    main()
