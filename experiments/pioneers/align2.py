#!/usr/bin/env python3
"""Pioneer B, attempt 2: align via each capture's SPARSE SfM scaffold
(sharp, feature-anchored geometry), then apply the recovered transform
to the fused scene clouds and score those.

Attempt 1 (directly aligning the fused depth-sheet clouds) failed:
the sheets are thick and smeared, the coarse objective was nearly
flat, and the search locked onto a yaw 55 deg from truth. The sparse
clouds are a different story -- each capture's camera centres agree
with the joint reconstruction to sub-% residuals, so the scaffold
carries real shape. Still geometry-only, still deterministic:
  * coarse: PCA axis frames of the two sparse clouds, all 24 signed
    axis assignments scored by the same inlier objective;
  * refine: gated similarity ICP (Umeyama LSQ) on sparse points;
  * convert the SfM-frame transform into the fused frames via each
    capture's own alignment.json rotation (its own pipeline output);
  * score on the fused dense cores; write the merged PLY.
Usage: align2.py <capA_dir> <capB_dir> <out_dir>
"""
import json, math, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import (load_points_bin, voxel_downsample, density_core,
                   umeyama, apply_T, write_ply)

def load_sparse(path: Path):
    with open(path) as f:
        lines = f.readlines()
    hi = next(i for i, l in enumerate(lines) if l.startswith("end_header"))
    n = int([l for l in lines if "element vertex" in l][0].split()[-1])
    arr = np.array([[float(v) for v in lines[hi + 1 + i].split()[:6]] for i in range(n)])
    return arr[:, :3], arr[:, 3:6].astype(np.uint8)

def pca_frame(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return c, vt  # rows = principal axes

def main():
    capA, capB, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    sA, _ = load_sparse(capA / "sparse_points.ply")
    sB, _ = load_sparse(capB / "sparse_points.ply")
    # drop far outliers (1-99 pct box, both clouds)
    for nm, S in (("A", sA), ("B", sB)):
        lo, hi = np.percentile(S, 1, axis=0), np.percentile(S, 99, axis=0)
        print(f"sparse {nm}: {len(S)} pts, 1-99 extents {(hi-lo).round(2)}")
    def clip(P):
        lo, hi = np.percentile(P, 1, axis=0), np.percentile(P, 99, axis=0)
        return P[np.all((P >= lo) & (P <= hi), axis=1)]
    sA, sB = clip(sA), clip(sB)
    ext = float(np.max(np.percentile(sA, 99, axis=0) - np.percentile(sA, 1, axis=0)))
    Aw, _, _ = voxel_downsample(sA, np.zeros((len(sA), 3), np.uint8), ext * 0.004)
    # scale seed from radial quantiles
    rA = np.linalg.norm(sA - sA.mean(0), axis=1); rB = np.linalg.norm(sB - sB.mean(0), axis=1)
    s_seed = float(np.quantile(rA, 0.9) / np.quantile(rB, 0.9))
    Bw, _, _ = voxel_downsample(sB, np.zeros((len(sB), 3), np.uint8), ext * 0.004 / s_seed)
    print(f"sparse working sets A {len(Aw)} B {len(Bw)}; scale seed {s_seed:.4f}; extent {ext:.2f}", flush=True)

    cA, VA = pca_frame(Aw)
    cB, VB = pca_frame(Bw)
    treeA = cKDTree(Aw)
    tau = ext * 0.01
    import itertools
    cands = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for i, (p, sg) in enumerate(zip(perm, signs)):
                M[i, p] = sg
            if round(np.linalg.det(M)) != 1:
                continue
            # rotation mapping B coords -> A coords in PCA frames:
            R = VA.T @ M @ VB
            t = cA - s_seed * (R @ cB)
            Bp = apply_T(Bw, s_seed, R, t)
            d, _ = treeA.query(Bp)
            cands.append((float((d < tau).mean()), R, t))
    cands.sort(key=lambda x: -x[0])
    print("PCA-candidate scores:", [round(c[0], 3) for c in cands[:6]], flush=True)
    _, R, t = cands[0]
    s = s_seed

    gate = ext * 0.03
    for it in range(80):
        Bp = apply_T(Bw, s, R, t)
        d, idx = treeA.query(Bp)
        m = d < gate
        if m.sum() < 50:
            print("ICP starved at iter", it); break
        s2, R2, t2 = umeyama(Bw[m], Aw[idx[m]])
        delta = abs(math.log(s2 / s)) + np.linalg.norm(t2 - t) / ext
        s, R, t = s2, R2, t2
        if it % 10 == 0 or delta < 1e-7:
            print(f"ICP it{it}: inliers {int(m.sum())} rms {float(np.sqrt((d[m]**2).mean())):.4f} s {s:.4f}", flush=True)
        if delta < 1e-7:
            break
    Bp = apply_T(Bw, s, R, t)
    dBA, _ = treeA.query(Bp)
    print(f"sparse-frame result: scale {s:.4f}  B->A inliers<1%ext {float((dBA < ext*0.01).mean()):.3f} "
          f"median {float(np.median(dBA))/ext*100:.2f}% of ext", flush=True)

    # ---- convert SfM-frame transform to fused frames:
    # fused = RA @ sfm_A ; fused_B = RB @ sfm_B ; sfm_B -> sfm_A is (s,R,t)
    RA = np.array(json.loads((capA / "alignment.json").read_text())["rotation"])
    RB = np.array(json.loads((capB / "alignment.json").read_text())["rotation"])
    Rf = RA @ R @ RB.T
    tf = RA @ t
    sf = s
    print(f"fused-frame transform: scale {sf:.4f}", flush=True)

    # ---- score on fused dense cores, same rules as attempt 1
    Araw, Arawc = load_points_bin(capA / "points.bin")
    Braw, Brawc = load_points_bin(capB / "points.bin")
    A, Ac = density_core(Araw, Arawc)
    B, Bc = density_core(Braw, Brawc)
    extents = np.percentile(A, 99, axis=0) - np.percentile(A, 1, axis=0)
    room = float(np.max(extents))
    vox = room * 0.006
    Awf, Awcf, _ = voxel_downsample(A, Ac, vox)
    Bwf, Bwcf, _ = voxel_downsample(B, Bc, vox / sf)
    BpF = apply_T(Bwf, sf, Rf, tf)
    tA = cKDTree(Awf); tB = cKDTree(BpF)
    d1, _ = tA.query(BpF); d2, _ = tB.query(Awf)
    thr = room * 0.05
    metrics = {
        "method": "sparse-scaffold PCA24 + similarity ICP, applied to fused clouds",
        "sfm_scale": float(s), "fused_scale": float(sf), "fused_R": Rf.tolist(),
        "fused_t": tf.tolist(), "room_size_A_units": room,
        "inlier_frac_B_within_5pct_room": float((d1 < thr).mean()),
        "inlier_frac_A_within_5pct_room": float((d2 < thr).mean()),
        "chamfer_median_BtoA_pct_room": float(np.median(d1) / room * 100),
        "chamfer_p90_BtoA_pct_room": float(np.percentile(d1, 90) / room * 100),
        "chamfer_median_AtoB_pct_room": float(np.median(d2) / room * 100),
        "chamfer_p90_AtoB_pct_room": float(np.percentile(d2, 90) / room * 100),
    }
    print(json.dumps(metrics, indent=2), flush=True)
    (out / "alignment2_metrics.json").write_text(json.dumps(metrics, indent=2))

    Bfull = apply_T(B, sf, Rf, tf)
    tint = np.clip(Bc.astype(np.int32) + np.array([90, 10, -60]), 0, 255).astype(np.uint8)
    write_ply(out / "B_merged.ply", np.vstack([A, Bfull]), np.vstack([Ac, tint]))
    print(f"merged PLY (attempt 2): {len(A) + len(Bfull)} pts", flush=True)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for name, (a0, a1) in (("top", (0, 2)), ("front", (0, 1))):
            fig, axs = plt.subplots(1, 2, figsize=(13, 5))
            for ax, (P, C, ttl) in zip(axs, (
                    (Awf[::3], Awcf[::3], "A alone (fused core)"),
                    (np.vstack([Awf[::3], BpF[::3]]),
                     np.vstack([Awcf[::3], np.tile([255, 120, 40], (len(BpF[::3]), 1))]),
                     "merged attempt2 (B=orange)"))):
                ax.scatter(P[:, a0], P[:, a1], c=C / 255.0, s=0.6)
                ax.set_title(ttl); ax.set_aspect("equal")
            fig.savefig(out / f"check2_{name}.png", dpi=110); plt.close(fig)
        print("check2 views written", flush=True)
    except Exception as e:
        print("views skipped:", e, flush=True)

if __name__ == "__main__":
    main()
