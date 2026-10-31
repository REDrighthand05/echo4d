"""Global least-squares refinement, step B: joint optimization.

Variables: every frame's depth-alignment pair (a_i, b_i) with Z = a/(rel-b),
plus the shared global pair for frames without their own fit.
Objective (plain least squares, explicit and deterministic):
  sum over anchor pairs   w_a^2 * (Z_pred - Z_colmap)^2
+ sum over sampled points w_v^2 * ||p - centroid(voxel)||^2
Voxel assignment and centroids are recomputed between outer rounds and
held fixed inside each solve (alternating Gauss-Newton style).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

JOB = Path("jobs/room1_big")
D = np.load(JOB / "refine" / "refine_data.npz", allow_pickle=True)
rays, rels, offs, fits = D["rays"], D["rels"], D["offs"], D["fits"]
fidx, inverse, multi = D["fidx"], D["inverse"], D["multi"]
anch_f, anch_rel, anch_z = D["anch_frame"], D["anch_rel"], D["anch_z"]
# Final anchor hygiene in depth space: drop far-out sparse points and any
# anchor whose initial predicted depth already misses by > 1 unit (its
# observed rel sits within noise of b, so Z_pred explodes under any fit).
_zcap = np.percentile(anch_z, 99.5)
_init_res = np.abs(fits[anch_f, 0] / (anch_rel - fits[anch_f, 1]) - anch_z)
_keep = (anch_z <= _zcap) & (_init_res <= 1.0)
anch_f, anch_rel, anch_z = anch_f[_keep], anch_rel[_keep], anch_z[_keep]
print(f"anchors kept {len(anch_f)}, sum-sq at init {( (_init_res[_keep])**2 ).sum():.4g}", flush=True)
stems = [str(s) for s in D["stems"]]
NF = len(fits)
W_A, W_V = 1.0, 1.0
MAX_PTS = 600_000

# deterministic subsample of cross-frame-voxel points
in_multi = multi[inverse]
cand = np.flatnonzero(in_multi)
if len(cand) > MAX_PTS:
    cand = cand[:: len(cand) // MAX_PTS + 1][:MAX_PTS]
print(f"voxel-term points: {len(cand)}", flush=True)
p_rays, p_rels = rays[cand], rels[cand]
p_fidx, p_inv = fidx[cand], inverse[cand]
p_offs = offs[p_fidx]

# parameter layout: frames with their own fit are exactly the frames whose
# (a,b) differ per frame in `fits`; simpler: give EVERY frame its own pair,
# initialized from `fits` (global-fit frames share the init but may drift
# individually; anchors + voxel terms still constrain them).
x0 = fits.reshape(-1).copy()
a0 = fits[:, 0].copy()
lo_b = np.empty_like(x0)
hi_b = np.empty_like(x0)
lo_b[0::2] = a0 * 0.3
hi_b[0::2] = a0 * 3.0
lo_b[1::2] = fits[:, 1] - 0.1
hi_b[1::2] = fits[:, 1] + 0.1


def points_for(x, sel_rays, sel_rels, sel_fidx, sel_offs):
    a = x[2 * sel_fidx]
    b = x[2 * sel_fidx + 1]
    z = a / (sel_rels - b)
    return z[:, None] * sel_rays + sel_offs, z


def spread_for(x):
    pts, _ = points_for(x, rays, rels, fidx, offs[fidx])
    nv = len(multi)
    cnt = np.bincount(inverse).astype(np.float64)
    cent = np.zeros((nv, 3))
    for k in range(3):
        cent[:, k] = np.bincount(inverse, weights=pts[:, k]) / cnt
    d = pts[in_multi] - cent[inverse[in_multi]]
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def make_funcs(cent_sub):
    def resid(x):
        pts, _ = points_for(x, p_rays, p_rels, p_fidx, p_offs)
        rv = (pts - cent_sub) * W_V
        a = x[2 * anch_f]
        b = x[2 * anch_f + 1]
        ra = (a / (anch_rel - b) - anch_z) * W_A
        return np.concatenate([rv.reshape(-1), ra])

    def jac(x):
        a = x[2 * p_fidx]
        b = x[2 * p_fidx + 1]
        z = a / (p_rels - b)
        dza = z / a
        dzb = z * z / a
        n = len(p_fidx)
        pair_cols = np.stack([2 * p_fidx, 2 * p_fidx + 1], axis=1).reshape(-1)
        rows_l, cols_l, vals_l = [], [], []
        for k in range(3):
            rows_l.append(np.repeat(3 * np.arange(n) + k, 2))
            cols_l.append(pair_cols)
            vals_l.append(
                np.stack([p_rays[:, k] * dza, p_rays[:, k] * dzb], axis=1).reshape(-1) * W_V
            )
        aa = x[2 * anch_f]
        bb = x[2 * anch_f + 1]
        za = aa / (anch_rel - bb)
        na = len(anch_f)
        rows_l.append(n * 3 + np.repeat(np.arange(na), 2))
        cols_l.append(np.stack([2 * anch_f, 2 * anch_f + 1], axis=1).reshape(-1))
        avals = np.empty(na * 2)
        avals[0::2] = (za / aa) * W_A
        avals[1::2] = (za * za / aa) * W_A
        vals_l.append(avals)
        rows = np.concatenate(rows_l)
        cols = np.concatenate(cols_l)
        vals = np.concatenate(vals_l)
        return coo_matrix((vals, (rows, cols)), shape=(n * 3 + na, 2 * NF)).tocsr()

    return resid, jac


x = x0.copy()
print(f"BEFORE spread {spread_for(x):.4f}", flush=True)
for rnd in range(3):
    pts_all, _ = points_for(x, rays, rels, fidx, offs[fidx])
    nv = len(multi)
    cnt = np.bincount(inverse).astype(np.float64)
    cent = np.zeros((nv, 3))
    for k in range(3):
        cent[:, k] = np.bincount(inverse, weights=pts_all[:, k]) / cnt
    resid, jac = make_funcs(cent[p_inv])
    t0 = time.time()
    sol = least_squares(resid, x, jac=jac, bounds=(lo_b, hi_b),
                        method="trf", x_scale="jac", max_nfev=40, verbose=0)
    x = sol.x
    anch_res = np.abs(x[2 * anch_f] / (anch_rel - x[2 * anch_f + 1]) - anch_z)
    print(f"round {rnd}: cost {sol.cost:.4g} nfev {sol.nfev} "
          f"{time.time()-t0:.0f}s spread {spread_for(x):.4f} "
          f"anchor |res| median {np.median(anch_res):.4f} p90 {np.percentile(anch_res, 90):.4f}",
          flush=True)

out = {stems[i]: [float(x[2 * i]), float(x[2 * i + 1])] for i in range(NF)}
(JOB / "refine" / "refined_fits.json").write_text(json.dumps(out, indent=1))
np.save(JOB / "refine" / "refined_x.npy", x)
print("saved refined_fits.json", flush=True)
