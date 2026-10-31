"""Global least-squares refinement, step C: per-frame deformation field.

The joint (a, b) refinement converged with ~0.5% gain: per-frame scale and
offset are exhausted. What remains is intra-frame warp of the monocular
depth. Model: each frame's depth is multiplied by exp(Delta(u, v)), where
Delta is a smooth 4x4 bilinear control grid over the image (16 coeffs per
frame, zero = identity). Objective (plain least squares):
  anchors (depth residuals, hygienic set)
+ cross-frame voxel consistency (fixed assignment, refreshed centroids)
+ Tikhonov ridge on coefficients + grid smoothness differences.
Base (a, b) are the converged joint fits from step B.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

JOB = Path("jobs/room1_big")
D = np.load(JOB / "refine" / "refine_data.npz", allow_pickle=True)
rays, rels, offs, uvs = D["rays"], D["rels"], D["offs"], D["uvs"]
fidx, inverse, multi = D["fidx"], D["inverse"], D["multi"]
anch_f, anch_rel, anch_z, anch_uv = D["anch_frame"], D["anch_rel"], D["anch_z"], D["anch_uv"]
stems = [str(s) for s in D["stems"]]
NF = len(D["fits"])
G = 4
LAM_R, LAM_S = 1.0, 3.0
MAX_PTS = 600_000

refined = json.loads((JOB / "refine" / "refined_fits.json").read_text())
base = np.array([refined.get(stems[i], list(D["fits"][i])) for i in range(NF)])
base_a, base_b = base[:, 0], base[:, 1]

# anchor depth-space hygiene (same as step B)
_zcap = np.percentile(anch_z, 99.5)
_init_res = np.abs(base_a[anch_f] / (anch_rel - base_b[anch_f]) - anch_z)
_keep = (anch_z <= _zcap) & (_init_res <= 1.0)
anch_f, anch_rel, anch_z, anch_uv = anch_f[_keep], anch_rel[_keep], anch_z[_keep], anch_uv[_keep]
print(f"anchors kept {len(anch_f)}", flush=True)

in_multi = multi[inverse]
cand = np.flatnonzero(in_multi)
if len(cand) > MAX_PTS:
    cand = cand[:: len(cand) // MAX_PTS + 1][:MAX_PTS]
print(f"voxel-term points: {len(cand)}", flush=True)


def bilinear(uv):
    """Return (node_flat_idx (N,4) within one frame's grid, weights (N,4))."""
    gu = np.clip(uv[:, 0], 0.0, 1.0) * (G - 1)
    gv = np.clip(uv[:, 1], 0.0, 1.0) * (G - 1)
    i0 = np.minimum(np.floor(gu).astype(int), G - 2)
    j0 = np.minimum(np.floor(gv).astype(int), G - 2)
    fu = gu - i0
    fv = gv - j0
    nodes = np.stack([j0 * G + i0, j0 * G + i0 + 1, (j0 + 1) * G + i0, (j0 + 1) * G + i0 + 1], axis=1)
    w = np.stack([(1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv], axis=1)
    return nodes, w


p_nodes, p_w = bilinear(uvs[cand])
p_rays, p_rels, p_fidx = rays[cand], rels[cand], fidx[cand]
p_offs, p_inv = offs[p_fidx], inverse[cand]
p_base_z = base_a[p_fidx] / (p_rels - base_b[p_fidx])
a_nodes, a_w = bilinear(anch_uv)
a_base_z = base_a[anch_f] / (anch_rel - base_b[anch_f])

# smoothness pairs (within each frame grid)
sm_pairs = []
for fr in range(NF):
    off = fr * G * G
    for j in range(G):
        for i in range(G):
            if i + 1 < G:
                sm_pairs.append((off + j * G + i, off + j * G + i + 1))
            if j + 1 < G:
                sm_pairs.append((off + j * G + i, off + (j + 1) * G + i))
sm_pairs = np.array(sm_pairs)
NC = NF * G * G


def delta_at(c, frame_idx, nodes, w):
    cc = c.reshape(NF, G * G)
    return np.sum(cc[frame_idx[:, None], nodes] * w, axis=1)


def all_points(c):
    z = (base_a[fidx] / (rels - base_b[fidx])) * np.exp(delta_at(c, fidx, *bilinear(uvs)))
    return z[:, None] * rays + offs[fidx]


def spread_for(c):
    pts = all_points(c)
    cnt = np.bincount(inverse).astype(np.float64)
    cent = np.zeros((len(multi), 3))
    for k in range(3):
        cent[:, k] = np.bincount(inverse, weights=pts[:, k]) / cnt
    d = pts[in_multi] - cent[inverse[in_multi]]
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def make_funcs(cent_sub):
    def resid(c):
        dz = np.exp(delta_at(c, p_fidx, p_nodes, p_w))
        pts = (p_base_z * dz)[:, None] * p_rays + p_offs
        rv = pts - cent_sub
        za = a_base_z * np.exp(delta_at(c, anch_f, a_nodes, a_w))
        ra = za - anch_z
        rr = c * LAM_R
        rs = (c[sm_pairs[:, 0]] - c[sm_pairs[:, 1]]) * LAM_S
        return np.concatenate([rv.reshape(-1), ra, rr, rs])

    def jac(c):
        dz = np.exp(delta_at(c, p_fidx, p_nodes, p_w))
        z = p_base_z * dz
        n = len(p_fidx)
        # voxel block: d p_k / d c_node = ray_k * z * w_node
        rows_l, cols_l, vals_l = [], [], []
        gcol = (p_fidx[:, None] * (G * G) + p_nodes).reshape(-1)
        for k in range(3):
            rows_l.append(np.repeat(3 * np.arange(n) + k, 4))
            cols_l.append(gcol)
            vals_l.append((p_rays[:, k][:, None] * (z[:, None] * p_w)).reshape(-1))
        # anchor block
        dza = np.exp(delta_at(c, anch_f, a_nodes, a_w))
        za = a_base_z * dza
        na = len(anch_f)
        rows_l.append(n * 3 + np.repeat(np.arange(na), 4))
        cols_l.append((anch_f[:, None] * (G * G) + a_nodes).reshape(-1))
        vals_l.append((za[:, None] * a_w).reshape(-1))
        # ridge
        rows_l.append(n * 3 + na + np.arange(NC))
        cols_l.append(np.arange(NC))
        vals_l.append(np.full(NC, LAM_R))
        # smoothness
        ns = len(sm_pairs)
        r0 = n * 3 + na + NC
        rows_l.append(r0 + np.repeat(np.arange(ns), 2))
        cols_l.append(np.stack([sm_pairs[:, 0], sm_pairs[:, 1]], axis=1).reshape(-1))
        sv = np.empty(ns * 2)
        sv[0::2] = LAM_S
        sv[1::2] = -LAM_S
        vals_l.append(sv)
        return coo_matrix(
            (np.concatenate(vals_l), (np.concatenate(rows_l), np.concatenate(cols_l))),
            shape=(n * 3 + na + NC + ns, NC),
        ).tocsr()

    return resid, jac


c = np.zeros(NC)
print(f"BEFORE spread {spread_for(c):.4f}", flush=True)
for rnd in range(2):
    pts_all = all_points(c)
    cnt = np.bincount(inverse).astype(np.float64)
    cent = np.zeros((len(multi), 3))
    for k in range(3):
        cent[:, k] = np.bincount(inverse, weights=pts_all[:, k]) / cnt
    resid, jac = make_funcs(cent[p_inv])
    t0 = time.time()
    sol = least_squares(resid, c, jac=jac, bounds=(-0.35, 0.35), method="trf",
                        x_scale="jac", max_nfev=30, verbose=0)
    c = sol.x
    za = a_base_z * np.exp(delta_at(c, anch_f, a_nodes, a_w))
    ares = np.abs(za - anch_z)
    print(f"round {rnd}: cost {sol.cost:.4g} nfev {sol.nfev} {time.time()-t0:.0f}s "
          f"spread {spread_for(c):.4f} anchor |res| median {np.median(ares):.4f} "
          f"p90 {np.percentile(ares, 90):.4f} |c| max {np.abs(c).max():.3f}", flush=True)

out = {stems[i]: [float(v) for v in c.reshape(NF, G * G)[i]] for i in range(NF)}
(JOB / "refine" / "field_coeffs.json").write_text(json.dumps({"G": G, "coeffs": out}, indent=1))
print("saved field_coeffs.json", flush=True)
