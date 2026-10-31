"""Joint route+depth refinement, step B: pose-only least squares.

Variables: per-frame pose corrections (axis-angle d-w, translation d-t),
applied in camera coordinates:  Xc' = dR (R Xw + t) + dt.
The (a, b) depth fits from joint_prep stay FIXED. Objective (plain least
squares, explicit, deterministic — no RANSAC, no learned parts, no
voting/averaging merges):

  term 1  co-visibility depth consistency: a point back-projected from
          frame i's depth map, moved into frame j with corrected poses,
          must sit at the depth frame j's own map reports at that pixel
          (both sides in Z via the fixed (a, b)). Target depths are
          re-sampled between outer rounds and held fixed inside a solve
          (same alternating pattern as optimize_refine's centroids).
  term 2  sparse-anchor reprojection guard: COLMAP sparse points must
          still project onto their observed pixels under corrected
          poses, so the trajectory cannot drift to cheat term 1.
  term 3  small Tikhonov pull of all corrections toward 0.

Solver: scipy least_squares (TRF) with an analytic sparse Jacobian
(first-order in the increments, re-linearized every outer round when the
increments are folded exactly into the base poses).
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_prep import bilinear_rel  # noqa: E402
from echo_scene.fuse import _quat_xyzw_to_matrix  # noqa: E402

JOB = Path("jobs/room2_joint")
D = np.load(JOB / "joint" / "joint_data.npz", allow_pickle=True)
stems = [str(s) for s in D["stems"]]
fits, R0, t0, cam, zcap = D["fits"], D["R0"], D["t0"], D["cam"], D["zcap"]
H, W = int(D["hw"][0]), int(D["hw"][1])
NF = len(stems)
s_src, s_tgt, s_pc = D["s_src"], D["s_tgt"], D["s_pc"]
anch_f, anch_X, anch_uv = D["anch_f"], D["anch_X"], D["anch_uv"]
DEPTH = JOB / "depth"

W1 = 1.0        # depth consistency, per scene unit
W2 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05   # anchor reproj, per px
W_TIK = 1.0     # Tikhonov on (rad, units)
BND_W = 0.015   # per-round bound on rotation increments, rad (~0.86 deg)
BND_T = 0.04    # per-round bound on translation increments, scene units
GATE_ABS_R, GATE_REL_R = 0.8, 0.45   # refresh gate (looser than prep)

# sample groups by (src, tgt) pair, fixed once
pair_key = s_src.astype(np.int64) * NF + s_tgt
order = np.argsort(pair_key, kind="stable")
pair_key_s = pair_key[order]
grp_starts = np.flatnonzero(np.r_[True, pair_key_s[1:] != pair_key_s[:-1]])
grp_pairs = [(int(s_src[order[s]]), int(s_tgt[order[s]])) for s in grp_starts]
grp_ends = np.r_[grp_starts[1:], len(order)]
print(f"samples {len(s_src)} in {len(grp_pairs)} ordered pairs; "
      f"anchors {len(anch_f)}", flush=True)

# source-frame groups (for applying dR_i^T per frame)
src_order = np.argsort(s_src, kind="stable")
src_starts = {}
_tmp = s_src[src_order]
for s in np.flatnonzero(np.r_[True, _tmp[1:] != _tmp[:-1]]):
    src_starts[int(_tmp[s])] = (int(s), int(_tmp[s]))
src_bounds = {}
_ends = np.r_[np.flatnonzero(np.r_[True, _tmp[1:] != _tmp[:-1]])[1:], len(_tmp)]
for k, s in enumerate(np.flatnonzero(np.r_[True, _tmp[1:] != _tmp[:-1]])):
    src_bounds[int(_tmp[s])] = (int(s), int(_ends[k]))


def skew(v):
    n = len(v)
    K = np.zeros((n, 3, 3))
    K[:, 0, 1] = -v[:, 2]
    K[:, 0, 2] = v[:, 1]
    K[:, 1, 0] = v[:, 2]
    K[:, 1, 2] = -v[:, 0]
    K[:, 2, 0] = -v[:, 1]
    K[:, 2, 1] = v[:, 0]
    return K


def rodrigues(w):
    """(N,3) axis-angle -> (N,3,3) rotation matrices."""
    th = np.linalg.norm(w, axis=1)
    K = skew(w)
    K2 = K @ K
    s = np.where(th > 1e-12, np.sin(np.maximum(th, 1e-12)) / np.maximum(th, 1e-12), 1.0 - th * th / 6.0)
    c = np.where(th > 1e-12, (1 - np.cos(th)) / np.maximum(th, 1e-12) ** 2, 0.5 - th * th / 24.0)
    return np.eye(3)[None] + s[:, None, None] * K + c[:, None, None] * K2


def apply(M, V):
    """Per-sample M @ v: M (N,3,3), V (N,3) -> (N,3)."""
    return np.einsum("njk,nk->nj", M, V)


def refresh_Zj(R, t):
    """Recompute target depths + validity under base poses (R, t)."""
    Zj = np.full(len(s_src), np.nan)
    valid = np.zeros(len(s_src), dtype=bool)
    by_tgt = defaultdict(list)
    for gi, (i, j) in enumerate(grp_pairs):
        by_tgt[j].append(gi)
    for j, gis in sorted(by_tgt.items()):
        npy = DEPTH / f"{stems[j]}.npy"
        if not npy.exists():
            continue
        rel_map = np.load(npy).astype(np.float64)
        a_j, b_j = fits[j]
        denom_ok = (rel_map - b_j) > 1e-4
        f, cx, cy = cam[j, 0], cam[j, 1], cam[j, 2]
        for gi in gis:
            i, _ = grp_pairs[gi]
            sel = order[grp_starts[gi]:grp_ends[gi]]
            Pw = (s_pc[sel] - t[i]) @ R[i]
            Pj = Pw @ R[j].T + t[j]
            z = Pj[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = f * Pj[:, 0] / z + cx
                v = f * Pj[:, 1] / z + cy
            inb = (z > 0.3) & (u >= 2) & (u < W - 2) & (v >= 2) & (v < H - 2)
            if not inb.any():
                continue
            idx = np.flatnonzero(inb)
            rel_j = bilinear_rel(rel_map, denom_ok, u[idx], v[idx])
            with np.errstate(divide="ignore", invalid="ignore"):
                Z = a_j / (rel_j - b_j)
            good = np.isfinite(Z) & (Z > 0.05) & (Z < zcap[j]) \
                & (np.abs(z[idx] - Z) <= np.maximum(GATE_ABS_R, GATE_REL_R * Z))
            keep_idx = sel[idx[good]]
            Zj[keep_idx] = Z[good]
            valid[keep_idx] = True
        del rel_map, denom_ok
    return Zj, valid


def project_px(Xc, f, cx, cy, k):
    z = Xc[:, 2]
    xn = Xc[:, 0] / z
    yn = Xc[:, 1] / z
    d = 1.0 + k * (xn * xn + yn * yn)
    return np.stack([f * xn * d + cx, f * yn * d + cy], axis=1)


def anchor_errors(R, t):
    Xc = apply(R[anch_f], anch_X) + t[anch_f]
    pr = project_px(Xc, cam[anch_f, 0], cam[anch_f, 1], cam[anch_f, 2], cam[anch_f, 3])
    return np.linalg.norm(pr - anch_uv, axis=1)


class Round:
    """One outer round: fixed targets Zj, base poses (R, t)."""

    def __init__(self, R, t, keep, Zj):
        self.R, self.t = R, t
        self.si = s_src[keep]
        self.ti = s_tgt[keep]
        self.Pc = s_pc[keep]
        self.Zj = Zj[keep]
        self.M = R[self.ti] @ R[self.si].transpose(0, 2, 1)  # R_j R_i^T per sample
        self.RT = R.transpose(0, 2, 1)
        self.Xc0 = apply(R[anch_f], anch_X) + t[anch_f]
        # per-source-frame slices of the kept samples
        self.by_src = {}
        so = np.argsort(self.si, kind="stable")
        self.so = so
        ss = self.si[so]
        st = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
        en = np.r_[st[1:], len(ss)]
        for a, b in zip(st, en):
            self.by_src[int(ss[a])] = (int(a), int(b))

    def _terms(self, x):
        w = x.reshape(NF, 6)[:, :3]
        tau = x.reshape(NF, 6)[:, 3:]
        dR = rodrigues(w)
        dRT = dR.transpose(0, 2, 1)
        # term-1 point chain: Qi = dR_i^T (Pc - tau_i); Pw = R_i^T (Qi - t_i)
        Qi = self.Pc.copy()
        for fi, (a, b) in self.by_src.items():
            sl = self.so[a:b]
            Qi[sl] = apply(dRT[np.full(b - a, fi)], self.Pc[sl] - tau[fi])
        Pw = apply(self.RT[self.si], Qi - self.t[self.si])
        Sj = apply(self.R[self.ti], Pw) + self.t[self.ti]
        Pj = apply(dR[self.ti], Sj) + tau[self.ti]
        Xc = apply(dR[anch_f], self.Xc0) + tau[anch_f]
        return dR, tau, Pj, Sj, Xc

    def resid(self, x):
        _, _, Pj, _, Xc = self._terms(x)
        r1 = (Pj[:, 2] - self.Zj) * W1
        pr = project_px(Xc, cam[anch_f, 0], cam[anch_f, 1],
                        cam[anch_f, 2], cam[anch_f, 3])
        r2 = (pr - anch_uv).reshape(-1) * W2
        r3 = x * W_TIK
        return np.concatenate([r1, r2, r3])

    def jac(self, x):
        dR, tau, Pj, Sj, Xc = self._terms(x)
        n1 = len(self.si)
        na = len(anch_f)
        rows, cols, vals = [], [], []

        def add(rr, cc, vv):
            rows.append(rr)
            cols.append(cc)
            vals.append(vv)

        ridx = np.arange(n1)
        # --- term 1, source frame i: dPj/dw_i = dR_j M [Pc]x ; dPj/dtau_i = -dR_j M
        A = dR[self.ti] @ self.M @ skew(self.Pc)
        B = dR[self.ti] @ self.M
        for c in range(3):
            add(ridx, 6 * self.si + c, A[:, 2, c] * W1)
            add(ridx, 6 * self.si + 3 + c, -B[:, 2, c] * W1)
        # --- term 1, target frame j: dPj/dw_j = -[Pj - tau_j]x (z-row), dPj/dtau_j = ez
        v = Pj - tau[self.ti]
        add(ridx, 6 * self.ti + 0, v[:, 1] * W1)
        add(ridx, 6 * self.ti + 1, -v[:, 0] * W1)
        add(ridx, 6 * self.ti + 5, np.full(n1, W1))
        # --- term 2 anchors
        f = cam[anch_f, 0]
        cx, cy, k = cam[anch_f, 1], cam[anch_f, 2], cam[anch_f, 3]
        X, Y, Z = Xc[:, 0], Xc[:, 1], Xc[:, 2]
        xn, yn = X / Z, Y / Z
        r2d = xn * xn + yn * yn
        dd = 1.0 + k * r2d
        # d pixel / d Xc  (2x3 per anchor)
        dxn = np.stack([1.0 / Z, np.zeros(na), -xn / Z], axis=1)
        dyn = np.stack([np.zeros(na), 1.0 / Z, -yn / Z], axis=1)
        d_r2 = 2.0 * k[:, None] * (xn[:, None] * dxn + yn[:, None] * dyn)
        Jp = np.empty((na, 2, 3))
        Jp[:, 0] = f[:, None] * (dxn * dd[:, None] + xn[:, None] * d_r2)
        Jp[:, 1] = f[:, None] * (dyn * dd[:, None] + yn[:, None] * d_r2)
        Jw = Jp @ (-skew(Xc))          # d pixel / d w_k
        arows = n1 + np.arange(2 * na)
        for c in range(3):
            add(arows[0::2], 6 * anch_f + c, Jw[:, 0, c] * W2)
            add(arows[1::2], 6 * anch_f + c, Jw[:, 1, c] * W2)
            add(arows[0::2], 6 * anch_f + 3 + c, Jp[:, 0, c] * W2)
            add(arows[1::2], 6 * anch_f + 3 + c, Jp[:, 1, c] * W2)
        # --- term 3 Tikhonov
        tidx = n1 + 2 * na + np.arange(6 * NF)
        add(tidx, np.arange(6 * NF), np.full(6 * NF, W_TIK))
        J = coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n1 + 2 * na + 6 * NF, 6 * NF),
        )
        return J.tocsr()


def mat_to_quat_xyzw(Rm):
    tr = np.trace(Rm)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (Rm[2, 1] - Rm[1, 2]) / s
        y = (Rm[0, 2] - Rm[2, 0]) / s
        z = (Rm[1, 0] - Rm[0, 1]) / s
    elif Rm[0, 0] > Rm[1, 1] and Rm[0, 0] > Rm[2, 2]:
        s = np.sqrt(1.0 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2]) * 2
        w = (Rm[2, 1] - Rm[1, 2]) / s
        x = 0.25 * s
        y = (Rm[0, 1] + Rm[1, 0]) / s
        z = (Rm[0, 2] + Rm[2, 0]) / s
    elif Rm[1, 1] > Rm[2, 2]:
        s = np.sqrt(1.0 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2]) * 2
        w = (Rm[0, 2] - Rm[2, 0]) / s
        x = (Rm[0, 1] + Rm[1, 0]) / s
        y = 0.25 * s
        z = (Rm[1, 2] + Rm[2, 1]) / s
    else:
        s = np.sqrt(1.0 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1]) * 2
        w = (Rm[1, 0] - Rm[0, 1]) / s
        x = (Rm[0, 2] + Rm[2, 0]) / s
        y = (Rm[1, 2] + Rm[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def main():
    R = R0.copy()
    t = t0.copy()
    poses_orig = json.loads((JOB / "poses.json").read_text())
    (JOB / "joint" / "poses_original.json").write_text(json.dumps(poses_orig))
    names = sorted(poses_orig)
    err0 = anchor_errors(R, t)
    print(f"anchor reproj (original poses): mean {err0.mean():.4f}px "
          f"median {np.median(err0):.4f}px", flush=True)

    history = []
    prev_rms = None
    for rnd in range(3):
        t_start = time.time()
        Zj, valid = refresh_Zj(R, t)
        keep = np.flatnonzero(valid)
        # initial term-1 RMS this round at x = 0
        rd = Round(R, t, keep, Zj)
        r0 = rd.resid(np.zeros(6 * NF))
        n1 = len(keep)
        rms0 = float(np.sqrt(np.mean(r0[:n1] ** 2))) / W1
        print(f"[round {rnd}] samples kept {n1}/{len(s_src)}; "
              f"term-1 RMS at x=0: {rms0:.5f}", flush=True)
        lo = np.tile(np.array([-BND_W] * 3 + [-BND_T] * 3), NF)
        hi = np.tile(np.array([BND_W] * 3 + [BND_T] * 3), NF)
        sol = least_squares(rd.resid, np.zeros(6 * NF), jac=rd.jac,
                            bounds=(lo, hi),
                            method="trf", x_scale="jac", max_nfev=30,
                            ftol=1e-11, xtol=1e-11, gtol=1e-11, verbose=0)
        r1v = sol.fun[:n1] / W1
        rms1 = float(np.sqrt(np.mean(r1v ** 2)))
        w = sol.x.reshape(NF, 6)[:, :3]
        tau = sol.x.reshape(NF, 6)[:, 3:]
        dR = rodrigues(w)
        R = dR @ R
        t = np.einsum("njk,nk->nj", dR, t) + tau
        err = anchor_errors(R, t)
        rot_deg = np.degrees(np.linalg.norm(w, axis=1))
        tra = np.linalg.norm(tau, axis=1)
        print(f"[round {rnd}] nfev {sol.nfev} cost {sol.cost:.4g} "
              f"term-1 RMS {rms0:.5f} -> {rms1:.5f}; "
              f"anchor mean {err.mean():.4f}px; "
              f"round corr: rot med {np.median(rot_deg):.4f}deg max {rot_deg.max():.4f}deg, "
              f"trans med {np.median(tra):.5f} max {tra.max():.5f}; "
              f"{time.time() - t_start:.0f}s", flush=True)
        history.append({"round": rnd, "samples": int(n1), "rms_before": rms0,
                        "rms_after": rms1, "anchor_mean_px": float(err.mean()),
                        "nfev": int(sol.nfev)})
        if prev_rms is not None and rms1 > prev_rms * 0.995:
            print("term-1 RMS stalled; stopping rounds", flush=True)
            prev_rms = rms1
            break
        prev_rms = rms1

    # total correction vs original poses
    dR_tot = R @ R0.transpose(0, 2, 1)
    ang = np.degrees(np.arccos(np.clip(
        (np.trace(dR_tot, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    tau_tot = t - np.einsum("njk,nk->nj", dR_tot, t0)
    tra_tot = np.linalg.norm(tau_tot, axis=1)
    err_f = anchor_errors(R, t)
    stats = {
        "weights": {"W1": W1, "W2": W2, "W_TIK": W_TIK},
        "rounds": history,
        "anchor_mean_px_original": float(err0.mean()),
        "anchor_mean_px_final": float(err_f.mean()),
        "anchor_median_px_final": float(np.median(err_f)),
        "total_correction": {
            "rot_deg_median": float(np.median(ang)),
            "rot_deg_max": float(ang.max()),
            "trans_median": float(np.median(tra_tot)),
            "trans_max": float(tra_tot.max()),
        },
    }
    (JOB / "joint" / "joint_result.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1), flush=True)

    # write corrected poses.json (same structure as stage_poses output)
    out = {}
    for i, name in enumerate(names):
        p = dict(poses_orig[name])
        p["cam_from_world_quat_xyzw"] = [float(v) for v in mat_to_quat_xyzw(R[i])]
        p["cam_from_world_t"] = [float(v) for v in t[i]]
        out[name] = p
    (JOB / "poses.json").write_text(json.dumps(out, indent=1))
    print("wrote corrected poses.json", flush=True)


if __name__ == "__main__":
    main()
