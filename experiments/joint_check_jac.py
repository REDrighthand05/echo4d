"""Finite-difference check of joint_optimize's analytic Jacobian."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint_optimize as jo  # noqa: E402

rng = np.random.default_rng(0)
Zj, valid = jo.refresh_Zj(jo.R0, jo.t0)
keep = np.flatnonzero(valid)[:4000]
rd = jo.Round(jo.R0, jo.t0, keep, Zj)
print(f"check round: {len(keep)} samples, {len(jo.anch_f)} anchors", flush=True)

NF = jo.NF
x = rng.normal(scale=1e-3, size=6 * NF)
J = rd.jac(x).toarray()
r0 = rd.resid(x)

# columns to test: rotation+translation of a source frame, a target
# frame, an anchor-heavy frame, plus a random frame
frames_test = [int(rd.si[0]), int(rd.ti[0]), int(jo.anch_f[0]), 100]
cols = []
for fi in frames_test:
    cols += [6 * fi + 0, 6 * fi + 2, 6 * fi + 3, 6 * fi + 5]
eps = 1e-6
worst = 0.0
for c in cols:
    xp = x.copy(); xp[c] += eps
    xm = x.copy(); xm[c] -= eps
    fd = (rd.resid(xp) - rd.resid(xm)) / (2 * eps)
    an = J[:, c]
    denom = np.maximum(np.abs(fd).max(), 1e-12)
    rel = np.abs(fd - an).max() / denom
    worst = max(worst, rel)
    print(f"col {c} (frame {c // 6}, {'w' if c % 6 < 3 else 't'}{c % 6 % 3}): "
          f"max|fd| {np.abs(fd).max():.4g} max|an| {np.abs(an).max():.4g} "
          f"rel err {rel:.2e}", flush=True)
print(f"WORST rel err: {worst:.2e}")
print("resid at x=0 term-1 RMS:", float(np.sqrt(np.mean(
    rd.resid(np.zeros(6 * NF))[:len(keep)] ** 2))))
