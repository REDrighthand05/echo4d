#!/usr/bin/env python3
"""Pioneer B -- EVALUATION ONLY. Never used to align.

Ground truth for the A<->B transform, chained through the joint room2
reconstruction: each sub-capture's frames also exist in room2's joint
SfM, so camera centres give exact frame correspondences. For each
capture: fit a similarity from its fused-frame camera centres to the
room2-frame centres of the same frames (Umeyama LSQ). Composing the
two fits yields the true B-fused -> A-fused transform, against which
the geometry-only estimate (align.py) is scored. The fits' own
residuals are the reconstruction noise floor: no alignment method
could honestly score better than that.
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import load_points_bin, umeyama, apply_T, density_core

APP = Path(__file__).resolve().parents[2]

def centers(poses: dict):
    out = {}
    for name, p in poses.items():
        q = np.array(p["cam_from_world_quat_xyzw"]); t = np.array(p["cam_from_world_t"])
        x, y, z, w = q
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        out[name] = -R.T @ t
    return out

def load_capture(capdir: Path):
    poses = json.loads((capdir / "poses.json").read_text())
    c_sfm = centers(poses)
    rot = np.array(json.loads((capdir / "alignment.json").read_text())["rotation"])
    # fused = R @ sfm (column convention)
    return {k: rot @ v for k, v in c_sfm.items()}

def invert(s, R, t):
    Ri = R.T
    return 1.0 / s, Ri, -Ri @ t / s

def compose(s1, R1, t1, s2, R2, t2):
    """apply (s1,R1,t1) first, then (s2,R2,t2)"""
    return s2 * s1, R2 @ R1, s2 * (R2 @ t1) + t2

def main():
    work = APP / "experiments" / "pioneers" / "work"
    room2 = centers(json.loads((APP / "jobs" / "room2" / "poses.json").read_text()))
    fits = {}
    for cap in ("capA", "capB"):
        cf = load_capture(work / cap)
        common = sorted(set(cf) & set(room2))
        src = np.array([cf[k] for k in common]); dst = np.array([room2[k] for k in common])
        s, R, t = umeyama(src, dst)
        resid = np.linalg.norm(apply_T(src, s, R, t) - dst, axis=1)
        fits[cap] = (s, R, t)
        print(f"{cap}: {len(common)} shared frames, centre-fit scale={s:.4f}, "
              f"residual median={np.median(resid):.4f} p90={np.percentile(resid,90):.4f} "
              f"(room2 units; room2 cloud extent ~22)")
    # GT: B-fused -> A-fused  =  S_A^{-1} o S_B
    sB = fits["capB"]; sAinv = invert(*fits["capA"])
    gt = compose(*sB, *sAinv)
    # my estimate
    m = json.loads((APP / "experiments" / "pioneers" / "alignment_metrics.json").read_text())
    # NOTE: align.py searched yaw about Y on Y-up clouds; its R is the full
    # ICP-refined rotation only if saved -- align.py saves scale/yaw/t.
    # Rebuild est R from yaw for scoring rotation/scale; the point-
    # displacement score uses the merged PLY directly instead (exact).
    print(f"GT   : scale={gt[0]:.4f}")
    print(f"EST  : scale={m['scale']:.4f}  yaw={m['yaw_deg']:.2f}deg")
    print(f"scale error: {(m['scale']/gt[0]-1)*100:+.2f}%")
    # point-displacement score: B points placed by GT vs by EST (full R).
    Rest = np.array(m["R"])
    # rotation angle between EST and GT rotations
    cosang = (np.trace(Rest.T @ gt[1]) - 1) / 2
    ang = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
    print(f"rotation error EST vs GT: {ang:.2f} deg")
    B, Bc = load_points_bin(work / "capB" / "points.bin")
    B, Bc = density_core(B, Bc)
    room = m["room_size_A_units"]
    step = max(1, len(B) // 20000)
    p_gt = apply_T(B[::step], *gt)
    p_est = apply_T(B[::step], m["scale"], Rest, np.array(m["translation"]))
    disp = np.linalg.norm(p_gt - p_est, axis=1) / room * 100
    print(f"EST-vs-GT point displacement (% of room): median={np.median(disp):.2f} "
          f"p90={np.percentile(disp,90):.2f}")
    out = {"gt_scale": gt[0], "est_scale": m["scale"],
           "rotation_error_deg": ang,
           "scale_error_pct": float((m["scale"]/gt[0]-1)*100),
           "disp_median_pct_room": float(np.median(disp)),
           "disp_p90_pct_room": float(np.percentile(disp, 90))}
    (APP / "experiments" / "pioneers" / "gt_eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
