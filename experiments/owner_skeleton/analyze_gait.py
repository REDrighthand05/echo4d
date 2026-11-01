"""Owner-skeleton signal analysis on room2's recovered camera poses (v2).

The camera is held by the owner, so its recovered path is the owner's
root motion sampled at the pose rate. This measures what that stream
actually encodes: path/speed, camera height above the floor, head
direction vs travel direction. Scale = shipped v1 estimate.
"""
import json, os
import numpy as np

HERE = os.path.dirname(__file__)
JOB = os.path.join(HERE, "..", "..", "jobs", "room2")
SCALE = 0.2761          # m per scene unit (v1 estimate)
EXTRACT_FPS = 320 / 159.4597   # frames extracted from the source video

poses = json.load(open(os.path.join(JOB, "poses.json")))
names = sorted(poses.keys(), key=lambda n: int(n.split("_")[1].split(".")[0]))
idx = np.array([int(n.split("_")[1].split(".")[0]) for n in names], dtype=float)
t_s = idx / EXTRACT_FPS

def quat_R(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])

C, F = [], []
for n in names:
    R = quat_R(poses[n]["cam_from_world_quat_xyzw"])
    C.append(-R.T @ np.array(poses[n]["cam_from_world_t"]))
    F.append(R.T @ np.array([0.0, 0.0, 1.0]))
C = np.array(C); F = np.array(F)

# fused cloud with proper mixed dtype
raw = open(os.path.join(JOB, "scene_points.ply"), "rb").read()
he = raw.index(b"end_header") + len(b"end_header") + 1
dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("r", "u1"), ("g", "u1"), ("b", "u1")])
arr = np.frombuffer(raw[he:he + 295511 * dt.itemsize], dtype=dt, count=295511)
P = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float64)
P = P[np.isfinite(P).all(1)]

# up axis: the axis whose low end has the sharpest dense plane (the floor)
best = None
for a in range(3):
    lo, hi = np.percentile(P[:, a], [0.2, 99.8])
    dens = np.mean(P[:, a] < lo + 0.03 * (hi - lo))
    if best is None or dens > best[0]:
        best = (dens, a, lo)
_, ax, floor = best
if np.median(C[:, ax]) < np.median(P[:, ax]):
    s = -1.0
else:
    s = 1.0
height_m = (C[:, ax] * s - floor * s) * SCALE

seg = np.linalg.norm(np.diff(C, axis=0), axis=1) * SCALE
dt_ = np.diff(t_s)
speed = seg / dt_
path_m = seg.sum()
dur = t_s[-1] - t_s[0]

vel = np.diff(C, axis=0)
vel_h = vel.copy(); vel_h[:, ax] = 0
fh = F[:-1].copy(); fh[:, ax] = 0
nv = np.linalg.norm(vel_h, axis=1); nf = np.linalg.norm(fh, axis=1)
ok = (nv > 1e-6) & (nf > 1e-6)
cosang = np.sum(vel_h[ok] * fh[ok], 1) / (nv[ok] * nf[ok])
ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))

# vertical signal: what survives at ~2 Hz sampling
h = height_m - height_m.mean()
print(f"poses {len(names)}  span {dur:.0f}s  path {path_m:.1f} m  mean speed {speed.mean():.2f} m/s  median {np.median(speed):.2f}")
print(f"camera height above floor (m, est): mean {height_m.mean():.2f}  p10 {np.percentile(height_m,10):.2f}  p90 {np.percentile(height_m,90):.2f}  std {height_m.std():.3f}")
print(f"moving share (speed>0.15 m/s): {np.mean(speed>0.15):.0%};  near-still: {np.mean(speed<0.05):.0%}")
print(f"head-vs-travel angle (deg): median {np.median(ang):.0f}  p90 {np.percentile(ang,90):.0f}  share>45deg {np.mean(ang>45):.0%}")
print(f"vertical bob std in height series: {h.std()*100:.1f} cm (pose-rate {EXTRACT_FPS:.1f} Hz; step rhythm ~2 Hz is above this rate)")
np.save(os.path.join(HERE, "centers.npy"), C)
json.dump({"t_s": t_s.tolist(), "height_m": height_m.tolist(),
           "speed": speed.tolist()}, open(os.path.join(HERE, "series.json"), "w"), indent=0)
