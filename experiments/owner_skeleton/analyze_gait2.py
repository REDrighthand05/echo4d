"""Owner-skeleton signals v3: vertical from the trajectory's own geometry.

A walking person's camera path lies near a horizontal plane, so the
smallest-variance axis of the path IS the vertical. Floor = the cloud's
dense low plane along that axis. Heading stats only on segments fast
enough for direction to be meaningful.
"""
import json, os
import numpy as np

HERE = os.path.dirname(__file__)
JOB = os.path.join(HERE, "..", "..", "jobs", "room2")
SCALE = 0.2761
EXTRACT_FPS = 320 / 159.4597

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

raw = open(os.path.join(JOB, "scene_points.ply"), "rb").read()
he = raw.index(b"end_header") + len(b"end_header") + 1
dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
arr = np.frombuffer(raw[he:he + 295511 * dt.itemsize], dtype=dt, count=295511)
P = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float64)
P = P[np.isfinite(P).all(1)]

# vertical = smallest-variance axis of the camera path
var = C.var(axis=0)
ax = int(np.argmin(var))
# floor along that axis: cameras must be ABOVE it by 1.0-2.0 m / SCALE units
lo, hi = np.percentile(P[:, ax], [0.2, 99.8])
for cand_floor, s in ((lo, 1.0), (hi, -1.0)):
    h = (C[:, ax] - cand_floor) * s * SCALE
    if 1.0 < h.mean() < 2.1:
        floor, sign, height_m = cand_floor, s, h
        break
else:
    floor, sign = lo, 1.0
    height_m = (C[:, ax] - floor) * SCALE

seg = np.linalg.norm(np.diff(C, axis=0), axis=1) * SCALE
speed = seg / np.diff(t_s)
vel = np.diff(C, axis=0); vel[:, ax] = 0
fh = F[:-1].copy(); fh[:, ax] = 0
nv = np.linalg.norm(vel, axis=1); nf = np.linalg.norm(fh, axis=1)
fast = (speed > 0.3) & (nf > 0.3)
cosang = np.sum(vel[fast] * fh[fast], 1) / (nv[fast] * nf[fast])
ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))

print(f"vertical axis: {'XYZ'[ax]} (path variance {var[ax]:.3f} vs others {np.sort(var)[1]:.2f}/{np.sort(var)[2]:.2f})")
print(f"path {seg.sum():.1f} m in {t_s[-1]-t_s[0]:.0f} s; speed mean {speed.mean():.2f} median {np.median(speed):.2f} m/s")
print(f"camera height above floor (m, est): mean {height_m.mean():.2f} p10 {np.percentile(height_m,10):.2f} p90 {np.percentile(height_m,90):.2f} std {height_m.std():.2f}")
print(f"pauses (speed<0.05): {np.mean(speed<0.05):.0%} of segments")
print(f"head-vs-travel on clearly-moving segments (n={fast.sum()}): median {np.median(ang):.0f} deg, share>45deg {np.mean(ang>45):.0%}")
json.dump({"t_s": t_s.tolist(), "height_m": height_m.tolist(), "speed": speed.tolist(),
           "up_axis": 'XYZ'[ax]}, open(os.path.join(HERE, "series.json"), "w"), indent=0)
