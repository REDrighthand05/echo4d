"""Bench for the living graffiti (echo_app.glass.ParticleField).

Run as a script: .venv/bin/python tests/test_glass_particles.py
Checks: the strokes become a 3D bead volume; idle sway stays a
breath, not a wander; the mouse pushes beads away, near beads
wider than far ones; everything settles home afterwards; the
frosted render still reads as the blue graffiti.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from echo_app.glass import ParticleField

fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""))
    if not ok:
        fails.append(name)


f = ParticleField(1020, 800)
check("particle count sane", 180 <= f.count <= 900, f"count={f.count}")
check("depth spread is real", float(np.std(f.hz)) > 0.30,
      f"std(hz)={np.std(f.hz):.3f}")
check("homes inside window",
      bool((f.home[:, 0] > -70).all() and (f.home[:, 0] < 1090).all()
           and (f.home[:, 1] > -20).all() and (f.home[:, 1] < 820).all()))

# Idle: a breath, not a wander.
for _ in range(80):
    f.step(0.05, None)
disp = np.hypot(*(f.pos - f.home).T)
check("idle sway bounded", float(disp.max()) < 14.0,
      f"max={disp.max():.1f}px")

# Dodge: park the cursor in the middle of the M.
mouse = (275.0, 190.0)
home_dist = np.hypot(*(f.home - np.array(mouse)).T)
inner = home_dist < 55
before = np.hypot(*(f.pos - np.array(mouse)).T)[inner].mean()
for _ in range(25):
    f.step(0.05, mouse)
after = np.hypot(*(f.pos - np.array(mouse)).T)[inner].mean()
check("beads dodge the mouse", after - before >= 18.0,
      f"mean dist {before:.0f} -> {after:.0f}px")

# Parallax, mechanism-level: two beads at the same spot, one
# floating near, one far -- the near one must dodge wider.
f2 = ParticleField(200, 200)
for idx, zval in ((0, 0.8), (1, -0.8)):
    f2.home[idx] = (100.0, 60.0)
    f2.pos[idx] = (100.0, 60.0)
    f2.hz[idx] = zval
    f2.z[idx] = zval
    f2.vel[idx] = 0.0
for _ in range(3):
    f2.step(0.05, (100.0, 110.0))
d_near = float(np.hypot(*(f2.pos[0] - (100.0, 60.0))))
d_far = float(np.hypot(*(f2.pos[1] - (100.0, 60.0))))
check("parallax: near dodges wider",
      d_near > d_far * 1.25 and f2.pos[0][1] < 60.0,
      f"near={d_near:.1f} far={d_far:.1f}px")

# Scatter is volumetric: the burst runs through depth, not just
# sideways -- depth spread must blow open while the mouse sits.
f3 = ParticleField(1020, 800)
band = np.hypot(*(f3.home - np.array(mouse)).T) < 90
hz0 = f3.hz[band].copy()
for _ in range(14):
    f3.step(0.05, mouse)
dz = np.abs(f3.z[band] - hz0)
check("scatter bursts through depth",
      float(dz.mean()) > 0.30
      and float(f3.z[band].std()) > float(hz0.std()) * 1.5,
      f"mean|dz|={dz.mean():.2f} std {hz0.std():.2f}->{f3.z[band].std():.2f}")

# Settle: cursor gone, everyone drifts home.
for _ in range(300):
    f.step(0.05, None)
disp = np.hypot(*(f.pos - f.home).T)
check("settles back home", float(disp.mean()) < 7.0,
      f"mean={disp.mean():.1f}px")

# Render: frosted, right size, the blue still reads through.
img = f.render()
check("render size/mode", img.size == (1020, 800) and img.mode == "RGB")
arr = np.asarray(img).astype(int)
blue = int(((arr[:, :, 2] - arr[:, :, 0]) > 22).sum())
check("graffiti reads through frost", blue > 2500, f"blue px={blue}")

# Perf: a tick must stay cheap enough for a background.
t0 = time.time()
for _ in range(20):
    f.step(0.05, mouse)
    f.render()
avg = (time.time() - t0) / 20 * 1000
check("tick cost sane", avg < 150.0, f"avg={avg:.0f}ms")

print("FAILURES:", len(fails))
sys.exit(1 if fails else 0)
