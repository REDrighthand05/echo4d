#!/usr/bin/env python3
"""V2 environment: a GLOBAL blue lattice that fills the whole space
around the room (independent of the scene's own surfaces -- Detroit
android-vision style), built on robust point-cloud extents.
Output grid2.npz: lat_lines (N,2,3) float32, lat_shade (N,) float32.
The 150 scattered shards keep coming from data.npz GRID_LINES[-150:]."""
from pathlib import Path
import numpy as np

WORK = Path(__file__).resolve().parent / 'assets'
D = np.load(WORK / 'data.npz')
pts = D['pts'].astype(np.float64)
floor_y = float(D['floor_y'])

lo = np.percentile(pts, 0.5, axis=0)
hi = np.percentile(pts, 99.5, axis=0)
x0, x1 = lo[0] - 0.9, hi[0] + 0.9
z0, z1 = lo[2] - 0.9, hi[2] + 0.9
y0 = floor_y - 0.15
y1 = hi[1] + 0.7
cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
step = 0.8
print(f'lattice volume x[{x0:.2f},{x1:.2f}] y[{y0:.2f},{y1:.2f}] z[{z0:.2f},{z1:.2f}] step {step}')

lines, shades = [], []
def line(a, b, s):
    lines.append((a, b)); shades.append(s)

def axis_seq(a, b, st):
    v = [a]
    while v[-1] + st < b - 1e-6:
        v.append(v[-1] + st)
    v.append(b)
    return v

xs = axis_seq(x0, x1, step); zs = axis_seq(z0, z1, step); ys = axis_seq(y0, y1, step)
# floor + ceiling planes (grids lying across the whole footprint)
for x in xs:
    c = abs(x - cx) < step * 0.51
    line([x, y0, z0], [x, y0, z1], 0.85 if c else 0.5)
    line([x, y1, z0], [x, y1, z1], 0.22)
for z in zs:
    c = abs(z - cz) < step * 0.51
    line([x0, y0, z], [x1, y0, z], 0.85 if c else 0.5)
    line([x0, y1, z], [x1, y1, z], 0.22)
# four walls of the volume: verticals + horizontals
for x in xs:
    line([x, y0, z0], [x, y1, z0], 0.34); line([x, y0, z1], [x, y1, z1], 0.34)
for z in zs:
    line([x0, y0, z], [x0, y1, z], 0.34); line([x1, y0, z], [x1, y1, z], 0.34)
for y in ys[1:-1]:
    line([x0, y, z0], [x1, y, z0], 0.26); line([x0, y, z1], [x1, y, z1], 0.26)
    line([x0, y, z0], [x0, y, z1], 0.26); line([x1, y, z0], [x1, y, z1], 0.26)
# interior cross-sections through the middle of the volume (very dim):
# the grid visibly fills the SPACE, not just its shell
for z in zs:
    line([cx, y0, z], [cx, y1, z], 0.15)
for y in ys:
    line([cx, y, z0], [cx, y, z1], 0.15)
for x in xs:
    line([x, y0, cz], [x, y1, cz], 0.15)
for y in ys:
    line([x0, y, cz], [x1, y, cz], 0.15)

lat_lines = np.array([p for pair in lines for p in pair], dtype=np.float64).reshape(-1, 2, 3)
lat_shade = np.array(shades, dtype=np.float64)
print('lattice lines', len(lat_lines))
np.savez(WORK / 'grid2.npz', lat_lines=lat_lines.astype(np.float32),
         lat_shade=lat_shade.astype(np.float32))
print('saved grid2.npz')
