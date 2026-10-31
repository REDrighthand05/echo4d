#!/usr/bin/env python3
"""New-look assets: (1) wireframe lines that lie ON the actual surfaces
(edges of a coarsely decimated job mesh -- no box cage), (2) a feather
weight per sampled photo point (black-gradient transparent edges)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from echo_scene import preview

APP = Path(__file__).resolve().parent.parent
JOB = APP / 'jobs/room2_trim_acceptance'
OUT = Path(__file__).resolve().parent / 'assets'

mv, mc = preview._read_points_bin(JOB / 'mesh_vertices.bin')
tris = np.fromfile(JOB / 'mesh_indices.bin', dtype='<u4').reshape(-1, 3)
dv, dc, dt, full = preview._decimate_mesh(mv, mc, tris, max_tris=3500)
print('decimated', full, '->', len(dt), 'tris')
edges = set()
for a, b, c in dt:
    for p, q in ((a, b), (b, c), (c, a)):
        edges.add((min(p, q), max(p, q)))
E = np.array(sorted(edges), dtype=np.int64)
P0, P1 = dv[E[:, 0]], dv[E[:, 1]]
length = np.linalg.norm(P1 - P0, axis=1)
keep = length <= 3.0
print('edges', len(E), 'kept (len<=3.0)', int(keep.sum()), 'median len',
      round(float(np.median(length)), 3))
surf_lines = np.stack([P0[keep], P1[keep]], axis=1).astype(np.float32)

PH_NU, PH_NV = 68, 120
iu = (np.arange(PH_NU) + 0.5) / PH_NU
iv = (np.arange(PH_NV) + 0.5) / PH_NV
def feather_axis(u, edge=0.17):
    x = np.clip(np.minimum(u, 1 - u) / edge, 0, 1)
    return x * x * x * (x * (x * 6 - 15) + 10)  # smootherstep
feather = (feather_axis(iv)[:, None] * feather_axis(iu)[None, :]).reshape(-1)
print('feather range', feather.min(), feather.max(), 'mean', round(float(feather.mean()), 3))
np.savez(OUT / 'surface.npz', surf_lines=surf_lines,
         photo_feather=feather.astype(np.float32))
print('saved surface.npz')
