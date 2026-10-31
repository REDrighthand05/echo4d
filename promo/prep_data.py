#!/usr/bin/env python3
"""One-time data prep for the batch promos: parse preview.build_preview_data
(master schedule) blob into arrays, rebuild per-point frame provenance,
port the page's grid construction, and pre-sample cards/photos as points.
Run with .venv-mus python."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from PIL import Image
from echo_scene import preview

APP = Path(__file__).resolve().parent.parent
JOB = APP / 'jobs/room2_trim_acceptance'
PROV = APP / 'jobs/room2_trim2'
OUT = Path(__file__).resolve().parent / 'assets'

man, blob, atlas_jpeg = preview.build_preview_data(JOB, PROV, promo=False)
NP_ = man['n_points']; NC = man['n_cards']; NV = man['mesh_verts']; NT = man['mesh_tris']
OUT.mkdir(parents=True, exist_ok=True)
print('points', NP_, 'cards', NC, 'diss', man['n_frames'], 'mesh', NV, NT, flush=True)

off = 0
def take_f32(n):
    global off
    a = np.frombuffer(blob, dtype='<f4', count=n, offset=off).copy(); off += 4 * n; return a
def take_u8(n):
    global off
    a = np.frombuffer(blob, dtype=np.uint8, count=n, offset=off).copy(); off += n; off = (off + 3) & ~3; return a
def take_u32(n):
    global off
    a = np.frombuffer(blob, dtype='<u4', count=n, offset=off).copy(); off += 4 * n; return a
pts = take_f32(3 * NP_).reshape(-1, 3).astype(np.float64)
start_pos = take_f32(3 * NP_).reshape(-1, 3).astype(np.float64)
rgb = take_u8(3 * NP_).reshape(-1, 3)
take_f32(NP_); take_f32(NP_)
burst = take_f32(3 * NP_).reshape(-1, 3).astype(np.float64)
card_c = take_f32(3 * NC).reshape(-1, 3).astype(np.float64)
card_x = take_f32(3 * NC).reshape(-1, 3).astype(np.float64)
card_y = take_f32(3 * NC).reshape(-1, 3).astype(np.float64)
take_f32(NC)
route_pos = take_f32(3 * NC).reshape(-1, 3).astype(np.float64)
route_frac = take_f32(NC).astype(np.float64)
mesh_pos = take_f32(3 * NV).reshape(-1, 3).astype(np.float64) if NV else np.zeros((0, 3))
mesh_col = take_u8(3 * NV).reshape(-1, 3) if NV else np.zeros((0, 3), np.uint8)
print('blob consumed', off, 'of', len(blob), flush=True)

poses = json.loads((JOB / 'poses.json').read_text())
stems_sorted = sorted(Path(k).stem for k in poses)
depth_dir = JOB / 'depth'
cand = [s for s in stems_sorted if (depth_dir / f'{s}.npy').exists()]
id_to_stem = {k: s for k, s in enumerate(cand)}
frame_ids = np.load(PROV / 'points_frame_ids.npy').astype(np.int64)
assert len(frame_ids) == NP_
used_ids = sorted(set(int(i) for i in np.unique(frame_ids)) & set(id_to_stem))
diss_stems = [id_to_stem[k] for k in used_ids]
assert len(diss_stems) == man['n_frames']
for j, m in enumerate(man['frames']):
    assert m['stem'] == diss_stems[j]
    assert m['n_points'] == int((frame_ids == used_ids[j]).sum())
frame_geo = man['frames']
print('provenance ok; diss stems', len(diss_stems), flush=True)

bmin = np.array(man['bbox_min']); bmax = np.array(man['bbox_max']); floor_y = man['floor_y']
cx, cz = (bmin[0] + bmax[0]) / 2, (bmin[2] + bmax[2]) / 2
hx, hz = (bmax[0] - bmin[0]) / 2 + 1.2, (bmax[2] - bmin[2]) / 2 + 1.2
top_y = floor_y + (bmax[1] - floor_y)
lines, shades = [], []
def line(a, b, s):
    lines.append((a, b)); shades.append(s)
step = 0.75
x = cx - hx
while x <= cx + hx + 1e-4:
    line([x, floor_y, cz - hz], [x, floor_y, cz + hz], 0.9 if abs(x - cx) < step / 2 else 0.35)
    x += step
z = cz - hz
while z <= cz + hz + 1e-4:
    line([cx - hx, floor_y, z], [cx + hx, floor_y, z], 0.9 if abs(z - cz) < step / 2 else 0.35)
    z += step
cs = [[cx - hx, cz - hz], [cx + hx, cz - hz], [cx + hx, cz + hz], [cx - hx, cz + hz]]
for i in range(4):
    a, b = cs[i], cs[(i + 1) % 4]
    line([a[0], floor_y, a[1]], [b[0], floor_y, b[1]], 1.0)
    line([a[0], top_y, a[1]], [b[0], top_y, b[1]], 0.85)
    line([a[0], floor_y, a[1]], [a[0], top_y, a[1]], 0.85)
    length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    s = step
    while s < length - 1e-4:
        u = s / length
        line([a[0] + (b[0] - a[0]) * u, floor_y, a[1] + (b[1] - a[1]) * u],
             [a[0] + (b[0] - a[0]) * u, top_y, a[1] + (b[1] - a[1]) * u], 0.30)
        s += step
    y = floor_y + step
    while y < top_y - 1e-4:
        line([a[0], y, a[1]], [b[0], y, b[1]], 0.22)
        y += step
seed = 987654321
def rnd():
    global seed
    seed = (seed * 1103515245 + 12345) & 0x7fffffff
    return seed / 0x7fffffff
ext_l = float(max(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]))
for i in range(150):
    push = 0.45 + 0.75 * rnd()
    px = cx + (rnd() * 2 - 1) * hx * push
    pz = cz + (rnd() * 2 - 1) * hz * push
    py = floor_y + rnd() * (top_y - floor_y)
    dx, dy, dz = rnd() * 2 - 1, (rnd() * 2 - 1) * 0.55, rnd() * 2 - 1
    dl = np.hypot(dx, np.hypot(dy, dz)) or 1.0
    dx, dy, dz = dx / dl, dy / dl, dz / dl
    L = ext_l * (0.04 + 0.20 * rnd() ** 1.5)
    s = 0.25 + 0.75 * rnd() * rnd()
    line([px - dx * L / 2, py - dy * L / 2, pz - dz * L / 2],
         [px + dx * L / 2, py + dy * L / 2, pz + dz * L / 2], s)
grid_lines = np.array([p for pair in lines for p in pair], dtype=np.float64).reshape(-1, 2, 3)
grid_shade = np.array(shades, dtype=np.float64)
print('grid lines', len(grid_lines), flush=True)

def grade(img_f):
    lum = (img_f @ np.array([0.299, 0.587, 0.114]))[..., None]
    c = img_f * (1 - 0.30) + (lum * np.array([0.80, 0.94, 1.22])) * 0.30
    return np.clip(c, 0, 255)

frames_dir = JOB / 'frames'
CARD_NU, CARD_NV = 12, 21
card_pts, card_rgb = [], []
hw, hh = 0.34, 0.34 / (1080 / 1920)
for i, stem in enumerate(stems_sorted):
    img = Image.open(frames_dir / f'{stem}.jpg').convert('RGB').resize(
        (CARD_NU, CARD_NV), Image.LANCZOS)
    a = np.asarray(img).astype(np.float64)
    a = grade(a)
    iu = (np.arange(CARD_NU) + 0.5) / CARD_NU
    iv = (np.arange(CARD_NV) + 0.5) / CARD_NV
    ex = np.minimum(iu, 1 - iu)[None, :].repeat(CARD_NV, 0)
    ey = np.minimum(iv, 1 - iv)[:, None].repeat(CARD_NU, 1)
    e = np.minimum(ex, ey)
    tt = np.clip((e - 0.015) / (0.05 - 0.015), 0, 1)
    border = 1 - (tt * tt * (3 - 2 * tt))
    a = a * (1 - border[..., None] * 0.55) + np.array([0.35, 0.65, 1.0]) * 255 * (border[..., None] * 0.55)
    gu, gv = np.meshgrid((np.arange(CARD_NU) + 0.5) / CARD_NU, (np.arange(CARD_NV) + 0.5) / CARD_NV)
    P = (card_c[i][None, None, :] + card_x[i][None, None, :] * ((2 * gu - 1) * hw)[..., None]
         + card_y[i][None, None, :] * ((2 * gv - 1) * hh)[..., None])
    card_pts.append(P.reshape(-1, 3))
    card_rgb.append((a * 0.82).reshape(-1, 3))
card_pts = np.concatenate(card_pts); card_rgb = np.concatenate(card_rgb)
card_owner = np.repeat(np.arange(NC), CARD_NU * CARD_NV)
print('card points', card_pts.shape, flush=True)

PH_NU, PH_NV = 68, 120
photo_pts, photo_rgb, photo_owner = [], [], []
for j, m in enumerate(frame_geo):
    stem = m['stem']
    img = Image.open(frames_dir / f'{stem}.jpg').convert('RGB').resize(
        (PH_NU, PH_NV), Image.LANCZOS)
    a = np.asarray(img).astype(np.float64)
    a = grade(a)
    iu = (np.arange(PH_NU) + 0.5) / PH_NU
    iv = (np.arange(PH_NV) + 0.5) / PH_NV
    ex = np.minimum(iu, 1 - iu)[None, :].repeat(PH_NV, 0)
    ey = np.minimum(iv, 1 - iv)[:, None].repeat(PH_NU, 1)
    e = np.minimum(ex, ey)
    tt = np.clip((e - 0.004) / (0.02 - 0.004), 0, 1)
    border = 1 - (tt * tt * (3 - 2 * tt))
    a = a * (1 - border[..., None] * 0.8) + np.array([0.45, 0.72, 1.0]) * 255 * (border[..., None] * 0.8)
    C = np.array(m['centre']); X = np.array(m['x']); Y = np.array(m['y']); Z = np.array(m['z'])
    d = m['dmed']; f = m['f']
    gu, gv = np.meshgrid((np.arange(PH_NU) + 0.5) / PH_NU * m['W'], (np.arange(PH_NV) + 0.5) / PH_NV * m['H'])
    lx = (gu - m['cx']) / f * d; ly = (gv - m['cy']) / f * d
    P = C[None, None, :] + Z[None, None, :] * d + X[None, None, :] * lx[..., None] + Y[None, None, :] * ly[..., None]
    photo_pts.append(P.reshape(-1, 3))
    photo_rgb.append(a.reshape(-1, 3))
    photo_owner.append(np.full(PH_NU * PH_NV, j, dtype=np.int64))
photo_pts = np.concatenate(photo_pts); photo_rgb = np.concatenate(photo_rgb)
photo_owner = np.concatenate(photo_owner)
print('photo points', photo_pts.shape, flush=True)

np.savez(OUT / 'data.npz',
    pts=pts.astype(np.float32), start_pos=start_pos.astype(np.float32), rgb=rgb,
    burst=burst.astype(np.float32), frame_ids=frame_ids,
    used_ids=np.array(used_ids), card_c=card_c, card_x=card_x, card_y=card_y,
    route_pos=route_pos, route_frac=route_frac, mesh_pos=mesh_pos.astype(np.float32),
    mesh_col=mesh_col, grid_lines=grid_lines.astype(np.float32),
    grid_shade=grid_shade, card_pts=card_pts.astype(np.float32),
    card_rgb=card_rgb.astype(np.float32), card_owner=card_owner,
    photo_pts=photo_pts.astype(np.float32), photo_rgb=photo_rgb.astype(np.float32),
    photo_owner=photo_owner, bmin=bmin, bmax=bmax, floor_y=np.float64(floor_y))
(OUT / 'frame_geo.json').write_text(json.dumps(frame_geo))
(OUT / 'stems.json').write_text(json.dumps({'cards': stems_sorted, 'diss': diss_stems}))
print('SAVED', flush=True)
