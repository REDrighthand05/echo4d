#!/usr/bin/env python3
"""V3 assets: re-sample hero photos and route cards in TRUE colour
(v2 baked in a blue-darkening grade + border blend and sampled the
photos so sparsely that they read as black dotted slabs).

- photos: 136x240 samples per frame (2x v2 each axis), raw frame RGB,
  smootherstep feather (edge 0.17) for the black-gradient transparent
  edges, plus per-frame world step vectors (du, dv) so the renderer
  can quincunx-fill the photo plane.
- cards: 48x84 samples per pose (4x v2), raw RGB, same feather,
  per-card step vectors.
Geometry formulas mirror prep_data.py exactly (same planes).
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from PIL import Image

APP = Path(__file__).resolve().parent.parent
JOB = APP / 'jobs/room2_trim_acceptance'
WORK = Path(__file__).resolve().parent / 'assets'
D = np.load(WORK / 'data.npz')
FRAME_GEO = json.loads((WORK / 'frame_geo.json').read_text())
STEMS = json.loads((WORK / 'stems.json').read_text())
frames_dir = JOB / 'frames'


def feather_axis(u, edge=0.17):
    x = np.clip(np.minimum(u, 1 - u) / edge, 0, 1)
    return x * x * x * (x * (x * 6 - 15) + 10)  # smootherstep


# ---------------------------------------------------------------- photos
PH_NU, PH_NV = 136, 240
iu = (np.arange(PH_NU) + 0.5) / PH_NU
iv = (np.arange(PH_NV) + 0.5) / PH_NV
PH_FEATHER = (feather_axis(iv)[:, None] * feather_axis(iu)[None, :]).reshape(-1)

photo_pts, photo_rgb, photo_owner = [], [], []
photo_du, photo_dv = [], []
for j, m in enumerate(FRAME_GEO):
    stem = m['stem']
    img = Image.open(frames_dir / f'{stem}.jpg').convert('RGB').resize(
        (PH_NU, PH_NV), Image.LANCZOS)
    a = np.asarray(img)  # TRUE colour: no grade, no border blend
    C = np.array(m['centre']); X = np.array(m['x']); Y = np.array(m['y']); Z = np.array(m['z'])
    d = m['dmed']; f = m['f']
    gu, gv = np.meshgrid((np.arange(PH_NU) + 0.5) / PH_NU * m['W'],
                         (np.arange(PH_NV) + 0.5) / PH_NV * m['H'])
    lx = (gu - m['cx']) / f * d; ly = (gv - m['cy']) / f * d
    P = C[None, None, :] + Z[None, None, :] * d + X[None, None, :] * lx[..., None] \
        + Y[None, None, :] * ly[..., None]
    photo_pts.append(P.reshape(-1, 3))
    photo_rgb.append(a.reshape(-1, 3))
    photo_owner.append(np.full(PH_NU * PH_NV, j, dtype=np.int64))
    photo_du.append(X * (d / f) * (m['W'] / PH_NU))
    photo_dv.append(Y * (d / f) * (m['H'] / PH_NV))
photo_pts = np.concatenate(photo_pts); photo_rgb = np.concatenate(photo_rgb)
photo_owner = np.concatenate(photo_owner)
print('photo points', photo_pts.shape, flush=True)

# ----------------------------------------------------------------- cards
CARD_NU, CARD_NV = 48, 84
card_c = D['card_c'].astype(np.float64); card_x = D['card_x'].astype(np.float64)
card_y = D['card_y'].astype(np.float64)
NC = len(card_c)
iu = (np.arange(CARD_NU) + 0.5) / CARD_NU
iv = (np.arange(CARD_NV) + 0.5) / CARD_NV
CARD_FEATHER = (feather_axis(iv)[:, None] * feather_axis(iu)[None, :]).reshape(-1)
hw, hh = 0.34, 0.34 / (1080 / 1920)
card_pts, card_rgb = [], []
card_du, card_dv = [], []
for i, stem in enumerate(STEMS['cards']):
    img = Image.open(frames_dir / f'{stem}.jpg').convert('RGB').resize(
        (CARD_NU, CARD_NV), Image.LANCZOS)
    a = np.asarray(img)  # TRUE colour
    gu, gv = np.meshgrid((np.arange(CARD_NU) + 0.5) / CARD_NU,
                         (np.arange(CARD_NV) + 0.5) / CARD_NV)
    P = (card_c[i][None, None, :] + card_x[i][None, None, :] * ((2 * gu - 1) * hw)[..., None]
         + card_y[i][None, None, :] * ((2 * gv - 1) * hh)[..., None])
    card_pts.append(P.reshape(-1, 3))
    card_rgb.append(a.reshape(-1, 3))
    card_du.append(card_x[i] * (2 * hw / CARD_NU))
    card_dv.append(card_y[i] * (2 * hh / CARD_NV))
card_pts = np.concatenate(card_pts); card_rgb = np.concatenate(card_rgb)
card_owner = np.repeat(np.arange(NC), CARD_NU * CARD_NV)
print('card points', card_pts.shape, flush=True)

np.savez(WORK / 'v3_assets.npz',
         photo_pts=photo_pts.astype(np.float32), photo_rgb=photo_rgb.astype(np.uint8),
         photo_owner=photo_owner, photo_feather=PH_FEATHER.astype(np.float32),
         photo_du=np.array(photo_du), photo_dv=np.array(photo_dv),
         card_pts=card_pts.astype(np.float32), card_rgb=card_rgb.astype(np.uint8),
         card_owner=card_owner, card_feather=CARD_FEATHER.astype(np.float32),
         card_du=np.array(card_du), card_dv=np.array(card_dv),
         card_c=card_c)
print('SAVED v3_assets.npz', flush=True)
