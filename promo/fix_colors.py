#!/usr/bin/env python3
"""Replace card/photo colours in data.npz with TRUE colours (user feedback:
no blue-tinted ghost grade, no blue frame border on sampled photos)."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

WORK = Path(__file__).resolve().parent / 'assets'
APP = Path(__file__).resolve().parent.parent
JOB = APP / 'jobs/room2_trim_acceptance'
frames_dir = JOB / 'frames'
stems = json.loads((WORK / 'stems.json').read_text())
frame_geo = json.loads((WORK / 'frame_geo.json').read_text())

CARD_NU, CARD_NV = 12, 21
card_rgb = []
for stem in stems['cards']:
    img = Image.open(frames_dir / f'{stem}.jpg').convert('RGB').resize(
        (CARD_NU, CARD_NV), Image.LANCZOS)
    card_rgb.append(np.asarray(img).astype(np.float64).reshape(-1, 3))
card_rgb = np.concatenate(card_rgb)

PH_NU, PH_NV = 68, 120
photo_rgb = []
for m in frame_geo:
    img = Image.open(frames_dir / f"{m['stem']}.jpg").convert('RGB').resize(
        (PH_NU, PH_NV), Image.LANCZOS)
    photo_rgb.append(np.asarray(img).astype(np.float64).reshape(-1, 3))
photo_rgb = np.concatenate(photo_rgb)

d = dict(np.load(WORK / 'data.npz'))
assert card_rgb.shape == d['card_rgb'].shape, (card_rgb.shape, d['card_rgb'].shape)
assert photo_rgb.shape == d['photo_rgb'].shape, (photo_rgb.shape, d['photo_rgb'].shape)
d['card_rgb'] = card_rgb.astype(np.float32)
d['photo_rgb'] = photo_rgb.astype(np.float32)
np.savez(WORK / 'data.npz', **d)
print('true colours written', card_rgb.shape, photo_rgb.shape)
