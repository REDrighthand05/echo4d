#!/usr/bin/env python3
"""Shared promo overlays: beat-map loading plus the T2 sentimental
treatment — warm grade and the glowing opening lines — kept verbatim
from the user-approved v2/v3 look. Used by render_track.py."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

from batch_lib import text_layer, paste_center, smooth

BM = Path(__file__).resolve().parent.parent / 'experiments' / 'batch_music'


def accents(n):
    d = json.loads((BM / f'track{n}_beatmap.json').read_text())
    return [(a['time'], a['strength']) for a in d['accents']], d['duration_s']


def warm_grade(img):
    f = img.astype(np.float32)
    lum = (f @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    f = f * 0.92 + lum * 0.08
    f *= np.array([1.03, 1.0, 0.95], np.float32)
    return np.clip(f, 0, 255).astype(np.uint8)


def opening_overlay(img, t):
    a1 = smooth(0.5, 1.6, t) * (1 - smooth(3.4, 4.2, t))
    a2 = smooth(1.8, 2.8, t) * (1 - smooth(3.4, 4.2, t))
    if a1 <= 0.003 and a2 <= 0.003:
        return img
    base = Image.fromarray(img).convert('RGBA')
    if a1 > 0.003:
        lay, _, _ = text_layer('回忆很珍贵', 56, (220, 236, 255), tracking=10,
                               bold=True, glow_alpha=0.7, glow_radius=16)
        paste_center(base, lay, 960, 462, alpha=a1)
    if a2 > 0.003:
        lay, _, _ = text_layer('把这一段生活，好好留住', 44, (214, 231, 252), tracking=6,
                               bold=True, glow_alpha=0.7, glow_radius=16)
        paste_center(base, lay, 960, 556, alpha=a2)
    return np.asarray(base.convert('RGB'))
