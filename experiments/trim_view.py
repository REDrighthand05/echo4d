"""Perspective point-cloud views at one shared framing (v6-preview style).

Orthographic point projection like experiments/sgbm_exp.py's renderer
(az=35, el=22, 900x900, black background), but vectorized and with center
+ radius computed ONCE from the baseline cloud so all variants share the
exact framing and shrinkage stays visible.

Usage: python experiments/trim_view.py <baseline_job> <job> <out.png> [<job> <out.png> ...]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_top import load_cloud  # noqa: E402

SIZE = 900


def render(pts, cols, center, radius, path):
    az, el = np.radians(35.0), np.radians(22.0)
    fwd = np.array([np.cos(el) * np.sin(az), -np.sin(el), np.cos(el) * np.cos(az)])
    right = np.array([np.cos(az), 0.0, -np.sin(az)])
    up = np.cross(right, fwd)
    rel = pts - center
    x, y, z = rel @ right, rel @ up, rel @ fwd
    s = SIZE * 0.46 / radius
    px = (SIZE / 2 + x * s).astype(np.int32)
    py = (SIZE / 2 - y * s).astype(np.int32)
    ok = (px >= 0) & (px < SIZE) & (py >= 0) & (py < SIZE)
    px, py, z, cols = px[ok], py[ok], z[ok], cols[ok]
    order = np.argsort(z, kind="stable")  # far first, near painted last
    img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    img[py[order], px[order]] = cols[order]
    Image.fromarray(img).save(path)
    print(f"{path}: {len(pts)} points")


def main() -> None:
    args = sys.argv[1:]
    base_pts, _ = load_cloud(Path(args[0]))
    center = (base_pts.max(0) + base_pts.min(0)) / 2
    radius = float(np.linalg.norm(base_pts.max(0) - base_pts.min(0))) / 2
    for i in range(1, len(args), 2):
        pts, cols = load_cloud(Path(args[i]))
        render(pts, cols, center, radius, args[i + 1])


if __name__ == "__main__":
    main()
