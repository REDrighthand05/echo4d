"""Bench for the coordinated icon set (echo_app.icons).

Run as a script: .venv/bin/python tests/test_icons.py
Every icon must render as a real drawn pictogram (not an empty
image), carry the ink-blue of the skin, and the application icon
must rebuild as a proper multi-size .ico with the pearl string
visible on its frosted square.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image

from echo_app.icons import ICON_NAMES, build_app_icon, icon_pil

fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""))
    if not ok:
        fails.append(name)


for name in ICON_NAMES:
    for dark in (False, True):
        img = icon_pil(name, 84, dark=dark)
        arr = np.asarray(img).astype(int)
        alpha = arr[:, :, 3]
        cover = float((alpha > 30).mean())
        blue = int(((arr[:, :, 2] - arr[:, :, 0] > 35)
                    & (alpha > 30)).sum())
        check(f"icon {name} dark={dark}", img.mode == "RGBA"
              and 0.03 < cover < 0.92 and blue > 60,
              f"cover={cover:.2f} blue={blue}")

with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "echo.ico")
    build_app_icon(path)
    ico = Image.open(path)
    sizes = ico.info.get("sizes", set())
    check("ico sizes", (256, 256) in sizes and (32, 32) in sizes,
          f"{sorted(sizes)}")
    ico.size = (256, 256)
    arr = np.asarray(ico.convert("RGB")).astype(int)
    blue = int(((arr[:, :, 2] - arr[:, :, 0] > 60)
                & (arr[:, :, 0] < 130)).sum())
    pale = int(((np.abs(arr - np.array([233, 238, 244])) < 18)
                .all(axis=2)).sum())
    check("ico carries pearl string on frost", blue > 2500 and pale > 8000,
          f"blue={blue} pale={pale}")

print("FAILURES:", len(fails))
sys.exit(1 if fails else 0)
