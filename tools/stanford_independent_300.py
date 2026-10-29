"""Independently reconstruct Stanford Lounge from 300 actual RGB-D frames.

This is a separate, genuine 300-frame pose-estimation and Mesh experiment.
It reuses the pinned complete Open3D system from #121 without modifying
the concurrent 600-frame implementation. No 3000/600-frame pose prior.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image, ImageDraw
import stanford_independent_600 as base


def compose_300(output: Path) -> Path:
    """Build actual three original-camera + two free-camera review image."""
    names = [f"recorded-{i:05d}.png" for i in (0, 150, 299)] + [
        "free-1.png", "free-2.png"
    ]
    canvas = Image.new("RGB", (640, 5 * 268 + 34), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 10), "original Stanford RGB / free view", fill="white")
    draw.text((330, 10), "300 RGB-D only: independent poses + Mesh", fill="white")
    for row, name in enumerate(names):
        with Image.open(output / "views" / name) as source:
            panel = source.convert("RGB")
            if panel.size != (640, 268) and panel.size != (320, 268):
                raise ValueError(f"Unexpected view size for {name}: {panel.size}")
            y = 34 + row * 268
            if panel.width == 640:
                canvas.paste(panel.crop((0, 28, 320, 268)), (0, y + 28))
                canvas.paste(panel.crop((320, 28, 640, 268)), (320, y + 28))
            else:
                canvas.paste(panel.crop((0, 28, 320, 268)), (320, y + 28))
            draw.text((6, y + 5), name[:-4], fill="white")
    path = output / "source-independent300-3plus2.png"
    canvas.save(path)
    return path


def compare_600_300(old: Path, new: Path) -> Path:
    """Preserve a directly visible same-room comparison, not a scorecard."""
    with Image.open(old) as old_im, Image.open(new) as new_im:
        a, b = old_im.convert("RGB"), new_im.convert("RGB")
        if a.size != b.size:
            raise ValueError(f"View panels differ: {a.size} vs {b.size}")
        out = Image.new("RGB", (a.width + b.width, a.height + 30), (18, 18, 18))
        out.paste(a, (0, 30))
        out.paste(b, (a.width, 30))
        d = ImageDraw.Draw(out)
        d.text((6, 8), "600 RGB-D: independent camera trajectory", fill="white")
        d.text((a.width + 6, 8), "300 RGB-D: independent camera trajectory", fill="white")
        path = new.parent / "independent600-vs-300-five-views.png"
        out.save(path)
        return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare-with", type=Path)
    args = parser.parse_args()
    if args.compare_with:
        if not args.output:
            parser.error("--output is required for comparison")
        print(compare_600_300(args.compare_with, args.output / "source-independent300-3plus2.png"))
    else:
        if not args.upstream or not args.output:
            parser.error("--upstream and --output are required")
        base.compose = compose_300
        base.run(args.upstream, args.output, sample_count=300)
