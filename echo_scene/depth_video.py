"""Depth video: turn a video into its black-and-white depth version.

PRESERVED FEATURE (user directive, 2026-10-31): the user loves the
depth look -- every frame converted to the greyscale depth image the
depth module produces -- and uses it creatively. Keep this callable:

    python -m echo_scene.depth_video <video> <out.mp4> [fps]

Frames are sampled at `fps` (default 8), each is run through the same
depth module the pipeline uses (DS-written wrapper, Small model for
speed -- the look the user knows from the moon previews), the preview
stills are kept next to the output, and the video is encoded from them.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


def make_depth_video(video: Path, out_mp4: Path, fps: int = 8) -> dict:
    from .depth import estimate_depth, save_depth_preview

    video = Path(video)
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = out_mp4.parent / (out_mp4.stem + "_frames")
    frames_dir.mkdir(exist_ok=True)

    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video),
         "-vf", f"fps={fps}", "-q:v", "2", str(frames_dir / "f_%05d.jpg")],
        check=True)
    frames = sorted(frames_dir.glob("f_*.jpg"))
    if not frames:
        return {"ok": False, "reason": "no frames extracted"}

    models_dir = Path(__file__).resolve().parent.parent / "models"
    model = models_dir / "depth_anything_v2_small.onnx"
    if not model.exists():
        model = models_dir / "depth_anything_v2_base.onnx"
    preview_dir = out_mp4.parent / (out_mp4.stem + "_depth")
    preview_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        depth = estimate_depth(frame, model)
        save_depth_preview(depth, preview_dir / f"d_{i + 1:05d}.png")

    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-framerate", str(fps),
         "-i", str(preview_dir / "d_%05d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         str(out_mp4)],
        check=True)
    return {"ok": True, "frames": len(frames), "fps": fps,
            "video": str(out_mp4), "depth_frames_dir": str(preview_dir)}


def main(argv) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    fps = int(argv[3]) if len(argv) > 3 else 8
    info = make_depth_video(Path(argv[1]), Path(argv[2]), fps)
    print(info)
    return 0 if info.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
