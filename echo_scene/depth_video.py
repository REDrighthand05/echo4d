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


def _run_ffmpeg(args: list, hwaccel: bool = False) -> None:
    """Run ffmpeg the way the pipeline does: resolved binary, and on
    Windows with no console window and no inherited stdin. A console
    window owned by a GUI app's child can be closed or signalled
    mid-run, which kills ffmpeg with STATUS_CONTROL_C_EXIT
    (0xC000013A) on long extractions; DEVNULL stdin also stops ffmpeg
    from treating the parent's stdin as its command channel. The
    encode parameters -- and therefore the output -- are unchanged.

    hwaccel=True (extraction only) inserts the probed hardware-decode
    args before -i, and retries in software if the hardware run fails.
    """
    from .pipeline import _hwaccel_args, _tool_path

    argv = [_tool_path("ffmpeg")] + [str(a) for a in args]
    kwargs: dict = {"check": True, "stdin": subprocess.DEVNULL}
    if sys.platform.startswith("win"):
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    accel = _hwaccel_args() if hwaccel else []
    if accel and "-i" in argv:
        at = argv.index("-i")
        try:
            subprocess.run(argv[:at] + accel + argv[at:], **kwargs)
            return
        except subprocess.CalledProcessError:
            pass  # fall through to the software run below
    subprocess.run(argv, **kwargs)


def _report(progress, stage: str, done: int, total: int) -> None:
    if progress is None:
        return
    try:
        progress(stage, done, total)
    except Exception:
        pass  # a broken progress sink must never break the render


def make_depth_video(video: Path, out_mp4: Path, fps: int = 8,
                     progress=None) -> dict:
    """progress, when given, is called as progress(stage, done, total)
    with stage in {"frames", "depth", "encode"} so a host UI can show
    real progress instead of a frozen bar."""
    from .depth import estimate_depth, save_depth_preview
    from .paths import require_depth_model

    video = Path(video)
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = out_mp4.parent / (out_mp4.stem + "_frames")
    frames_dir.mkdir(exist_ok=True)

    _run_ffmpeg(
        ["-y", "-v", "error", "-i", str(video),
         "-vf", f"fps={fps}", "-q:v", "2", str(frames_dir / "f_%05d.jpg")],
        hwaccel=True)
    frames = sorted(frames_dir.glob("f_*.jpg"))
    if not frames:
        return {"ok": False, "reason": "no frames extracted"}
    _report(progress, "frames", len(frames), len(frames))

    model = require_depth_model(prefer_base=False)
    preview_dir = out_mp4.parent / (out_mp4.stem + "_depth")
    preview_dir.mkdir(exist_ok=True)
    # Batched inference (DS kernel, merge-gated against the single-frame
    # results in tests/test_depth_batch.py); any batch-level failure
    # falls back to the original per-frame loop, and a frame the batch
    # could not do is retried singly below.
    depths = None
    try:
        from .depth import _get_session
        from .depth_batch import estimate_depth_batch

        depths = estimate_depth_batch(frames, _get_session(model),
                                      batch_size=4)
        if depths is None or len(depths) != len(frames):
            depths = None
    except Exception:
        depths = None
    for i, frame in enumerate(frames):
        depth = depths[i] if depths is not None else None
        if depth is None:
            depth = estimate_depth(frame, model)
        save_depth_preview(depth, preview_dir / f"d_{i + 1:05d}.png")
        _report(progress, "depth", i + 1, len(frames))

    _report(progress, "encode", len(frames), len(frames))
    _run_ffmpeg(
        ["-y", "-v", "error", "-framerate", str(fps),
         "-i", str(preview_dir / "d_%05d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         str(out_mp4)])
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
