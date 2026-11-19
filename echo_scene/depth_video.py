"""Depth video: turn a video into its black-and-white depth version.

PRESERVED FEATURE (user directive, 2026-10-31): the user loves the
depth look -- every frame converted to the greyscale depth image the
depth module produces -- and uses it creatively. Keep this callable:

    python -m echo_scene.depth_video <video> <out.mp4> [fps]

Frames are sampled at `fps` (default 8), each is run through the same
depth module the pipeline uses (DS-written wrapper, Small model for
speed -- the look the user knows from the moon previews), the preview
stills are kept next to the output, and the video is encoded from them.

v0.6.32: the encode no longer writes a full-HD PNG per frame and
re-reads it. Each frame is normalized once (the same 2-98 percentile
stretch save_depth_preview uses) and the bytes go two places at once:
the preview still on disk, and straight into ffmpeg's stdin as raw
grey frames. Same picture (checked: new vs old stills pixel-identical),
about a fifth faster, and no PNG round-trip in the middle. A frame the
batch could not do is retried singly; a frame that cannot be done at
all reuses the previous frame so the job still finishes with the right
frame count (counted in "substituted").
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

_BATCH_SIZE = 8
_CHUNK_SIZE = 16


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


def _normalize_u8(depth: np.ndarray) -> np.ndarray:
    """The same 2-98 percentile stretch save_depth_preview uses,
    as an 8-bit array (shared by the pipe encode and the stills)."""
    depth_array = np.asarray(depth, dtype=np.float32)
    low, high = np.percentile(depth_array, [2.0, 98.0])
    if high > low:
        normalized = (depth_array - low) / (high - low)
    else:
        normalized = np.zeros_like(depth_array, dtype=np.float32)
    display = np.clip(normalized, 0.0, 1.0) * 255.0
    return display.astype(np.uint8)


def make_depth_video(video: Path, out_mp4: Path, fps: int = 8,
                     progress=None, keep_previews: bool = True) -> dict:
    """progress, when given, is called as progress(stage, done, total)
    with stage in {"frames", "depth", "encode"} so a host UI can show
    real progress instead of a frozen bar.

    keep_previews=True (default) still writes the depth stills next
    to the output, as before -- the app shows them while the job
    runs. Pass False for a pure-speed run with no stills on disk.
    """
    from .depth import estimate_depth
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

    # Streamed batched inference (DS kernels, merge-gated against the
    # single-frame results in tests/test_depth_batch.py). The older
    # whole-list call held every full-res depth map (~8 MB each) until
    # the last frame finished -- a two-minute video at 8 fps is ~8 GB
    # resident, and a machine that starts swapping spends its time
    # there instead of on the model. iter_depth_estimates walks the
    # list in bounded chunks and yields each frame as its chunk
    # finishes, so the preview PNG is saved and the map freed right
    # away; batch_size 8 amortises one model call over more frames,
    # which is where the GPU time goes. A frame the batch could not
    # do is retried singly below, and any batch-level failure falls
    # back to the per-frame loop -- into the same encoder, so frames
    # already written stay written and the count stays right.
    from .depth import _get_session
    from .depth_batch import iter_depth_estimates
    from .pipeline import _tool_path

    session = _get_session(model)
    width = height = None
    encoder = None
    substituted = 0
    done = 0
    prev_u8 = None
    batched = True

    def _open_encoder(w: int, h: int):
        argv = [_tool_path("ffmpeg"), "-y", "-v", "error",
                "-f", "rawvideo", "-pix_fmt", "gray",
                "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                str(out_mp4)]
        kwargs: dict = {"stdin": subprocess.PIPE}
        if sys.platform.startswith("win"):
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        return subprocess.Popen(argv, **kwargs)

    def _emit(depth, index: int) -> None:
        nonlocal width, height, encoder, prev_u8
        u8 = _normalize_u8(depth)
        if width is None:
            height, width = u8.shape
            encoder = _open_encoder(width, height)
        if u8.shape != (height, width):
            from PIL import Image
            u8 = np.asarray(
                Image.fromarray(u8, mode="L").resize(
                    (width, height), resample=Image.BILINEAR),
                dtype=np.uint8)
        if keep_previews:
            # Same bytes save_depth_preview would write: it runs the
            # identical 2-98 percentile stretch, so write the still
            # from the array we already normalized for the pipe.
            from PIL import Image
            Image.fromarray(u8, mode="L").save(
                preview_dir / f"d_{index + 1:05d}.png")
        encoder.stdin.write(np.ascontiguousarray(u8).tobytes())
        prev_u8 = u8

    try:
        for i, depth in iter_depth_estimates(
                frames, session, batch_size=_BATCH_SIZE,
                chunk_size=_CHUNK_SIZE):
            if depth is None:
                depth = estimate_depth(frames[i], model)
            if depth is None:
                raise RuntimeError(f"no depth for frame {i}")
            _emit(depth, i)
            done = i + 1
            _report(progress, "depth", done, len(frames))
    except Exception:
        batched = False
        # Resume per-frame from where streaming stopped. Frames already
        # emitted stay emitted; the encoder keeps consuming in order.
        for i in range(done, len(frames)):
            try:
                depth = estimate_depth(frames[i], model)
            except Exception:
                depth = None
            if depth is None:
                if prev_u8 is None:
                    raise RuntimeError(f"no depth for frame {i} "
                                       "and no previous frame to reuse")
                substituted += 1
                if keep_previews:
                    from PIL import Image
                    Image.fromarray(prev_u8, mode="L").save(
                        preview_dir / f"d_{i + 1:05d}.png")
                encoder.stdin.write(np.ascontiguousarray(prev_u8).tobytes())
            else:
                _emit(depth, i)
            done = i + 1
            _report(progress, "depth", done, len(frames))

    _report(progress, "encode", len(frames), len(frames))
    if encoder is not None:
        try:
            encoder.stdin.close()
        except BrokenPipeError:
            pass
        rc = encoder.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, "ffmpeg pipe encode")
    return {"ok": True, "frames": len(frames), "fps": fps,
            "video": str(out_mp4), "depth_frames_dir": str(preview_dir),
            "batched": batched, "substituted": substituted,
            "keep_previews": keep_previews}


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
