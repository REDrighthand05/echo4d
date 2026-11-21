"""Depth video generation kernel interface."""

from pathlib import Path
from typing import Optional
from echo_engine.types import ProgressCallback


def run_depth_video(video: Path, progress: Optional[ProgressCallback] = None) -> Path:
    """Turn a video into its black-and-white depth version.

    The mp4 lands next to the source video with incremental suffix if needed.
    """
    from echo_scene.depth_video import make_depth_video

    out = video.with_name(f"{video.stem}_depth.mp4")
    base = out
    n = 1
    while out.exists():
        n += 1
        out = base.with_name(f"{base.stem}_{n}.mp4")

    info = make_depth_video(video, out, progress=progress)
    if not isinstance(info, dict) or info.get("ok") is False:
        reason = info.get("reason", "未知原因") if isinstance(info, dict) else "未知原因"
        raise RuntimeError(f"生成深度视频失败：{reason}")
    return out
