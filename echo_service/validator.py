"""Pre-flight validation guards for incoming video inputs and disk bounds."""

import shutil
from pathlib import Path
from typing import Optional, Set

VIDEO_SUFFIXES: Set[str] = {".mp4", ".mov", ".mkv", ".avi"}


def check_video(path: Path, jobs_root: Optional[Path] = None) -> Optional[str]:
    """Validate video file readability, container duration, and target disk space.

    Returns an error message string if invalid; None if validation passes.
    """
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        return "这个文件不是视频，换个视频文件试试。"

    if not path.is_file():
        return "没找到这个视频文件。"

    try:
        size = path.stat().st_size
        if size == 0:
            return "这个视频是空的（0 字节），换一个视频试试。"
    except OSError:
        return "没找到这个视频文件。"

    # Disk free space check (>= 500 MB required)
    if jobs_root is not None:
        try:
            target = jobs_root if jobs_root.exists() else jobs_root.parent
            usage = shutil.disk_usage(target)
            if usage.free < 500 * 1024 * 1024:
                mb = usage.free // (1024 * 1024)
                return f"磁盘空间不足（仅剩 {mb} MB），建议保留至少 500 MB 空间再开始生成。"
        except Exception:
            pass

    # Video stream validity and duration check
    try:
        from echo_scene.pipeline import _ffprobe_duration
        dur = _ffprobe_duration(path)
        if dur <= 0.2:
            return "视频时长过短（小于 0.2 秒），无法用于 3D 重建。"
    except Exception:
        return "视频文件无法解析（可能格式损坏或编码不支持），换个正常视频试试。"

    return None
