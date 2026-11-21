#!/usr/bin/env python3
"""ECHO: 3D Scene Studio Root Entrypoint & Facade.

Maintains 100% backward compatibility for all pipeline functions,
constants, and legacy invocations while delegating to the modular
echo_engine, echo_service, echo_cli, and echo_desktop packages.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

# Re-export engine core
import echo_engine as engine
from echo_engine import (
    job_display_name,
    job_file_flags,
    job_finished,
    job_mtime,
    list_previous_jobs,
    play_file,
    replay_available,
    replay_file,
    run_depth_video,
    run_pipeline,
    scale_status_line,
    scene_file,
)

# Legacy aliases
_job_mtime = job_mtime
_job_display_name = job_display_name
_replay_available = replay_available

JOBS_ROOT = Path.home() / "ECHO Jobs"
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}

PLAIN_STATUS = {
    "正在抽帧": "正在看画面",
    "正在挑清晰的帧": "正在挑清楚的画面",
    "正在算相机轨迹": "正在找回拍摄的路线",
    "正在算每帧的远近": "正在量每处的远近",
    "正在拼场景": "正在拼场景",
    "正在清噪": "正在把场景收拾干净",
    "正在定真尺度": "正在找参照物，量出真实大小",
    "正在生成网格": "正在生成表面",
    "正在生成像素模型": "正在把场景变成一个个小方块",
    "正在收尾": "马上就好",
    "正在生成过程回放": "正在做一段过程回放",
    "正在生成路线回放": "正在做一段路线回放",
}
STAGE_COUNT = len(PLAIN_STATUS)


def _resource_path(relative: str) -> Path:
    """Locate a bundled resource."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base) / relative
    return Path(__file__).resolve().parent / relative


# Visual constants
C_BG = "#E9EDF2"
C_SURFACE = "#FFFFFF"
C_BORDER = "#DADDE1"
C_INK = "#1C1E21"
C_SUBINK = "#606770"
C_ACCENT = "#0064D1"
C_ACCENT_HOVER = "#0053AD"
C_BTN2 = "#E9EDF3"
C_BTN2_HOVER = "#DCE3EC"
C_TRACK = "#E4E6EB"
C_LOG_BG = "#F4F6FA"
C_DRAG = "#E7F3FF"
C_PREVIEW_BG = "#05070E"
CORNER = 24
CORNER_SM = 16

SKINS = {
    "now": dict(corner=24, pane_milk=100, pane_blur=19, p_fg=C_ACCENT, p_hover=C_ACCENT_HOVER,
                p_text="#FFFFFF", s_fg=C_BTN2, s_hover=C_BTN2_HOVER, bar_h=10, track=C_TRACK),
    "a": dict(corner=28, pane_milk=58, pane_blur=14, p_fg="#AAC7FA", p_hover="#97B8F7",
              p_text="#1E4FD6", s_fg="#EFF4FB", s_hover="#E1EBF9", bar_h=6, track="#DFE8F3"),
    "b": dict(corner=20, pane_milk=118, pane_blur=16, p_fg=C_ACCENT, p_hover=C_ACCENT_HOVER,
              p_text="#FFFFFF", s_fg="#FAFCFF", s_hover="#EBF1FA", bar_h=8, track="#E3E9F2"),
    "c": dict(corner=32, pane_milk=26, pane_blur=10, p_fg="transparent", p_hover="#E1EBF9",
              p_text=C_ACCENT, s_fg="transparent", s_hover="#E7EFFA", bar_h=3, track="#E6EDF6"),
}
SKIN = SKINS.get(os.environ.get("ECHO_SKIN", "a"), SKINS["now"])
FONT_PREFS = ("Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI",
              "Noto Sans CJK SC", "Noto Sans Sans CJK SC",
              "WenQuanYi Zen Hei", "DejaVu Sans")


def main():
    """Main dispatch: CLI if arguments are supplied, otherwise launch desktop shell."""
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--gui"):
        from echo_cli.main import main as cli_main
        cli_main()
    else:
        try:
            from web_app import main as web_main
            web_main()
        except Exception:
            from echo_desktop.tk_gui import main as tk_main
            tk_main()


if __name__ == "__main__":
    main()
