"""Shared resource resolution: ONE place that knows where things live.

Modules used to each compute paths from their own __file__, which
silently disagreed across layouts: in the source tree the models sit
at <root>/models, in the installed app they sit at <install>/models
while the payload modules live one level deeper at <install>/app --
so the depth-video module looked in <install>/app/models, found
nothing, and died AFTER its (long) frame extraction had succeeded.
Every consumer now resolves through here, and a missing model raises
an error that lists the places searched instead of one wrong guess.
"""
from __future__ import annotations

import sys
from pathlib import Path

MODEL_SMALL = "depth_anything_v2_small.onnx"
MODEL_BASE = "depth_anything_v2_base.onnx"


def candidate_model_dirs() -> list:
    """Every directory that may hold the depth models, best first."""
    dirs: list = []

    def add(p) -> None:
        p = Path(p)
        if p not in dirs:
            dirs.append(p)

    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            add(Path(meipass) / "models")
        add(Path(sys.executable).resolve().parent / "models")
    here = Path(__file__).resolve().parent          # .../echo_scene
    add(here.parent / "models")                     # source tree / payload
    add(here.parent.parent / "models")              # install root
    add(Path.cwd() / "models")
    return dirs


def find_depth_model(prefer_base: bool = False,
                     dirs: list | None = None) -> Path | None:
    """First existing depth model, honouring the caller's preference
    (the pipeline prefers Base for surface quality; the depth-video
    feature prefers Small for speed -- both fall back to the other)."""
    names = ([MODEL_BASE, MODEL_SMALL] if prefer_base
             else [MODEL_SMALL, MODEL_BASE])
    for name in names:
        for d in (dirs if dirs is not None else candidate_model_dirs()):
            p = Path(d) / name
            if p.exists():
                return p
    return None


def require_depth_model(prefer_base: bool = False) -> Path:
    model = find_depth_model(prefer_base)
    if model is None:
        searched = "; ".join(str(d) for d in candidate_model_dirs())
        raise FileNotFoundError(
            f"找不到深度模型（depth model）。找过这些地方：{searched}")
    return model
