"""Echo Render — ECHO 配套高性能渲染算子 (high-performance CPU
point/line renderer; engine codename MUS).

Public API (the whole surface):

    from echo_render import render_frame, render_sequence

``render_frame`` draws one frame from plain numpy arrays + a camera
dict; ``render_sequence`` renders a JSON-described batch to PNGs.
The package is generic: it knows no application-specific file
formats and no hardcoded paths.
"""
from .core import render_frame, render_sequence

__version__ = "0.1.0"
__all__ = ["render_frame", "render_sequence"]
