"""PyWebview window lifecycle manager and application bootstrap."""

import sys
import threading
from pathlib import Path
from typing import Optional

from echo_desktop.bridge import Bridge
from echo_desktop.win32_hook import polish_and_hook


def _setup_dpi_awareness():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except Exception:
            pass


def _check_gpu_accel(bridge: Bridge):
    try:
        import onnxruntime as ort
        provs = ort.get_available_providers()
        if "DmlExecutionProvider" in provs or "CUDAExecutionProvider" in provs:
            bridge._log("显卡加速可用，深度计算会走显卡。")
        else:
            bridge._log("深度计算使用 CPU（未检测到显卡运行时）。")
    except Exception:
        pass


def _resolve_index_html() -> Path:
    base = Path(__file__).resolve().parent.parent / "web_ui" / "index.html"
    if base.exists():
        return base
    # Fallback to local working directory
    cur = Path("web_ui/index.html").resolve()
    return cur


def launch_desktop(custom_url: Optional[str] = None):
    """Bootstrap and start the ECHO desktop graphical application."""
    import webview

    _setup_dpi_awareness()
    bridge = Bridge()
    _check_gpu_accel(bridge)

    url = custom_url or str(_resolve_index_html())

    window = webview.create_window(
        title="ECHO 回声",
        url=url,
        js_api=bridge,
        width=1040,
        height=836,
        min_size=(880, 620),
        frameless=True,
        easy_drag=False,
        background_color="#dbe2eb",
    )
    bridge._window = window

    polish_and_hook(window)

    try:
        window.events.shown += lambda: polish_and_hook(window)
    except Exception:
        pass

    for ev_name in ("restored", "maximized"):
        try:
            ev = getattr(window.events, ev_name)
            ev += lambda: polish_and_hook(window)
        except Exception:
            pass

    for delay in (1.5, 4.0, 10.0, 30.0):
        timer = threading.Timer(delay, polish_and_hook, args=(window,))
        timer.daemon = True
        timer.start()

    webview.start()


if __name__ == "__main__":
    launch_desktop()
