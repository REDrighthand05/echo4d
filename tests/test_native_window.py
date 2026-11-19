"""Exercise the desktop caption handlers on a real Windows WebView2 window."""
import ctypes
from ctypes import wintypes
from pathlib import Path
import sys
import threading
import time
import uuid

import webview

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web_app import Bridge


def wait_until(predicate, operation):
    deadline = time.monotonic() + 10
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out during {operation}")
        time.sleep(0.05)


def main():
    user32 = ctypes.windll.user32
    user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    user32.FindWindowW.restype = wintypes.HWND
    for check in (user32.IsZoomed, user32.IsIconic):
        check.argtypes = [wintypes.HWND]
        check.restype = wintypes.BOOL
    title = "Echo window regression " + uuid.uuid4().hex
    url = (Path(__file__).resolve().parents[1] / "web_ui/index.html").as_uri()
    window = webview.create_window(title, url, width=800, height=600, frameless=True)
    ready = threading.Event()
    window.events.loaded += ready.set
    bridge = Bridge()
    bridge._window = window
    failures = []

    def exercise():
        try:
            assert ready.wait(30), "page did not load"
            hwnd = user32.FindWindowW(None, title)
            assert hwnd, "native window not found"
            window.evaluate_js("window.testDocument = document; true")
            for _ in range(10):
                bridge.win_toggle_max()
                wait_until(lambda: user32.IsZoomed(hwnd), "maximize")
                bridge.win_toggle_max()
                wait_until(lambda: not user32.IsZoomed(hwnd), "restore size")
            bridge.win_minimize()
            wait_until(lambda: user32.IsIconic(hwnd), "minimize")
            window.restore()
            wait_until(lambda: not user32.IsIconic(hwnd), "restore from minimize")
            assert window.evaluate_js("window.testDocument === document"), "page reloaded"
        except Exception as exc:
            failures.append(exc)
        finally:
            window.destroy()

    webview.start(exercise, gui="edgechromium")
    if failures:
        raise failures[0]


if __name__ == "__main__":
    main()
