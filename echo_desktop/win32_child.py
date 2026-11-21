"""Win32 browser-child window resizing and message loop deferral."""

import sys
from echo_desktop.win32_types import (
    _SWP_NOZORDER,
    _SWP_SHOWWINDOW,
    _WM_RESIZE_CHILD_DEFERRED,
)

_resize_child_pending = False


def is_child_pending() -> bool:
    return _resize_child_pending


def clear_child_pending():
    global _resize_child_pending
    _resize_child_pending = False


def post_resize_browser_child(hwnd: int) -> bool:
    """Queue deferred child resize onto message loop via private WM_APP message."""
    global _resize_child_pending
    if not hwnd or sys.platform != "win32":
        return False
    if _resize_child_pending:
        return True
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        user32.PostMessageW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        user32.PostMessageW.restype = wintypes.BOOL
        if user32.PostMessageW(hwnd, _WM_RESIZE_CHILD_DEFERRED, 0, 0):
            _resize_child_pending = True
            return True
        return False
    except Exception:
        return False


def resize_browser_child(hwnd: int, win32_facade=None) -> bool:
    """Resize the WebView2 child window to match parent client area dimensions."""
    if sys.platform != "win32" or not hwnd:
        return False
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        client = wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(client)):
            return False
        cw = client.right - client.left
        ch = client.bottom - client.top
        if cw <= 0 or ch <= 0:
            return False

        if win32_facade is None:
            from echo_desktop.win32_hook import real_win32_facade
            win32_facade = real_win32_facade()
            if win32_facade is None:
                return False

        kids = win32_facade.enum_children(hwnd)
        best = None
        best_area = -1
        for k in kids:
            if not k.get("visible") or not k.get("rect"):
                continue
            l, t, r, b = k["rect"]
            area = max(0, r - l) * max(0, b - t)
            if area > best_area:
                best_area = area
                best = k
        if best is None:
            return False
        bl, bt, br, bb = best["rect"]
        bw, bh = br - bl, bb - bt
        if bw == cw and bh == ch:
            return True
        ok = bool(
            user32.SetWindowPos(
                best["hwnd"], 0, 0, 0, cw, ch, _SWP_NOZORDER | _SWP_SHOWWINDOW
            )
        )
        return ok
    except Exception:
        return False
