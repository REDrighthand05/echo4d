"""Win32 window control operations (maximize, restore, toggle, minimize, close)."""

import os
import sys
import threading
from typing import Optional, Tuple

from echo_desktop.win32_types import (
    _SW_MAXIMIZE,
    _SW_MINIMIZE,
    _SW_RESTORE,
    _SWP_FRAMECHANGED,
    _SWP_NOMOVE,
    _SWP_NOSIZE,
    _SWP_NOZORDER,
    _SWP_SHOWWINDOW,
    get_window_rect,
    get_work_area_rect,
    resolve_hwnd,
    wh_rects_match,
)
from echo_desktop.win32_wndproc import ensure_maximize_styles

_pre_max_rect: Optional[Tuple[int, int, int, int]] = None


def is_window_maximized(window) -> bool:
    """True when window fills the work area with pre-max rect or Windows reports zoomed."""
    if sys.platform != "win32":
        return False
    try:
        hwnd = resolve_hwnd(window)
        if not hwnd:
            return False
        import ctypes

        work = get_work_area_rect(hwnd)
        now = get_window_rect(hwnd)
        global _pre_max_rect
        if wh_rects_match(now, work) and _pre_max_rect is not None:
            return True
        return bool(ctypes.windll.user32.IsZoomed(hwnd))
    except Exception:
        return False


def maximize_work_area(window) -> bool:
    """Fill monitor's work area while keeping taskbar visible."""
    if sys.platform != "win32":
        if window:
            window.maximize()
        return True
    hwnd = resolve_hwnd(window)
    if not hwnd:
        return False
    try:
        import ctypes

        user32 = ctypes.windll.user32
        now = get_window_rect(hwnd)
        work = get_work_area_rect(hwnd)
        if not now or not work:
            return False
        global _pre_max_rect
        _pre_max_rect = now
        ensure_maximize_styles(hwnd)
        user32.ShowWindow(hwnd, _SW_MAXIMIZE)
        after = get_window_rect(hwnd)
        zoomed = bool(user32.IsZoomed(hwnd))
        if wh_rects_match(after, work) or zoomed:
            return True
        ok = bool(
            user32.SetWindowPos(
                hwnd, 0, work[0], work[1], work[2], work[3],
                _SWP_NOZORDER | _SWP_SHOWWINDOW,
            )
        )
        after2 = get_window_rect(hwnd)
        return bool(ok and wh_rects_match(after2, work))
    except Exception:
        return False


def restore_work_area(window) -> bool:
    """Restore window to size and position prior to work-area maximization."""
    if sys.platform != "win32":
        if window:
            window.restore()
        return True
    hwnd = resolve_hwnd(window)
    if not hwnd:
        return False
    try:
        import ctypes

        user32 = ctypes.windll.user32
        global _pre_max_rect
        rect = _pre_max_rect
        if not rect:
            return False
        user32.ShowWindow(hwnd, _SW_RESTORE)
        after = get_window_rect(hwnd)
        if wh_rects_match(after, rect):
            return True
        ok = bool(
            user32.SetWindowPos(
                hwnd, 0, rect[0], rect[1], rect[2], rect[3],
                _SWP_NOZORDER | _SWP_SHOWWINDOW,
            )
        )
        after2 = get_window_rect(hwnd)
        return bool(ok and wh_rects_match(after2, rect))
    except Exception:
        return False


def toggle_maximize(window) -> bool:
    """Toggle between maximized work-area and restored normal window state."""
    if sys.platform != "win32":
        if window:
            window.maximize()
        return True
    try:
        hwnd = resolve_hwnd(window)
        if not hwnd:
            return False
        if is_window_maximized(window):
            done = restore_work_area(window)
            if not done and window:
                window.restore()
            is_max = False
        else:
            done = maximize_work_area(window)
            if not done and window:
                window.maximize()
            is_max = True
        try:
            import ctypes

            ctypes.windll.user32.SetWindowPos(
                hwnd, 0, 0, 0, 0, 0,
                _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOZORDER | _SWP_FRAMECHANGED,
            )
        except Exception:
            pass
        return is_max
    except Exception:
        return False


def minimize_window(window) -> bool:
    """Minimize window safely to taskbar."""
    if sys.platform == "win32":
        try:
            hwnd = resolve_hwnd(window)
            if hwnd:
                import ctypes

                if ctypes.windll.user32.ShowWindow(hwnd, _SW_MINIMIZE):
                    return True
        except Exception:
            pass
    if window is not None:
        try:
            window.minimize()
            return True
        except Exception:
            pass
    return False


def close_window(window):
    """Gracefully destroy window and exit process without deadlock."""
    try:
        def _destroy():
            try:
                if window is not None:
                    window.destroy()
            except Exception:
                pass

        threading.Thread(target=_destroy, daemon=True).start()
    except Exception:
        pass
    os._exit(0)
