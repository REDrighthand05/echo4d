"""Win32 window procedure handlers, hit-testing dispatch, and resize handling."""

import sys
import time
from typing import Callable, Optional

from echo_desktop.win32_child import (
    clear_child_pending,
    post_resize_browser_child,
    resize_browser_child,
)
from echo_desktop.win32_drop import handle_drop_files
from echo_desktop.win32_types import (
    _HTCLIENT,
    _SC_MAXIMIZE,
    _SC_MINIMIZE,
    _SC_RESTORE,
    _SIZE_MAXIMIZED,
    _SIZE_MINIMIZED,
    _SIZE_RESTORED,
    _SWP_FRAMECHANGED,
    _SWP_NOMOVE,
    _SWP_NOSIZE,
    _SWP_NOZORDER,
    _WM_DROPFILES,
    _WM_GETMINMAXINFO,
    _WM_NCCALCSIZE,
    _WM_NCHITTEST,
    _WM_RESIZE_CHILD_DEFERRED,
    _WM_SIZE,
    _WM_SYSCOMMAND,
    _WS_MAXIMIZEBOX,
    _WS_MINIMIZEBOX,
    _WS_THICKFRAME,
    hit_test_resize,
    signed_short,
    window_frame_rects,
)

_maximize_styles_done = set()
_resize_size_log_at = 0.0
_resize_size_log_kind = None
_resize_orig_proc = None
_log_callback: Optional[Callable[[str], None]] = None


def set_log_callback(cb: Optional[Callable[[str], None]]):
    global _log_callback
    _log_callback = cb


def _applog_write(msg: str):
    if _log_callback:
        try:
            _log_callback(msg)
        except Exception:
            pass


def _fill_maxinfo_from_work_area(hwnd: int, lparam: int):
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        class _MINMAXINFO(ctypes.Structure):
            _fields_ = [
                ("ptReserved", wintypes.POINT),
                ("ptMaxSize", wintypes.POINT),
                ("ptMaxPosition", wintypes.POINT),
                ("ptMinTrackSize", wintypes.POINT),
                ("ptMaxTrackSize", wintypes.POINT),
            ]

        class _MONITORINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT),
                ("dwFlags", wintypes.DWORD),
            ]

        user32 = ctypes.windll.user32
        mmi = ctypes.cast(lparam, ctypes.POINTER(_MINMAXINFO)).contents
        hmon = user32.MonitorFromWindow(hwnd, 2)
        if not hmon:
            return
        mi = _MONITORINFO()
        mi.cbSize = ctypes.sizeof(_MONITORINFO)
        if not user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
            return
        w, m = mi.rcWork, mi.rcMonitor
        if w.right - w.left <= 0 or w.bottom - w.top <= 0:
            return
        mmi.ptMaxPosition.x = w.left - m.left
        mmi.ptMaxPosition.y = w.top - m.top
        mmi.ptMaxSize.x = w.right - w.left
        mmi.ptMaxSize.y = w.bottom - w.top
        mmi.ptMaxTrackSize.x = mmi.ptMaxSize.x
        mmi.ptMaxTrackSize.y = mmi.ptMaxSize.y
        if mmi.ptMinTrackSize.x > mmi.ptMaxTrackSize.x:
            mmi.ptMinTrackSize.x = mmi.ptMaxTrackSize.x
        if mmi.ptMinTrackSize.y > mmi.ptMaxTrackSize.y:
            mmi.ptMinTrackSize.y = mmi.ptMaxTrackSize.y
    except Exception:
        pass


def ensure_maximize_styles(hwnd: int):
    if not hwnd or hwnd in _maximize_styles_done or sys.platform != "win32":
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32
        style = user32.GetWindowLongW(hwnd, -16)
        want = style | _WS_THICKFRAME | _WS_MAXIMIZEBOX | _WS_MINIMIZEBOX
        if want != style:
            user32.SetWindowLongW(hwnd, -16, want)
            user32.SetWindowPos(
                hwnd,
                0,
                0,
                0,
                0,
                0,
                _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOZORDER | _SWP_FRAMECHANGED,
            )
        _maximize_styles_done.add(hwnd)
    except Exception:
        pass


def resize_wndproc_inner(hwnd: int, msg: int, wparam: int, lparam: int) -> int:
    import ctypes

    if msg == _WM_GETMINMAXINFO:
        _fill_maxinfo_from_work_area(hwnd, lparam)
    if msg == _WM_NCCALCSIZE:
        return 0
    if msg == _WM_NCHITTEST:
        try:
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            x = signed_short(lparam & 0xFFFF)
            y = signed_short((lparam >> 16) & 0xFFFF)
            rect = wintypes.RECT()
            if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                frames = window_frame_rects(hwnd) or ()
                code = hit_test_resize(
                    x,
                    y,
                    (rect.left, rect.top, rect.right, rect.bottom),
                    frames=frames,
                )
                if code != _HTCLIENT:
                    return code
        except Exception:
            pass

    if msg == _WM_SYSCOMMAND:
        cmd = wparam & 0xFFF0
        name = {
            _SC_MINIMIZE: "最小化",
            _SC_RESTORE: "还原",
            _SC_MAXIMIZE: "最大化",
        }.get(cmd, f"系统命令 {cmd:#x}")
        _applog_write(f"窗口系统命令：{name}")

    if msg == _WM_SIZE:
        kind = {
            _SIZE_RESTORED: "还原",
            _SIZE_MINIMIZED: "最小化",
            _SIZE_MAXIMIZED: "最大化",
        }.get(wparam, f"尺寸变化 {wparam}")
        w, h = lparam & 0xFFFF, (lparam >> 16) & 0xFFFF
        global _resize_size_log_at, _resize_size_log_kind
        now = time.monotonic()
        if kind != _resize_size_log_kind or now - _resize_size_log_at >= 1.0:
            _applog_write(f"窗口尺寸变化：{kind} {w}x{h}")
            _resize_size_log_at = now
            _resize_size_log_kind = kind
        if wparam in (_SIZE_RESTORED, _SIZE_MAXIMIZED):
            post_resize_browser_child(hwnd)

    if msg == _WM_DROPFILES:
        handle_drop_files(wparam)
        return 0

    if msg == _WM_RESIZE_CHILD_DEFERRED:
        clear_child_pending()
        resize_browser_child(hwnd)
        return 0

    orig = _resize_orig_proc
    if orig is None:
        return _HTCLIENT
    return ctypes.windll.user32.CallWindowProcW(orig, hwnd, msg, wparam, lparam)


def resize_wndproc(hwnd: int, msg: int, wparam: int, lparam: int) -> int:
    try:
        return resize_wndproc_inner(hwnd, msg, wparam, lparam)
    except Exception as exc:
        _applog_write(f"窗口过程异常：{exc}")
        try:
            orig = _resize_orig_proc
            if orig is not None:
                import ctypes
                return ctypes.windll.user32.CallWindowProcW(
                    orig, hwnd, msg, wparam, lparam
                )
        except Exception:
            pass
        return _HTCLIENT
