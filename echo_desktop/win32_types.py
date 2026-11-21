"""Win32 API constants, geometry helpers, and structures for frameless desktop styling."""

import sys
from typing import Optional, Sequence, Tuple

_RESIZE_BORDER = 8

_HTCLIENT = 1
_HTLEFT = 10
_HTRIGHT = 11
_HTTOP = 12
_HTTOPLEFT = 13
_HTTOPRIGHT = 14
_HTBOTTOM = 15
_HTBOTTOMLEFT = 16
_HTBOTTOMRIGHT = 17

_WM_NCHITTEST = 0x0084
_GWLP_WNDPROC = -4
_WM_NCCALCSIZE = 0x0083
_WM_GETMINMAXINFO = 0x0024
_GWL_STYLE = -16
_WS_THICKFRAME = 0x00040000
_WS_MAXIMIZEBOX = 0x00010000
_WS_MINIMIZEBOX = 0x00020000
_SW_MAXIMIZE = 3
_SW_RESTORE = 9
_SW_MINIMIZE = 6
_SWP_NOMOVE = 0x0002
_SWP_NOSIZE = 0x0001
_SWP_NOZORDER = 0x0004
_SWP_SHOWWINDOW = 0x0040
_SWP_FRAMECHANGED = 0x0020
_WM_SYSCOMMAND = 0x0112
_WM_SIZE = 0x0005
_WM_DROPFILES = 0x0233
_WM_RESIZE_CHILD_DEFERRED = 0x8001
_SC_MINIMIZE = 0xF020
_SC_RESTORE = 0xF120
_SC_MAXIMIZE = 0xF030
_SIZE_RESTORED = 0
_SIZE_MINIMIZED = 1
_SIZE_MAXIMIZED = 2

_resize_argtypes_set = False


def hit_test_resize(
    x: int,
    y: int,
    rect: Tuple[int, int, int, int],
    border: int = _RESIZE_BORDER,
    frames: Sequence[Tuple[int, int, int, int]] = (),
) -> int:
    """Pure geometry: calculate hit code for cursor against window rect."""
    left, top, right, bottom = rect
    for fl, ft, fr, fb in frames:
        if (
            abs(left - fl) <= 2
            and abs(top - ft) <= 2
            and abs(right - fr) <= 2
            and abs(bottom - fb) <= 2
        ):
            return _HTCLIENT

    at_left = x < left + border
    at_right = x >= right - border
    at_top = y < top + border
    at_bottom = y >= bottom - border

    if at_top and at_left:
        return _HTTOPLEFT
    if at_top and at_right:
        return _HTTOPRIGHT
    if at_bottom and at_left:
        return _HTBOTTOMLEFT
    if at_bottom and at_right:
        return _HTBOTTOMRIGHT
    if at_left:
        return _HTLEFT
    if at_right:
        return _HTRIGHT
    if at_top:
        return _HTTOP
    if at_bottom:
        return _HTBOTTOM
    return _HTCLIENT


def signed_short(v: int) -> int:
    """Return low/high word of lParam as a signed 16-bit integer."""
    import ctypes
    return ctypes.c_short(v & 0xFFFF).value


def window_frame_rects(hwnd: int) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
    """Return (work_area, full_monitor) rects for the window's monitor."""
    if sys.platform != "win32" or not hwnd:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32

        class MONITORINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT),
                ("dwFlags", wintypes.DWORD),
            ]

        hmon = user32.MonitorFromWindow(hwnd, 2)
        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)
        if hmon and user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
            w, m = mi.rcWork, mi.rcMonitor
            return (
                (w.left, w.top, w.right, w.bottom),
                (m.left, m.top, m.right, m.bottom),
            )
    except Exception:
        pass
    return None


def ensure_win32_argtypes(user32, wndproc_type, set_long, get_long=None):
    """Pin the ctypes signatures for user32 calls to prevent 64-bit address truncation."""
    global _resize_argtypes_set
    if _resize_argtypes_set or sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        user32.CallWindowProcW.argtypes = [
            wndproc_type,
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        user32.CallWindowProcW.restype = ctypes.c_ssize_t
        set_long.argtypes = [wintypes.HWND, ctypes.c_int, wndproc_type]
        set_long.restype = ctypes.c_void_p
        if get_long is not None:
            get_long.argtypes = [wintypes.HWND, ctypes.c_int]
            get_long.restype = ctypes.c_void_p
        user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        user32.GetWindowRect.restype = wintypes.BOOL
        _resize_argtypes_set = True
    except Exception:
        pass


def get_window_rect(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
    """Return window rect as (left, top, width, height) or None."""
    if sys.platform != "win32" or not hwnd:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        rect = wintypes.RECT()
        if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        return (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)
    except Exception:
        return None


def get_work_area_rect(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
    """Return work area rect of monitor containing hwnd as (left, top, width, height) or None."""
    if sys.platform != "win32" or not hwnd:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        hmon = user32.MonitorFromWindow(hwnd, 2)

        class MONITORINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT),
                ("dwFlags", wintypes.DWORD),
            ]

        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)
        if hmon and user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
            w = mi.rcWork
            return (w.left, w.top, w.right - w.left, w.bottom - w.top)
        work = wintypes.RECT()
        if user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(work), 0):
            return (work.left, work.top, work.right - work.left, work.bottom - work.top)
        return None
    except Exception:
        return None


def wh_rects_match(
    a: Optional[Tuple[int, int, int, int]],
    b: Optional[Tuple[int, int, int, int]],
    tol: int = 2,
) -> bool:
    """True when two (left, top, width, height) rects agree within tol pixels."""
    if a is None or b is None:
        return False
    return (
        abs(a[0] - b[0]) <= tol
        and abs(a[1] - b[1]) <= tol
        and abs(a[2] - b[2]) <= tol
        and abs(a[3] - b[3]) <= tol
    )


def resolve_hwnd(window) -> Optional[int]:
    """Extract Win32 HWND from PyWebview window or fallback search."""
    if sys.platform != "win32":
        return None
    hwnd = None
    native = getattr(window, "native", None)
    if native is not None:
        try:
            hwnd = int(native.Handle)
        except Exception:
            hwnd = None
    if not hwnd:
        try:
            import ctypes

            hwnd = ctypes.windll.user32.FindWindowW(None, "ECHO 回声") or None
        except Exception:
            hwnd = None
    return hwnd


