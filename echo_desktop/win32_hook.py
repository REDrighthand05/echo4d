"""Win32 window procedure subclassing, DWM polish, and drag/drop registration."""

import sys
from typing import Any, Dict, Optional

from echo_desktop.win32_types import (
    ensure_win32_argtypes,
    resolve_hwnd,
)
from echo_desktop.win32_window import (
    close_window,
    is_window_maximized,
    maximize_work_area,
    minimize_window,
    restore_work_area,
    toggle_maximize,
)
from echo_desktop.win32_wndproc import (
    ensure_maximize_styles,
    handle_drop_files,
    resize_wndproc,
)

_resize_orig_proc = None
_resize_hook_hwnd = None
_resize_callback = None
_resize_status_info = {"installed": False, "hwnd": None, "msg": "未初始化"}


class Win32Facade:
    """Facade wrapping user32 subclassing and window queries."""

    def __init__(self, user32, wndproc_type, set_long, get_long):
        self._user32 = user32
        self._wndproc_type = wndproc_type
        self._set_long = set_long
        self._get_long = get_long

    def make_callback(self, fn):
        return self._wndproc_type(fn)

    def wrap_proc(self, addr):
        if not addr:
            return None
        return self._wndproc_type(addr)

    def set_proc(self, hwnd: int, callback):
        return self._set_long(hwnd, -4, callback)

    def get_proc(self, hwnd: int):
        val = self._get_long(hwnd, -4)
        return int(val) if val else None

    def enum_children(self, hwnd: int):
        import ctypes
        from ctypes import wintypes

        user32 = self._user32
        out = []
        queue = [(hwnd, 0)]
        seen = set()
        while queue:
            parent, depth = queue.pop(0)
            if depth >= 4 or parent in seen:
                continue
            seen.add(parent)
            found = []

            def _collect(child, _lp, _found=found):
                try:
                    buf = ctypes.create_unicode_buffer(256)
                    user32.GetClassNameW(child, buf, 256)
                    cls = buf.value
                    rect = wintypes.RECT()
                    ok = bool(user32.GetWindowRect(child, ctypes.byref(rect)))
                    vis = bool(user32.IsWindowVisible(child))
                    _found.append({
                        "hwnd": int(child),
                        "class": cls,
                        "rect": (rect.left, rect.top, rect.right, rect.bottom) if ok else None,
                        "visible": vis,
                    })
                except Exception:
                    pass
                return True

            try:
                cb_type = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
                enum_cb = cb_type(_collect)
                user32.EnumChildWindows(parent, ctypes.cast(enum_cb, ctypes.c_void_p), 0)
            except Exception:
                continue
            for ch in found:
                out.append(ch)
                queue.append((ch["hwnd"], depth + 1))
        return out


def real_win32_facade() -> Optional[Win32Facade]:
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        wndproc_type = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t,
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        )
        set_long = getattr(user32, "SetWindowLongPtrW", None) or getattr(user32, "SetWindowLongW", None)
        get_long = getattr(user32, "GetWindowLongPtrW", None) or getattr(user32, "GetWindowLongW", None)
        if set_long is None or get_long is None:
            return None
        ensure_win32_argtypes(user32, wndproc_type, set_long, get_long)
        return Win32Facade(user32, wndproc_type, set_long, get_long)
    except Exception:
        return None


def callback_addr(cb) -> Optional[int]:
    if cb is None:
        return None
    try:
        return int(getattr(cb, "address"))
    except Exception:
        pass
    try:
        import ctypes

        val = ctypes.cast(cb, ctypes.c_void_p).value
        return int(val) if val else None
    except Exception:
        return None


def resize_proc_is_ours(current, ours) -> bool:
    if current is None or ours is None:
        return False
    try:
        return int(current) == int(ours)
    except Exception:
        return False


def resize_needs_reinstall(current, ours, installed_hwnd, hwnd) -> bool:
    if hwnd is None:
        return False
    if installed_hwnd != hwnd:
        return True
    return not resize_proc_is_ours(current, ours)


def install_resize_hook(window, win32_facade=None) -> bool:
    global _resize_orig_proc, _resize_hook_hwnd, _resize_callback, _resize_status_info
    facade = win32_facade or real_win32_facade()
    if facade is None:
        return False
    try:
        hwnd = resolve_hwnd(window)
        if not hwnd:
            _resize_status_info = {"installed": False, "hwnd": None, "msg": "找不到窗口句柄"}
            return False

        try:
            current = facade.get_proc(hwnd)
        except Exception:
            current = None

        if not resize_needs_reinstall(
            current, callback_addr(_resize_callback), _resize_hook_hwnd, hwnd
        ):
            _resize_hook_hwnd = hwnd
            _resize_status_info = {"installed": True, "hwnd": hwnd, "msg": "已装上（读回确认）"}
            return True

        cb = facade.make_callback(resize_wndproc)
        prev = facade.set_proc(hwnd, cb)
        if not prev:
            _resize_status_info = {"installed": False, "hwnd": hwnd, "msg": "换过程失败"}
            return False

        try:
            current = facade.get_proc(hwnd)
        except Exception:
            current = None

        if not resize_proc_is_ours(current, callback_addr(cb)):
            _resize_callback = cb
            _resize_status_info = {"installed": False, "hwnd": hwnd, "msg": "装完读回的过程不是我们的"}
            return False

        _resize_callback = cb
        _resize_orig_proc = facade.wrap_proc(prev)
        _resize_hook_hwnd = hwnd

        import echo_desktop.win32_wndproc as wp

        wp._resize_orig_proc = _resize_orig_proc

        _resize_status_info = {"installed": True, "hwnd": hwnd, "msg": "已装上"}
        return True
    except Exception as exc:
        _resize_status_info = {"installed": False, "hwnd": None, "msg": f"异常: {exc}"}
        return False


def polish_window(window):
    """Win11 DWM cosmetic attributes (round corners, soft caption tint)."""
    if sys.platform != "win32":
        return
    hwnd = resolve_hwnd(window)
    if not hwnd:
        return
    try:
        import ctypes

        dwm = ctypes.windll.dwmapi

        def _set(attr, val):
            v = ctypes.c_int(val)
            dwm.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(v), ctypes.sizeof(v))

        _set(20, 0)         # light glass
        _set(33, 2)         # round corners
        _set(35, 0xF7F2EE)  # caption colour #eef2f7
        _set(36, 0x211E1C)  # caption text #1c1e21
    except Exception:
        pass


def install_drop_accept(window):
    if sys.platform != "win32":
        return
    hwnd = resolve_hwnd(window)
    if not hwnd:
        return
    try:
        import ctypes

        ctypes.windll.shell32.DragAcceptFiles(hwnd, True)
    except Exception:
        pass


def polish_and_hook(window):
    """Apply DWM polish, resize wndproc hook, and drop accept in one unified call."""
    polish_window(window)
    install_resize_hook(window)
    install_drop_accept(window)


def resize_hook_status() -> Dict[str, Any]:
    return dict(_resize_status_info)


_resolve_hwnd = resolve_hwnd
