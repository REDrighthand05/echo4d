"""Native Win32 drag-and-drop file ingestion for frameless desktop."""

import sys
from typing import Callable, List, Optional

_drop_callback: Optional[Callable[[str], None]] = None
_log_callback: Optional[Callable[[str], None]] = None


def set_drop_callback(cb: Optional[Callable[[str], None]]):
    global _drop_callback
    _drop_callback = cb


def set_log_callback(cb: Optional[Callable[[str], None]]):
    global _log_callback
    _log_callback = cb


def _log(msg: str):
    if _log_callback:
        try:
            _log_callback(msg)
        except Exception:
            pass


def drop_files_to_paths(hdrop, _shell32=None) -> List[str]:
    """Extract list of dropped file paths from an HDROP handle."""
    try:
        import ctypes

        shell32 = _shell32 or ctypes.windll.shell32
        count = shell32.DragQueryFileW(hdrop, 0xFFFFFFFF, None, 0)
        paths = []
        for index in range(count):
            buf = ctypes.create_unicode_buffer(32768)
            got = shell32.DragQueryFileW(hdrop, index, buf, 32768)
            if got:
                paths.append(buf.value)
        return paths
    except Exception:
        return []


def handle_drop_files(hdrop, _shell32=None) -> bool:
    """Handle WM_DROPFILES message and dispatch first file to drop callback."""
    import ctypes

    try:
        paths = drop_files_to_paths(hdrop, _shell32=_shell32)
        if paths and _drop_callback:
            _drop_callback(paths[0])
            _log(f"拖放文件：{paths[0]}")
            return True
        return False
    except Exception:
        return False
    finally:
        try:
            shell32 = _shell32 or ctypes.windll.shell32
            shell32.DragFinish(hdrop)
        except Exception:
            pass


def install_drop_accept(window):
    """Enable drag-and-drop on native window handle."""
    if sys.platform != "win32":
        return
    native = getattr(window, "native", None)
    hwnd = getattr(native, "Handle", None) if native else None
    if not hwnd:
        return
    try:
        import ctypes
        ctypes.windll.shell32.DragAcceptFiles(int(hwnd), True)
    except Exception:
        pass
