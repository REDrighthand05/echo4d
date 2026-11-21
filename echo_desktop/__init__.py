"""echo_desktop: Native desktop window management, Win32 subclassing, and JS bridge."""

from echo_desktop.bridge import Bridge
from echo_desktop.launcher import launch_desktop
from echo_desktop.win32_hook import (
    install_resize_hook,
    maximize_work_area,
    polish_and_hook,
    polish_window,
    resize_hook_status,
)

__all__ = [
    "Bridge",
    "launch_desktop",
    "polish_and_hook",
    "polish_window",
    "install_resize_hook",
    "maximize_work_area",
    "resize_hook_status",
]
