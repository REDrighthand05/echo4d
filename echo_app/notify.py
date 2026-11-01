"""Completion toast: one Windows notification when a job finishes or
fails while the window is hidden in the tray.

Backend: winotify -- pure Python (it composes the toast XML and hands
it to a hidden PowerShell), so it freezes cleanly with PyInstaller
and adds no native dependency. The app_id is "ECHO Scene", the exact
name of the Start-menu shortcut the installer creates; that is what
lets Windows attribute the toast to the app.

Containment contract: notify() NEVER raises and returns False
wherever a toast cannot be shown (non-Windows, winotify missing,
PowerShell blocked, ...). The caller fires it from a background
thread and ignores the result -- a broken toast must not touch the
app, let alone a running job. Note winotify imports winreg at module
level, so the import itself is Windows-only and stays behind the
platform check.
"""
import sys

APP_ID = "ECHO Scene"


def notify(title: str, message: str, app_id: str = APP_ID) -> bool:
    """Show one toast. True when it was handed to Windows."""
    if not sys.platform.startswith("win"):
        return False
    try:
        from winotify import Notification

        toast = Notification(app_id=app_id, title=title, msg=message,
                             duration="short")
        toast.show()
        return True
    except Exception:
        return False
