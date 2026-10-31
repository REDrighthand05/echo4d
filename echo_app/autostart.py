"""开机自启 (launch at sign-in), per-user only -- matching the
per-user, no-admin install: no service, no HKLM, nothing an uninstall
would leave behind beyond the app's own entry.

Windows writes the classic HKCU Run key
(Software\\Microsoft\\Windows\\CurrentVersion\\Run, value "ECHO
Scene"); other platforms write an XDG autostart .desktop file so the
same settings toggle means something in dev runs too. The registered
command launches the app with --tray: at sign-in the window stays
hidden and only the tray icon appears, WeChat-style.

Every function swallows its own failures and reports False -- a
broken autostart entry is a nuisance, never a crash.
"""
import sys
from pathlib import Path

VALUE_NAME = "ECHO Scene"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
DESKTOP_NAME = "echo-scene.desktop"


def launch_command(exe_path=None) -> str:
    """The command line autostart registers: the exe (or, in a source
    run, python + app.py) with the --tray flag."""
    if exe_path:
        return f'"{exe_path}" --tray'
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}" --tray'
    app_py = Path(__file__).resolve().parent.parent / "app.py"
    return f'"{sys.executable}" "{app_py}" --tray'


def _default_exe():
    return sys.executable if getattr(sys, "frozen", False) else None


def _desktop_file(home=None) -> Path:
    home = Path(home) if home else Path.home()
    return home / ".config" / "autostart" / DESKTOP_NAME


def _set_windows(enabled: bool, command: str) -> bool:
    import winreg  # Windows-only stdlib; imported lazily on purpose

    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0,
                         winreg.KEY_SET_VALUE)
    try:
        if enabled:
            winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_SZ, command)
        else:
            try:
                winreg.DeleteValue(key, VALUE_NAME)
            except FileNotFoundError:
                pass  # already absent: disabling is idempotent
        return True
    finally:
        winreg.CloseKey(key)


def _read_windows():
    import winreg

    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0,
                         winreg.KEY_READ)
    try:
        value, _ = winreg.QueryValueEx(key, VALUE_NAME)
        return value
    except FileNotFoundError:
        return None
    finally:
        winreg.CloseKey(key)


def _set_desktop(enabled: bool, command: str, home=None) -> bool:
    target = _desktop_file(home)
    if enabled:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "[Desktop Entry]\n"
            "Type=Application\n"
            f"Name={VALUE_NAME}\n"
            f"Exec={command}\n"
            "X-GNOME-Autostart-enabled=true\n",
            encoding="utf-8")
    else:
        target.unlink(missing_ok=True)
    return True


def set_autostart(enabled: bool, exe_path=None, platform=None,
                  home=None) -> bool:
    """Register or remove the sign-in launch entry. Never raises."""
    platform = sys.platform if platform is None else platform
    command = launch_command(exe_path or _default_exe())
    try:
        if platform.startswith("win"):
            return _set_windows(enabled, command)
        return _set_desktop(enabled, command, home)
    except Exception:
        return False


def autostart_command(platform=None, home=None):
    """The currently registered autostart command, or None when
    autostart is off (or unreadable). Never raises."""
    platform = sys.platform if platform is None else platform
    try:
        if platform.startswith("win"):
            return _read_windows()
        target = _desktop_file(home)
        if not target.exists():
            return None
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.startswith("Exec="):
                return line[len("Exec="):]
        return None
    except Exception:
        return None


def is_autostart_enabled(platform=None, home=None) -> bool:
    return autostart_command(platform, home) is not None

