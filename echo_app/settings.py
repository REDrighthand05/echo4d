"""Persisted user settings.

Stored as JSON under the user's own app-data folder (%APPDATA%\\ECHO
on Windows, ~/.config/ECHO elsewhere) -- never in the install dir:
the per-user install must stay read-only clean, and an uninstall or
upgrade must never carry settings away with the program files.

What is stored:
  jobs_root     -- where new jobs go (None = the default
                   ~/ECHO Jobs). Existing jobs are never moved and
                   never deleted when this changes.
  autostart     -- launch at sign-in (开机自启), hidden to the tray.
  close_to_tray -- closing the window minimises to the system tray
                   (WeChat behaviour) instead of quitting.

Load is forgiving by construction: a missing or corrupt file yields
defaults, unknown keys are ignored, wrong-typed values fall back.
Save is atomic (temp file + replace) so a crash mid-write can never
leave a torn settings file behind.
"""
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

APP_DIR_NAME = "ECHO"
SETTINGS_NAME = "settings.json"


def default_jobs_root() -> Path:
    """Where jobs live unless the user points elsewhere. Same value
    app.py has always used, so upgrading changes nothing."""
    return Path.home() / "ECHO Jobs"


def settings_dir(platform=None, environ=None) -> Path:
    """The per-user app-data folder for settings."""
    platform = sys.platform if platform is None else platform
    environ = os.environ if environ is None else environ
    if platform.startswith("win"):
        appdata = environ.get("APPDATA")
        if appdata:
            return Path(appdata) / APP_DIR_NAME
        return Path.home() / "AppData" / "Roaming" / APP_DIR_NAME
    return Path.home() / ".config" / APP_DIR_NAME


def settings_path(platform=None, environ=None) -> Path:
    return settings_dir(platform, environ) / SETTINGS_NAME


@dataclass
class Settings:
    jobs_root: str = ""          # "" = default (default_jobs_root())
    autostart: bool = False
    close_to_tray: bool = True   # WeChat behaviour out of the box

    def jobs_root_path(self) -> Path:
        return Path(self.jobs_root) if self.jobs_root else default_jobs_root()


def _coerce(data) -> Settings:
    """Build Settings from untrusted JSON data, field by field."""
    out = Settings()
    if not isinstance(data, dict):
        return out
    if isinstance(data.get("jobs_root"), str):
        out.jobs_root = data["jobs_root"]
    for key in ("autostart", "close_to_tray"):
        if isinstance(data.get(key), bool):
            setattr(out, key, data[key])
    return out


def load_settings(path=None) -> Settings:
    """Read settings; defaults on any problem. Never raises."""
    path = Path(path) if path else settings_path()
    try:
        return _coerce(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return Settings()


def save_settings(settings: Settings, path=None) -> bool:
    """Persist settings atomically. False (not an exception) on any
    failure -- a settings write must never take the app down."""
    path = Path(path) if path else settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps(asdict(settings), ensure_ascii=False, indent=2),
            encoding="utf-8")
        tmp.replace(path)
        return True
    except Exception:
        return False
