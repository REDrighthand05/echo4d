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
  appearance    -- the web shell's look, tuned by the user in
                   设置 → 显示: frosted-glass numbers, the giant
                   graffiti, the mist dots and the ambient motion.
                   Defaults are exactly the shipped v0.6.4 values,
                   so an untouched install looks untouched.

Load is forgiving by construction: a missing or corrupt file yields
defaults, unknown keys are ignored, wrong-typed values fall back.
Save is atomic (temp file + replace) so a crash mid-write can never
leave a torn settings file behind.
"""
import json
import os
import sys
from dataclasses import asdict, dataclass, field
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
class Appearance:
    """The web shell's tunable look (设置 → 显示). Every default is
    the value v0.6.4 shipped hard-coded, so carrying these settings
    changes nothing until the user drags a slider."""
    # the main sheet of light glass (#frost)
    frost_blur: float = 2.5        # backdrop blur, px
    frost_veil: float = 0.26       # milk-white wash alpha
    frost_saturate: float = 155.0  # backdrop saturate, %
    # the running state: switchable glass. 0 = v0.6.4's nearly
    # clear pane; higher adds haze back towards the idle frost.
    work_haze: float = 0.0         # 0..100
    # the settings page's own veil
    set_blur: float = 24.0         # backdrop blur, px
    set_veil: float = 0.72         # veil alpha (colour depth)
    set_saturate: float = 200.0    # backdrop saturate, %
    # the giant graffiti wall
    graf_count: int = 0            # strokes per hand; 0 = random 1-3
    graf_width: float = 1.0        # brush width multiplier
    graf_opacity: float = 1.0      # wall layer opacity
    graf_saturate: float = 115.0   # wall layer saturate, %
    # the mist dots the strokes are made of
    mist_density: float = 100.0    # %, 0 = the wall is blank
    mist_size: float = 1.0         # dot radius multiplier
    # ambient motion （常态动效）
    motion_on: bool = True
    motion_speed: float = 1.0      # ambient clock multiplier
    motion_amp: float = 1.0        # wander + breath strength
    sheen_on: bool = True          # the pale sheen pass


# (field, kind, lo, hi) -- kind "f" float / "i" int / "b" bool.
# The one table both the JSON load and the web bridge clamp by.
APPEARANCE_SPEC = [
    ("frost_blur", "f", 0.0, 20.0),
    ("frost_veil", "f", 0.0, 0.9),
    ("frost_saturate", "f", 0.0, 300.0),
    ("work_haze", "f", 0.0, 100.0),
    ("set_blur", "f", 0.0, 40.0),
    ("set_veil", "f", 0.0, 0.98),
    ("set_saturate", "f", 0.0, 300.0),
    ("graf_count", "i", 0, 3),
    ("graf_width", "f", 0.4, 2.0),
    ("graf_opacity", "f", 0.0, 1.0),
    ("graf_saturate", "f", 0.0, 300.0),
    ("mist_density", "f", 0.0, 150.0),
    ("mist_size", "f", 0.5, 2.0),
    ("motion_on", "b", None, None),
    ("motion_speed", "f", 0.2, 3.0),
    ("motion_amp", "f", 0.0, 2.0),
    ("sheen_on", "b", None, None),
]


def coerce_appearance(data, base=None) -> Appearance:
    """Build an Appearance from untrusted data (snake_case keys),
    field by field: unknown keys ignored, wrong types skipped,
    numbers clamped to the slider ranges. `base` carries the
    values a partial update does not mention."""
    out = base if base is not None else Appearance()
    if not isinstance(data, dict):
        return out
    for name, kind, lo, hi in APPEARANCE_SPEC:
        if name not in data:
            continue
        value = data[name]
        if kind == "b":
            if isinstance(value, bool):
                setattr(out, name, value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            value = max(lo, min(hi, float(value)))
            setattr(out, name, int(round(value)) if kind == "i" else value)
    return out


@dataclass
class Settings:
    jobs_root: str = ""          # "" = default (default_jobs_root())
    autostart: bool = False
    close_to_tray: bool = True   # WeChat behaviour out of the box
    update_proxy: str = ""       # "" = direct; else e.g.
    # "http://127.0.0.1:7890" -- update checks/downloads only; some
    # networks cannot reach GitHub without the user's own proxy.
    appearance: Appearance = field(default_factory=Appearance)

    def jobs_root_path(self) -> Path:
        return Path(self.jobs_root) if self.jobs_root else default_jobs_root()


def _coerce(data) -> Settings:
    """Build Settings from untrusted JSON data, field by field."""
    out = Settings()
    if not isinstance(data, dict):
        return out
    if isinstance(data.get("jobs_root"), str):
        out.jobs_root = data["jobs_root"]
    if isinstance(data.get("update_proxy"), str):
        out.update_proxy = data["update_proxy"].strip()
    for key in ("autostart", "close_to_tray"):
        if isinstance(data.get(key), bool):
            setattr(out, key, data[key])
    out.appearance = coerce_appearance(data.get("appearance"))
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
