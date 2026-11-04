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
                   设置 → 显示. Since v0.6.12 the ambient look is
                   a baked looping picture (tools/bake_ambient.py),
                   so what is left to tune is playback: which theme,
                   motion on/off, the picture's brightness, and the
                   static veil alphas above it. The veil defaults
                   carry the shipped v0.6.4 numbers over unchanged,
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


# Ambient themes baked into web_ui/assets/ambient/<id>.mp4.
# One ships today; the settings surface is built to carry more.
AMBIENT_THEMES = ("lightfield",)


@dataclass
class Appearance:
    """The web shell's tunable look (设置 → 显示).

    v0.6.12 replaced the live-rendered ambient wall (bead sprites
    under a full-window backdrop blur -- a whole CPU core at idle)
    with a baked seamless loop the page merely plays back. What
    remains tunable is playback and the static veils above the
    picture; there is no blur radius, stroke or mist knob left
    because there is nothing left that renders live. The veil
    defaults are the v0.6.4 shipped numbers carried over, so an
    untouched install looks untouched."""
    # the baked ambient picture (web_ui/assets/ambient/)
    motion_on: bool = True          # False = hold one still frame
    theme: str = "lightfield"       # which baked loop plays
    brightness: float = 100.0       # picture brightness, %
    # the main sheet's static milk-white veil over the picture
    veil: float = 0.26              # wash alpha (was frost_veil)
    # the running state: switchable glass. The veil drops to this
    # alpha so the scene behind it shows through untouched.
    work_veil: float = 0.05         # v0.6.4's cleared pane
    # the settings page's own veil
    set_veil: float = 0.72          # veil alpha (colour depth)


# (field, kind, lo, hi) -- kind "f" float / "i" int / "b" bool /
# "t" theme id (a string from AMBIENT_THEMES). The one table both
# the JSON load and the web bridge clamp by. Keys from the retired
# live-rendered look (frost_blur, graf_*, mist_* ...) are simply
# absent here, which is what retires them: coerce ignores unknown
# keys, so an old settings.json loads onto the new defaults and
# the dead numbers are never written back.
APPEARANCE_SPEC = [
    ("motion_on", "b", None, None),
    ("theme", "t", None, None),
    ("brightness", "f", 50.0, 130.0),
    ("veil", "f", 0.0, 0.9),
    ("work_veil", "f", 0.0, 0.85),
    ("set_veil", "f", 0.0, 0.98),
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
        elif kind == "t":
            if isinstance(value, str) and value in AMBIENT_THEMES:
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
