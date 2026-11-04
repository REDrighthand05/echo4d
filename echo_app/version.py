"""The app version: the install manifest first, the installer stamp second.

Two ledgers carry the product version, and they move at different
speeds. The hot updater advances ``.echo/manifest.json`` with every
pack it applies, while ``installer/version.inc`` is the stamp the
installer was built with -- it only changes when a new installer is
built. The manifest is therefore the authority for a running
install: it is the version the hot-update chain itself believes
(see hotupdate.local_version), and it is the only ledger that moves
after install day. app_version() reads the manifest first, falls
back to version.inc for fresh installs that have no manifest yet
(in the PyInstaller bundle, the copy the spec ships as datas under
installer/; in a source tree, ../../installer/version.inc relative
to this file), and finally to "0.0.0", which simply makes every
real release look newer -- a safe direction to be wrong in.

Reading the manifest here, by hand, is deliberate: hotupdate
imports this module, so this module must not import hotupdate
back. The install roots below mirror hotupdate.install_root's
conventions: a frozen build roots at the executable's directory; a
loose install roots at the parent of the app/ folder this file
ships in; a source checkout roots at the repo top.
"""
import json
import re
import sys
from pathlib import Path

_FALLBACK = "0.0.0"
_DEFINE_RE = re.compile(r'#define\s+AppVersion\s+"([0-9]+(?:\.[0-9]+)*)"')
_TAG_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


def parse_version_inc(text: str):
    """Extract AppVersion from version.inc text, or None."""
    m = _DEFINE_RE.search(text or "")
    return m.group(1) if m else None


def _manifest_candidates():
    """Possible .echo/manifest.json locations, most authoritative
    first (see the module docstring for the root conventions)."""
    if getattr(sys, "frozen", False):
        yield Path(sys.executable).resolve().parent / ".echo" / "manifest.json"
    here = Path(__file__).resolve()
    # Loose install layout: <root>/app/echo_app/version.py
    yield here.parent.parent.parent / ".echo" / "manifest.json"
    # Source checkout: <root>/echo_app/version.py
    yield here.parent.parent / ".echo" / "manifest.json"


def _manifest_version_at(path):
    """The version recorded in the manifest at `path`, or None when
    the file is missing, unreadable, or its version field is not a
    plain three-part version."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    version = data.get("version")
    if not isinstance(version, str):
        return None
    m = _TAG_RE.match(version.strip())
    if m is None:
        return None
    return ".".join(m.groups())


def _version_inc_candidates():
    frozen_base = getattr(sys, "_MEIPASS", None)
    if frozen_base:
        yield Path(frozen_base) / "installer" / "version.inc"
    yield Path(__file__).resolve().parent.parent / "installer" / "version.inc"


def app_version() -> str:
    """The running app's version string, e.g. "0.4.0".

    The install manifest wins over version.inc whenever it can be
    located and parsed -- after any hot update the two disagree,
    and the manifest is the one that is right (see docstring)."""
    for candidate in _manifest_candidates():
        version = _manifest_version_at(candidate)
        if version:
            return version
    for candidate in _version_inc_candidates():
        try:
            version = parse_version_inc(candidate.read_text(encoding="utf-8"))
        except Exception:
            version = None
        if version:
            return version
    return _FALLBACK


APP_VERSION = app_version()


def parse_version(text):
    """Parse "v1.2.3" / "1.2.3" into a (major, minor, patch) tuple of
    ints, or None when the text is not a plain three-part version."""
    if not isinstance(text, str):
        return None
    m = _TAG_RE.match(text.strip())
    if not m:
        return None
    return tuple(int(part) for part in m.groups())


def is_newer(candidate, current) -> bool:
    """True when version `candidate` is strictly newer than `current`.
    Unparseable input is never "newer" -- a garbled tag must not nag
    the user into an update that may not exist."""
    cand = parse_version(candidate)
    cur = parse_version(current)
    if cand is None or cur is None:
        return False
    return cand > cur
