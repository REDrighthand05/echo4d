"""The app version: one source of truth, shared with the installer.

installer/version.inc is THE version source for the whole product --
the Inno Setup script includes it and the Windows build workflow
reads it (or overrides it from the release tag vX.Y.Z). The app reads
the same file so the version it compares against GitHub Releases can
never drift from the version the installer stamped: in a source tree
it reads ../../installer/version.inc relative to this file; in the
PyInstaller bundle it reads the copy the spec ships as datas under
installer/. If neither is readable the fallback is "0.0.0", which
simply makes every real release look newer -- a safe direction to be
wrong in.
"""
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


def _version_inc_candidates():
    frozen_base = getattr(sys, "_MEIPASS", None)
    if frozen_base:
        yield Path(frozen_base) / "installer" / "version.inc"
    yield Path(__file__).resolve().parent.parent / "installer" / "version.inc"


def app_version() -> str:
    """The running app's version string, e.g. "0.4.0"."""
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
