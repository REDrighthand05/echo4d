"""检查更新 (check for updates) against GitHub Releases.

The update path is deliberately boring: ask the GitHub API for the
latest release of REDrighthand05/echo4d, compare its vX.Y.Z tag with
the running version (echo_app.version, from installer/version.inc),
and -- only when the user clicks -- download that release's
EchoScene-Setup.exe and launch it. The Inno installer already
upgrades in place and closes the running app, so there is no
patching framework here at all (hotfix was considered and shelved by
the user: installer upgrades cover the visible benefit without the
maintenance burden).

Everything network-shaped is fail-silent: the startup check runs on a
background thread the UI spawns, every call carries a timeout, and
any error (offline, rate limit, garbled JSON) collapses to None --
"couldn't check" is never an error the user has to read about.

The parsing half (parse_release) is a pure function so tests can pin
it against a recorded API payload with no network.
"""
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .version import is_newer, parse_version

REPO = "REDrighthand05/echo4d"
LATEST_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
SETUP_ASSET = "EchoScene-Setup.exe"
CHECK_TIMEOUT = 6.0          # seconds, per socket operation
DOWNLOAD_TIMEOUT = 30.0      # seconds, per socket read
_USER_AGENT = "ECHO-Scene-Updater"


@dataclass
class UpdateInfo:
    version: str             # "0.5.0" (no leading v)
    tag: str                 # "v0.5.0"
    download_url: str        # browser_download_url of the setup exe
    page_url: str = ""       # the release page, for reference


def parse_release(payload, current_version):
    """Decide from a releases/latest payload whether it offers an
    upgrade. Returns UpdateInfo only when the tag is a plain vX.Y.Z,
    strictly newer than current_version, not a draft/prerelease, and
    the release actually carries the setup exe -- a zip-only release
    must not nag, because there would be nothing calm to offer.
    Any malformed shape returns None. Never raises."""
    try:
        if not isinstance(payload, dict):
            return None
        if payload.get("draft") or payload.get("prerelease"):
            return None
        tag = payload.get("tag_name")
        version = parse_version(tag)
        if version is None or not is_newer(tag, current_version):
            return None
        assets = payload.get("assets")
        if not isinstance(assets, list):
            return None
        for asset in assets:
            if isinstance(asset, dict) and asset.get("name") == SETUP_ASSET:
                url = asset.get("browser_download_url")
                if isinstance(url, str) and url.startswith("https://"):
                    return UpdateInfo(
                        version=".".join(str(p) for p in version),
                        tag=tag,
                        download_url=url,
                        page_url=payload.get("html_url") or "")
        return None
    except Exception:
        return None


def _fetch_json(url=LATEST_URL, timeout=CHECK_TIMEOUT):
    req = urllib.request.Request(
        url, headers={"User-Agent": _USER_AGENT,
                      "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_for_update(current_version, fetch=None):
    """One update check. `fetch` is injectable for tests; the default
    hits the GitHub API with a timeout. Returns UpdateInfo or None;
    None covers BOTH "already latest" and "could not check" -- the
    startup caller must not distinguish them, by design."""
    _status, info = check_for_update_status(current_version, fetch)
    return info


def check_for_update_status(current_version, fetch=None):
    """Tri-state variant for the MANUAL check, whose user deserves an
    honest answer: returns ("update", UpdateInfo), ("latest", None)
    when the check completed and nothing newer exists, or
    ("error", None) when the check itself failed (offline, rate
    limit, garbled response). Never raises."""
    try:
        payload = (fetch or _fetch_json)()
    except Exception:
        return ("error", None)
    info = parse_release(payload, current_version)
    return ("update", info) if info else ("latest", None)


def download_installer(info: UpdateInfo, dest_dir,
                       open_url=None, timeout=DOWNLOAD_TIMEOUT) -> Path:
    """Stream EchoScene-Setup.exe for `info` into dest_dir and return
    its path. Streams to a .part file and renames on success, so a
    half-downloaded installer is never launched. Raises on failure --
    the caller (a background thread in the UI) turns that into one
    calm status line. `open_url` is injectable for tests."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / f"EchoScene-Setup-{info.version}.exe"
    part = final.with_name(final.name + ".part")
    opener = open_url or (
        lambda url: urllib.request.urlopen(
            urllib.request.Request(
                url, headers={"User-Agent": _USER_AGENT}),
            timeout=timeout))
    try:
        with opener(info.download_url) as resp, part.open("wb") as out:
            while True:
                chunk = resp.read(1 << 18)
                if not chunk:
                    break
                out.write(chunk)
        part.replace(final)
        return final
    except Exception:
        part.unlink(missing_ok=True)
        raise
