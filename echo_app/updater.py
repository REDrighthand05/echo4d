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
RELEASES_URL = f"https://api.github.com/repos/{REPO}/releases?per_page=30"
SETUP_ASSET = "EchoScene-Setup.exe"
CHECK_TIMEOUT = 6.0          # seconds, per socket operation
DOWNLOAD_TIMEOUT = 30.0      # seconds, per socket read
_USER_AGENT = "ECHO-Scene-Updater"

# --- Reaching GitHub: try every route, so the user never has to.
#
# The update machinery has exactly one external job: reach GitHub.
# On some networks a plain connection never arrives (a whole
# evening was lost to exactly that), so _open walks a route list
# until one answers:
#   1. the proxy the user typed in Settings, if any
#   2. direct (urllib also honours the Windows system proxy here)
#   3. the usual local proxy ports, probed in order
# The first route that answers is remembered for the process, and
# an auto-found local proxy is reported through auto_proxy() so
# the bridge can save it -- next launch it becomes route 1 and
# nobody has to think about any of this again.
_PROXY = ""
_ROUTE = None            # None = untried; "" = direct; else a URL
_AUTO_PROXY = ""         # a probed candidate that answered
PROBE_TIMEOUT = 3.0
CANDIDATE_PROXIES = (
    "http://127.0.0.1:7897",   # Clash Verge
    "http://127.0.0.1:7890",   # Clash
    "http://127.0.0.1:10809",  # v2rayN
    "http://127.0.0.1:10808",  # v2rayN mixed port
    "http://127.0.0.1:1080",
    "http://127.0.0.1:8080",
)


def set_proxy(url):
    global _PROXY, _ROUTE
    _PROXY = (url or "").strip()
    _ROUTE = None            # the routes changed: probe afresh


def auto_proxy():
    """The auto-found local proxy that answered, if one did."""
    return _AUTO_PROXY


def _urlopen_with(route, req, timeout):
    """One attempt on one route ('' = direct / default opener)."""
    if route:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler(
            {"http": route, "https": route}))
        return opener.open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


def _routes():
    seen, out = set(), []
    head = [_PROXY] if _PROXY else []
    for route in head + [""] + list(CANDIDATE_PROXIES):
        if route not in seen:
            seen.add(route)
            out.append(route)
    return out


def _open(req, timeout):
    """Open req on the first route that answers and remember it.
    While the route is unknown each attempt is capped at
    PROBE_TIMEOUT so dead routes cannot eat the caller's budget;
    a winning probe is then re-issued at the full timeout (every
    request here is an idempotent GET). Raises the last error
    when no route answers."""
    global _ROUTE, _AUTO_PROXY
    if _ROUTE is not None:
        return _urlopen_with(_ROUTE, req, timeout)
    last_exc = None
    for route in _routes():
        try:
            resp = _urlopen_with(route, req, min(timeout, PROBE_TIMEOUT))
        except Exception as exc:  # dead route: try the next one
            last_exc = exc
            continue
        _ROUTE = route
        if route and route != _PROXY:
            _AUTO_PROXY = route
        if timeout > PROBE_TIMEOUT:
            try:
                resp.close()
            except Exception:
                pass
            return _urlopen_with(route, req, timeout)
        return resp
    if last_exc is not None:
        raise last_exc
    raise OSError("no update route available")


@dataclass
class UpdateInfo:
    version: str             # "0.5.0" (no leading v)
    tag: str                 # "v0.5.0"
    download_url: str        # browser_download_url of the setup exe
    page_url: str = ""       # the release page, for reference
    size: int = 0            # asset size from the release, when known


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
                    size = asset.get("size")
                    return UpdateInfo(
                        version=".".join(str(p) for p in version),
                        tag=tag,
                        download_url=url,
                        page_url=payload.get("html_url") or "",
                        size=size if isinstance(size, int) and size > 0
                        else 0)
        return None
    except Exception:
        return None


# --- The relay: our own static mirror of the update assets. ------
#
# The GitHub API is the one door with a quota: unauthenticated calls
# share 60 requests/hour per egress IP, and a user behind a shared
# proxy exit found it permanently exhausted (403 "rate limit
# exceeded" while every plain download worked). So the app checks
# the relay first -- a static site carrying, per release, the
# manifest and the hot-update pack plus an index of the hop chain --
# and the API remains only as a fallback. Static files have no
# quota, and the route cascade above still applies to reaching the
# relay host itself.
RELAY_BASE = "https://redrighthand05.github.io/echo4d-update"


def fetch_relay_index():
    """GET the relay index through the route cascade. Raises on any
    failure; the caller falls back to the GitHub API flow."""
    req = urllib.request.Request(
        RELAY_BASE + "/index.json",
        headers={"User-Agent": _USER_AGENT, "Accept": "application/json"})
    with _open(req, CHECK_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def relay_plan(current_version, index):
    """Decide the update from a relay index. Pure, so tests can pin
    it. Returns ("latest", None, None), ("update", UpdateInfo,
    hop_release_or_None), or None when the index is unusable (the
    caller then falls back to the GitHub API flow). The hop is the
    entry whose "from" is exactly the running version -- packs are
    chained, so any other entry would not apply."""
    try:
        entries = index["versions"]
        if not isinstance(entries, list) or not entries:
            return None
        newer = []
        for entry in entries:
            tag = "v" + str(entry["version"])
            if is_newer(tag, current_version):
                newer.append(entry)
        if not newer:
            return ("latest", None, None)
        newer.sort(key=lambda e: parse_version("v" + str(e["version"])))
        latest_entry = newer[-1]
        latest_tag = "v" + str(latest_entry["version"])
        setup_url = (f"https://github.com/{REPO}/releases/download/"
                     f"{latest_tag}/{SETUP_ASSET}")
        size = latest_entry.get("setup_size")
        info = UpdateInfo(
            version=str(latest_entry["version"]), tag=latest_tag,
            download_url=setup_url,
            size=size if isinstance(size, int) and size > 0 else 0)
        hop_release = None
        for entry in newer:
            if entry.get("from") == current_version:
                hop_release = {
                    "tag_name": "v" + str(entry["version"]),
                    "draft": False, "prerelease": False,
                    "assets": [
                        {"name": "echoscene-manifest.json",
                         "browser_download_url":
                             RELAY_BASE + "/" + entry["manifest"]},
                        {"name": "EchoScene-Update.zip",
                         "browser_download_url":
                             RELAY_BASE + "/" + entry["pack"]},
                        {"name": SETUP_ASSET,
                         "browser_download_url": setup_url}]}
                break
        return ("update", info, hop_release)
    except Exception:
        return None


def _fetch_json(url=LATEST_URL, timeout=CHECK_TIMEOUT):
    req = urllib.request.Request(
        url, headers={"User-Agent": _USER_AGENT,
                      "Accept": "application/vnd.github+json"})
    with _open(req, timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_releases():
    """The recent-releases list (newest first). Raises on any
    failure -- callers decide what silence is worth."""
    payload = _fetch_json(RELEASES_URL)
    if not isinstance(payload, list):
        raise ValueError("releases payload is not a list")
    return payload


def pick_hop(releases, current_version):
    """The next single step up from current_version: the release
    with the SMALLEST parseable version that is still strictly
    newer. Hot packs are chained (each starts where the previous
    ended), so a client several versions behind must climb one
    release at a time; this picks its next handhold. None when
    already at the top. Pure; never raises."""
    try:
        best = None
        best_key = None
        for rel in releases or []:
            if not isinstance(rel, dict):
                continue
            if rel.get("draft") or rel.get("prerelease"):
                continue
            key = parse_version(rel.get("tag_name"))
            if key is None or not is_newer(rel.get("tag_name"),
                                            current_version):
                continue
            if best_key is None or key < best_key:
                best, best_key = rel, key
        return best
    except Exception:
        return None


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


def _content_length(resp) -> int:
    """The response's Content-Length, or 0 when it does not say."""
    try:
        value = resp.headers.get("Content-Length")
        return int(value) if value else 0
    except Exception:
        return 0


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
        lambda url: _open(
            urllib.request.Request(
                url, headers={"User-Agent": _USER_AGENT}),
            timeout))
    try:
        written = 0
        with opener(info.download_url) as resp, part.open("wb") as out:
            while True:
                chunk = resp.read(1 << 18)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
            expected = info.size or _content_length(resp)
            if expected and written != expected:
                # A dropped connection can look exactly like a clean
                # EOF to urllib; without this check the truncated
                # file was renamed and launched, and the user got
                # Inno's "setup files are corrupted" instead of a
                # plain "download failed" (seen live, 2026-11-03:
                # 60 MB of a 640 MB installer).
                raise OSError(
                    f"installer download incomplete: {written} of "
                    f"{expected} bytes")
        part.replace(final)
        return final
    except Exception:
        part.unlink(missing_ok=True)
        raise
