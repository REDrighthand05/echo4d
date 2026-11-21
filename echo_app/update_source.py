"""Hot-update source resolution, configuration file support, and handover guide helpers."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

DEFAULT_REPO = "REDrighthand05/echo4d"
DEFAULT_RELAY_REPO = "REDrighthand05/echo4d-update"
DEFAULT_BRANCH = "main"


def _find_update_source_config() -> Optional[Dict[str, Any]]:
    """Look for an update_source.json configuration file.

    Checked locations (in order):
    1. Working directory
    2. The application install root (sys.executable parent or repo root)
    3. The APPDATA/ECHO directory (user persistent settings)
    """
    candidates = []
    try:
        candidates.append(Path.cwd() / "update_source.json")
    except Exception:
        pass

    if getattr(sys, "frozen", False):
        try:
            candidates.append(Path(sys.executable).resolve().parent / "update_source.json")
            candidates.append(Path(sys.executable).resolve().parent / "app" / "update_source.json")
        except Exception:
            pass
    else:
        try:
            candidates.append(Path(__file__).resolve().parent.parent / "update_source.json")
        except Exception:
            pass

    try:
        from echo_app.settings import settings_dir

        candidates.append(settings_dir() / "update_source.json")
    except Exception:
        pass

    for candidate in candidates:
        if candidate and candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
    return None


def get_repo() -> str:
    """Resolve the GitHub repository slug (e.g. 'owner/repo') for updates."""
    env = os.environ.get("ECHO_UPDATE_REPO", "").strip()
    if env:
        return env

    cfg = _find_update_source_config()
    if cfg and cfg.get("repo"):
        return str(cfg["repo"]).strip()

    return DEFAULT_REPO


def get_relay_repo() -> str:
    """Resolve the GitHub Pages/Relay repository slug (e.g. 'owner/repo-update')."""
    env = os.environ.get("ECHO_UPDATE_RELAY_REPO", "").strip()
    if env:
        return env

    # If ECHO_UPDATE_REPO is explicitly in environment, derive from it unless relay repo was specified
    if os.environ.get("ECHO_UPDATE_REPO", "").strip():
        repo = get_repo()
        if "/" in repo:
            owner, name = repo.split("/", 1)
            return f"{owner}/{name}-update"

    cfg = _find_update_source_config()
    if cfg and cfg.get("relay_repo"):
        return str(cfg["relay_repo"]).strip()

    repo = get_repo()
    if "/" in repo:
        owner, name = repo.split("/", 1)
        return f"{owner}/{name}-update"
    return DEFAULT_RELAY_REPO


def get_latest_url() -> str:
    """Resolve the latest release metadata URL."""
    env = os.environ.get("ECHO_UPDATE_LATEST_URL", "").strip()
    if env:
        return env
    cfg = _find_update_source_config()
    if cfg and cfg.get("latest_url"):
        return str(cfg["latest_url"]).strip()
    return f"https://api.github.com/repos/{get_repo()}/releases/latest"


def get_releases_url(per_page: int = 30) -> str:
    """Resolve the releases list URL."""
    env = os.environ.get("ECHO_UPDATE_RELEASES_URL", "").strip()
    if env:
        return env
    cfg = _find_update_source_config()
    if cfg and cfg.get("releases_url"):
        return str(cfg["releases_url"]).strip()
    return f"https://api.github.com/repos/{get_repo()}/releases?per_page={per_page}"


def get_relay_bases() -> Tuple[str, ...]:
    """Resolve the ordered tuple of relay URLs for static updates."""
    env = os.environ.get("ECHO_UPDATE_RELAY_BASE", "").strip()
    if env:
        return (env,)

    # If ECHO_UPDATE_REPO was set in env, derive relay bases from it
    if os.environ.get("ECHO_UPDATE_REPO", "").strip():
        relay_repo = get_relay_repo()
        owner = relay_repo.split("/")[0] if "/" in relay_repo else "REDrighthand05"
        name = relay_repo.split("/")[1] if "/" in relay_repo else "echo4d-update"
        jsdelivr = f"https://cdn.jsdelivr.net/gh/{relay_repo}@{DEFAULT_BRANCH}"
        gh_pages = f"https://{owner.lower()}.github.io/{name}"
        return (jsdelivr, gh_pages)

    cfg = _find_update_source_config()
    if cfg:
        custom_bases = cfg.get("relay_bases")
        if isinstance(custom_bases, (list, tuple)) and custom_bases:
            return tuple(str(b).strip() for b in custom_bases if b)
        single = cfg.get("relay_base")
        if single:
            return (str(single).strip(),)

    relay_repo = get_relay_repo()
    owner = relay_repo.split("/")[0] if "/" in relay_repo else "REDrighthand05"
    name = relay_repo.split("/")[1] if "/" in relay_repo else "echo4d-update"

    jsdelivr = f"https://cdn.jsdelivr.net/gh/{relay_repo}@{DEFAULT_BRANCH}"
    gh_pages = f"https://{owner.lower()}.github.io/{name}"
    return (jsdelivr, gh_pages)


def get_source_manifest() -> Dict[str, Any]:
    """Diagnostic manifest explaining active update source routing for maintainers."""
    cfg = _find_update_source_config()
    return {
        "repo": get_repo(),
        "relay_repo": get_relay_repo(),
        "latest_url": get_latest_url(),
        "releases_url": get_releases_url(),
        "relay_bases": list(get_relay_bases()),
        "has_override_file": cfg is not None,
        "is_customized": bool(
            os.environ.get("ECHO_UPDATE_REPO")
            or (cfg and (cfg.get("repo") or cfg.get("relay_bases") or cfg.get("relay_base")))
        ),
    }
