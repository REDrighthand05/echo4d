"""Hot-update client: the smart shell around the dumb wrench.

Flow (see hotapply.py for the state files under ``.echo/``):

  check    the latest GitHub release carries ``echoscene-manifest.json``
           (the target tree) and ``EchoScene-Update.zip`` (the pack
           built from the previous release). The registered batch-5
           kernels diff the local manifest against the target.
  stage    download the pack (megabytes, not the 638MB installer),
           unpack into ``.echo/staging``, then re-verify EVERYTHING
           locally: the pack's plan must equal the kernel-computed
           plan, its steps must equal the kernel-ordered steps, and
           every staged file must hash to the target manifest.
  apply    write ``pending.json`` and hand over to the frozen
           launcher (``--echo-apply``); the caller exits so the
           wrench can swap files, then relaunches normally.

Eligibility is deliberately narrow: every changed path must live
under ``app/`` (the loose payload) and the pack must start exactly
at the local version. Anything else -- a dependency changed, a
version skipped -- returns None here and the caller falls back to
the full installer, which always works. A staged pack that fails
any verification is wiped, never partially applied.
"""
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from echo_super.ds_batch5 import order_apply_steps, plan_update

from . import hotapply

MANIFEST_ASSET = "echoscene-manifest.json"
PACK_ASSET = "EchoScene-Update.zip"
_USER_AGENT = "ECHO-Scene-Updater"
_TIMEOUT = 30.0


class HotUpdateError(Exception):
    """Any hot-path failure; the caller falls back to the installer."""


@dataclass
class HotInfo:
    version: str
    pack_url: str
    target_manifest: dict
    plan: dict
    total_bytes: int = field(default=0)


def install_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def echo_dir(root: Path) -> Path:
    return Path(root) / hotapply.ECHO_DIR


def local_manifest(root: Path) -> dict | None:
    try:
        return json.loads(
            (echo_dir(root) / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return None


def local_version(root: Path) -> str:
    manifest = local_manifest(root)
    if manifest and isinstance(manifest.get("version"), str):
        return manifest["version"]
    from .version import app_version
    return app_version()


def _asset_url(release: dict, name: str) -> str | None:
    for asset in release.get("assets") or []:
        if isinstance(asset, dict) and asset.get("name") == name:
            return asset.get("browser_download_url")
    return None


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        return response.read()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_hot_update(release: dict, root: Path) -> HotInfo | None:
    """Decide whether the hot path can serve this release. Returns
    None (installer fallback) for any ineligibility; raises nothing."""
    try:
        from .version import is_newer, parse_version

        tag = release.get("tag_name")
        if release.get("draft") or release.get("prerelease"):
            return None
        if parse_version(tag) is None:
            return None
        current = local_version(root)
        if not is_newer(tag, current):
            return None
        local = local_manifest(root)
        if local is None:
            return None
        manifest_url = _asset_url(release, MANIFEST_ASSET)
        pack_url = _asset_url(release, PACK_ASSET)
        if not manifest_url or not pack_url:
            return None
        target = json.loads(_fetch(manifest_url).decode("utf-8"))
        plan = plan_update(local, target)
        if not plan["to_change"] and not plan["to_delete"]:
            return None
        paths = [c["path"] for c in plan["to_change"]] + plan["to_delete"]
        if any(not p.startswith(hotapply.PAYLOAD_PREFIX) for p in paths):
            return None  # a base file changed: installer territory
        return HotInfo(version=target["version"], pack_url=pack_url,
                       target_manifest=target, plan=plan,
                       total_bytes=plan["total_bytes"])
    except Exception:
        return None


def download_and_stage(root: Path, info: HotInfo) -> Path:
    """Download the pack and stage it, fully verified. Raises
    HotUpdateError on any inconsistency; staging is wiped then."""
    root = Path(root)
    staging = echo_dir(root) / "staging"
    try:
        blob = _fetch(info.pack_url)
        archive = zipfile.ZipFile(io.BytesIO(blob))
        names = archive.namelist()
        for name in names:
            if name.startswith("/") or ".." in Path(name).parts:
                raise HotUpdateError(f"unsafe pack entry: {name}")
        if staging.exists():
            import shutil
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        archive.extractall(staging)
        plan = json.loads((staging / "plan.json").read_text(encoding="utf-8"))
        target = json.loads(
            (staging / "target-manifest.json").read_text(encoding="utf-8"))
        local = local_manifest(root)
        if local is None or plan.get("from_version") != local["version"]:
            raise HotUpdateError("pack starts at a different version")
        if target != info.target_manifest:
            raise HotUpdateError("pack manifest differs from release manifest")
        expected = plan_update(local, target)
        if ([c["path"] for c in plan["to_change"]]
                != [c["path"] for c in expected["to_change"]]
                or plan["to_delete"] != expected["to_delete"]):
            raise HotUpdateError("pack plan differs from kernel plan")
        if plan.get("steps") != order_apply_steps(expected):
            raise HotUpdateError("pack steps differ from kernel order")
        for change in expected["to_change"]:
            staged = staging / "files" / change["path"]
            if not staged.exists():
                raise HotUpdateError(f"staged file missing: {change['path']}")
            data = staged.read_bytes()
            if (len(data) != change["size"]
                    or sha256_bytes(data) != change["sha256"]):
                raise HotUpdateError(f"staged file corrupt: {change['path']}")
        return staging
    except HotUpdateError:
        _wipe_staging(root)
        raise
    except Exception as exc:
        _wipe_staging(root)
        raise HotUpdateError(f"{type(exc).__name__}: {exc}") from exc


def _wipe_staging(root: Path) -> None:
    import shutil
    shutil.rmtree(echo_dir(root) / "staging", ignore_errors=True)


def request_apply(root: Path, info: HotInfo) -> list[str]:
    """Mark the staged update pending and return the relaunch
    command for the frozen launcher's apply mode. The caller spawns
    it detached and exits promptly."""
    root = Path(root)
    echo = echo_dir(root)
    echo.mkdir(parents=True, exist_ok=True)
    (echo / "pending.json").write_text(json.dumps(
        {"to_version": info.version}, ensure_ascii=False),
        encoding="utf-8")
    if getattr(sys, "frozen", False):
        return [sys.executable, "--echo-apply-relaunch"]
    return [sys.executable, str(Path(__file__).resolve().parent.parent
                                / "launcher.py"), "--echo-apply"]


def last_result(root: Path) -> dict | None:
    try:
        return json.loads(
            (echo_dir(root) / "result.json").read_text(encoding="utf-8"))
    except Exception:
        return None


def recover(root: Path) -> dict | None:
    """Startup housekeeping: repair an interrupted apply if one is
    pending. Safe to call on every start."""
    try:
        return hotapply.rollback_interrupted(Path(root))
    except Exception:
        return None
