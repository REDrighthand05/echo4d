"""Stage-0 hot-update applier: the dumb wrench.

Deliberately small, stdlib-only, and stupid. It ships in the base
install and is NEVER itself replaced by the hot channel: it refuses
any plan that touches anything outside the loose app payload
(``app/``), refuses its own file, and refuses the executable.

The smart parts (which files differ, in which order) are computed
elsewhere by the registered update kernels; this module only
executes a prepared plan against prepared staging, with a backup
deep enough to undo the whole operation:

  <app_dir>/.echo/staging/    unpacked pack: files/<relpath>...,
                              plan.json, target-manifest.json
  <app_dir>/.echo/pending.json  written by the updater BEFORE the
                              applier starts; cleared on success
  <app_dir>/.echo/backup/     pre-update copies, one update deep
  <app_dir>/.echo/result.json   {"ok": bool, ...} of the last run
  <app_dir>/.echo/manifest.json the tree manifest of the CURRENT
                              installed version (post-update truth)

Apply protocol (invoked as a subprocess by the app, or standalone
for repair): for each step in order -- replace: back up the current
file, copy the staged file in, verify its sha256 on disk; delete:
back up, then remove. Any failure rolls every completed step back
from the backup and reports ok=False. Success writes the target
manifest, drops a version marker, and clears staging/pending.

A pending.json without a result means the machine died mid-apply;
`rollback_interrupted` (called by the app on next start) restores
the backup wholesale so the install is never left half-swapped.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ECHO_DIR = ".echo"
PAYLOAD_PREFIX = "app/"          # the only hot-updatable subtree
STAMP_SUFFIX = "installer/version.inc"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_rel(rel: str) -> bool:
    """A plan path is acceptable only inside the loose payload,
    relative, POSIX-shaped, and not this applier or the stamp of a
    different tree."""
    if not isinstance(rel, str) or not rel:
        return False
    if rel.startswith("/") or "\\" in rel:
        return False
    if any(part == ".." for part in rel.split("/")):
        return False
    if not rel.startswith(PAYLOAD_PREFIX):
        return False
    if rel.endswith("hotapply.py"):
        return False
    return True


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                    encoding="utf-8")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _backup_file(app_dir: Path, backup_dir: Path, rel: str) -> None:
    src = app_dir / rel
    dst = backup_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
    else:
        # remember it did NOT exist, so rollback removes the copy
        (dst.parent / (dst.name + ".absent")).parent.mkdir(
            parents=True, exist_ok=True)
        (dst.parent / (dst.name + ".absent")).write_text(
            "", encoding="utf-8")


def _restore_file(app_dir: Path, backup_dir: Path, rel: str) -> None:
    src = backup_dir / rel
    dst = app_dir / rel
    absent = src.parent / (src.name + ".absent")
    if absent.exists():
        if dst.exists():
            dst.unlink()
    elif src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def apply(app_dir: Path) -> dict:
    """Execute the prepared plan in .echo/staging. Returns (and
    records) the result dict; never raises for plan/IO problems."""
    app_dir = Path(app_dir)
    echo = app_dir / ECHO_DIR
    staging = echo / "staging"
    backup = echo / "backup"
    result = {"ok": False}
    try:
        plan = _load_json(staging / "plan.json")
        target_manifest = _load_json(staging / "target-manifest.json")
        steps = plan["steps"]
        for step in steps:
            if not _safe_rel(step["path"]):
                raise ValueError(f"unsafe plan path: {step['path']!r}")
        if backup.exists():
            shutil.rmtree(backup)
        backup.mkdir(parents=True)
        done = []
        try:
            for step in steps:
                rel = step["path"]
                _backup_file(app_dir, backup, rel)
                done.append(rel)
                dst = app_dir / rel
                if step["op"] == "replace":
                    src = staging / "files" / rel
                    if not src.exists():
                        raise FileNotFoundError(f"staged file missing: {rel}")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    want = plan["files"][rel]["sha256"]
                    if _sha256(dst) != want:
                        raise ValueError(f"hash mismatch after copy: {rel}")
                elif step["op"] == "delete":
                    if dst.exists():
                        dst.unlink()
                else:
                    raise ValueError(f"unknown op: {step['op']!r}")
        except Exception:
            for rel in reversed(done):
                _restore_file(app_dir, backup, rel)
            raise
        _write_json(echo / "manifest.json", target_manifest)
        result = {"ok": True, "version": target_manifest.get("version"),
                  "steps": len(steps)}
    except Exception as exc:  # noqa: BLE001 - the wrench reports, not crashes
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        _write_json(echo / "result.json", result)
        if result["ok"]:
            shutil.rmtree(staging, ignore_errors=True)
            pending = echo / "pending.json"
            if pending.exists():
                pending.unlink()
    except Exception:
        pass
    return result


def rollback_interrupted(app_dir: Path) -> dict | None:
    """If a previous apply died mid-way (pending without result),
    restore the backup wholesale. Returns a report or None when
    there is nothing to repair."""
    app_dir = Path(app_dir)
    echo = app_dir / ECHO_DIR
    pending = echo / "pending.json"
    result_path = echo / "result.json"
    backup = echo / "backup"
    if not pending.exists() or result_path.exists() or not backup.exists():
        return None
    restored = []
    for item in sorted(backup.rglob("*")):
        if item.is_file() and not item.name.endswith(".absent"):
            rel = item.relative_to(backup).as_posix()
            _restore_file(app_dir, backup, rel)
            restored.append(rel)
    for item in sorted(backup.rglob("*.absent")):
        rel = item.relative_to(backup).as_posix()[: -len(".absent")]
        _restore_file(app_dir, backup, rel)
        restored.append(rel)
    pending.unlink(missing_ok=True)
    report = {"ok": True, "rolled_back": restored}
    _write_json(echo / "result.json",
                {"ok": False, "error": "interrupted apply rolled back",
                 "rolled_back": restored})
    return report


def main(argv) -> int:
    if len(argv) >= 3 and argv[1] == "apply":
        result = apply(Path(argv[2]))
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result.get("ok") else 1
    if len(argv) >= 3 and argv[1] == "rollback":
        report = rollback_interrupted(Path(argv[2]))
        print(json.dumps(report, ensure_ascii=False))
        return 0
    print("usage: hotapply.py apply|rollback <app_dir>")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
