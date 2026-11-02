"""Frozen entry point of the shipped app (the base layer).

This tiny script is what PyInstaller freezes into the executable.
Everything else -- the whole product -- ships as loose files under
``app/`` next to the exe and is hot-updatable. Jobs, in order:

1. ``--echo-apply``: run the stage-0 wrench (echo_app.hotapply)
   against the install root and exit with its status. Retries for a
   few seconds: the parent app that spawned us may still be letting
   go of file handles.
2. Interrupted-update repair: if ``.echo/pending.json`` exists with
   no ``result.json``, a previous apply died mid-swap -- restore the
   backup wholesale before anything else starts. This mini-rollback
   is self-contained on purpose (stdlib only, no payload imports):
   it must work even when the payload itself is what is broken.
3. Normal start: put ``app/`` on sys.path, pull in frozen_imports
   (the build-time dependency magnet), then run the web shell,
   falling back to the Tk shell if the web shell fails to import.

In a source tree (not frozen) the payload is the repo itself and
step 3 simply imports web_app from here.
"""
import json
import shutil
import sys
import time
from pathlib import Path

APPLY_ARG = "--echo-apply"


def install_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def payload_dir(root: Path) -> Path:
    candidate = root / "app"
    return candidate if candidate.is_dir() else root


def _mini_rollback(root: Path) -> None:
    """Restore .echo/backup wholesale; mirrors
    echo_app.hotapply.rollback_interrupted without importing it."""
    echo = root / ".echo"
    pending = echo / "pending.json"
    backup = echo / "backup"
    if not pending.exists() or (echo / "result.json").exists():
        return
    if not backup.is_dir():
        pending.unlink(missing_ok=True)
        return
    restored = []
    for item in sorted(backup.rglob("*")):
        if not item.is_file():
            continue
        rel = item.relative_to(backup).as_posix()
        if rel.endswith(".absent"):
            target = root / rel[: -len(".absent")]
            if target.exists():
                target.unlink()
            restored.append(rel[: -len(".absent")])
        else:
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            restored.append(rel)
    pending.unlink(missing_ok=True)
    (echo / "result.json").write_text(json.dumps(
        {"ok": False, "error": "interrupted apply rolled back",
         "rolled_back": restored}, ensure_ascii=False),
        encoding="utf-8")


def _apply(root: Path) -> int:
    sys.path.insert(0, str(payload_dir(root)))
    from echo_app.hotapply import apply

    result = {"ok": False, "error": "not attempted"}
    for attempt in range(6):
        result = apply(root)
        if result.get("ok"):
            break
        err = str(result.get("error", ""))
        if "PermissionError" not in err and "being used" not in err:
            break
        time.sleep(1.0)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("ok") else 1


def main() -> int:
    root = install_root()
    if "--echo-apply-relaunch" in sys.argv:
        rc = _apply(root)
        if getattr(sys, "frozen", False):
            import subprocess
            subprocess.Popen([sys.executable],
                             close_fds=True, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        return rc
    if APPLY_ARG in sys.argv:
        return _apply(root)
    try:
        _mini_rollback(root)
    except Exception:
        pass  # repair must never block the app from starting
    sys.path.insert(0, str(payload_dir(root)))
    try:
        import frozen_imports  # noqa: F401  (dependency magnet)
    except Exception:
        pass  # in a source tree the magnet is optional
    try:
        import web_app
        return web_app.main()
    except Exception:
        import app
        return app.main()


if __name__ == "__main__":
    raise SystemExit(main())
