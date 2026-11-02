"""End-to-end hot-update simulation on fake dist trees.

Covers the whole chain with the real components: build_manifest ->
build_update_pack -> hotupdate.check/stage -> hotapply.apply, plus
the three failure shapes that must never brick an install:
a corrupt pack (rejected at staging), a mid-apply failure (rolled
back), and an interrupted apply (repaired on next start).
"""
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echo_app import hotapply, hotupdate  # noqa: E402
from tools.build_manifest import build_manifest  # noqa: E402
from tools.build_update_pack import main as pack_main  # noqa: E402

FAIL = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAIL.append(name)


def write(root: Path, rel: str, text: str):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def make_tree(root: Path, version: str, code: str, extra=True):
    write(root, "EchoScene.exe", "exe")
    write(root, "_internal/base.dll", "base")
    write(root, "app/echo_app/x.py", code)
    write(root, "app/web_ui/style.css", "css-" + version)
    write(root, "app/installer/version.inc",
          f'#define AppVersion "{version}"')
    if extra:
        write(root, "app/echo_app/old.py", "old")


def run(base: Path):
    dist_a = base / "distA"
    dist_b = base / "distB"
    make_tree(dist_a, "0.4.0", "v1")
    make_tree(dist_b, "0.5.0", "v2", extra=False)
    write(dist_b, "app/echo_app/new.py", "new")

    man_a = build_manifest(dist_a, "0.4.0")
    man_b = build_manifest(dist_b, "0.5.0")
    (base / "man_a.json").write_text(json.dumps(man_a), encoding="utf-8")
    (base / "man_b.json").write_text(json.dumps(man_b), encoding="utf-8")
    # a real install carries its manifest from day one
    echo_a = dist_a / ".echo"
    echo_a.mkdir(exist_ok=True)
    (echo_a / "manifest.json").write_text(json.dumps(man_a),
                                          encoding="utf-8")

    rc = pack_main(["x", str(base / "man_a.json"), str(dist_b),
                    str(base / "man_b.json"), str(base / "pack.zip")])
    check("pack built", rc == 0 and (base / "pack.zip").exists())

    release = {"tag_name": "v0.5.0", "draft": False, "prerelease": False,
               "assets": [
                   {"name": "echoscene-manifest.json",
                    "browser_download_url": "mem://man_b.json"},
                   {"name": "EchoScene-Update.zip",
                    "browser_download_url": "mem://pack.zip"},
                   {"name": "EchoScene-Setup.exe",
                    "browser_download_url": "mem://setup.exe"}]}
    blobs = {"mem://man_b.json": (base / "man_b.json").read_bytes(),
             "mem://pack.zip": (base / "pack.zip").read_bytes()}
    hotupdate._fetch = lambda url: blobs[url]

    info = hotupdate.check_hot_update(release, dist_a)
    check("hot path eligible", info is not None and info.version == "0.5.0")
    check("plan is payload-only",
          all(c["path"].startswith("app/") for c in info.plan["to_change"]))

    hotupdate.download_and_stage(dist_a, info)
    check("staged", (dist_a / ".echo/staging/plan.json").exists())

    result = hotapply.apply(dist_a)
    check("apply ok", result.get("ok") is True)
    check("code swapped", (dist_a / "app/echo_app/x.py")
          .read_text(encoding="utf-8") == "v2")
    check("file added", (dist_a / "app/echo_app/new.py").exists())
    check("file deleted", not (dist_a / "app/echo_app/old.py").exists())
    check("version stamp flipped",
          "0.5.0" in (dist_a / "app/installer/version.inc")
          .read_text(encoding="utf-8"))
    check("base untouched", (dist_a / "_internal/base.dll")
          .read_text(encoding="utf-8") == "base")
    new_local = hotupdate.local_manifest(dist_a)
    check("local manifest advanced",
          new_local and new_local["version"] == "0.5.0")

    # -- base drift must not lock a client out of hot updates ------
    # Same payload, but the base came from a different (equivalent)
    # build: its base-file hashes differ from the release lineage.
    dist_e = base / "distE"
    make_tree(dist_e, "0.4.0", "v1")
    drifted = json.loads(json.dumps(man_a))
    drifted["files"]["EchoScene.exe"] = {"sha256": "0" * 64, "size": 3}
    drifted["files"]["_internal/base.dll"] = {"sha256": "1" * 64,
                                              "size": 4}
    (dist_e / ".echo").mkdir(exist_ok=True)
    (dist_e / ".echo/manifest.json").write_text(json.dumps(drifted),
                                                encoding="utf-8")
    info_e = hotupdate.check_hot_update(release, dist_e)
    check("drifted base still eligible", info_e is not None)
    hotupdate.download_and_stage(dist_e, info_e)
    result_e = hotapply.apply(dist_e)
    check("drifted base applies", result_e.get("ok") is True)
    check("drifted base code swapped",
          (dist_e / "app/echo_app/x.py").read_text(encoding="utf-8")
          == "v2")

    # -- payload lineage drift: extra local payload files ---------
    # The installer-snapshot lineage can ship payload files the
    # quick-line lineage never tracked (build-workorder outputs).
    # They are absent from the target, so the kernel plan deletes
    # them; the pack carries no such steps. Staging must merge the
    # extra deletes instead of rejecting the whole update (the
    # 2026-11-03 live failure: 13 stray files sank a 3-file hop).
    dist_f = base / "distF"
    make_tree(dist_f, "0.4.0", "v1")
    write(dist_f, "app/echo_super/workorders/stray.json", "{}")
    man_f = build_manifest(dist_f, "0.4.0")
    (dist_f / ".echo").mkdir(exist_ok=True)
    (dist_f / ".echo/manifest.json").write_text(json.dumps(man_f),
                                                encoding="utf-8")
    info_f = hotupdate.check_hot_update(release, dist_f)
    check("stray-payload client still eligible", info_f is not None)
    try:
        hotupdate.download_and_stage(dist_f, info_f)
        staged_f = True
    except hotupdate.HotUpdateError:
        staged_f = False
    check("stray deletes merged at staging", staged_f)
    if staged_f:
        staged_plan = json.loads(
            (dist_f / ".echo/staging/plan.json").read_text(
                encoding="utf-8"))
        check("merged plan deletes the stray file",
              "app/echo_super/workorders/stray.json"
              in staged_plan["to_delete"])
        result_f = hotapply.apply(dist_f)
        check("stray-payload applies", result_f.get("ok") is True)
        check("stray file gone after apply",
              not (dist_f / "app/echo_super/workorders/stray.json")
              .exists())
        check("stray-payload code swapped",
              (dist_f / "app/echo_app/x.py")
              .read_text(encoding="utf-8") == "v2")

    # -- failure 1: corrupt pack is rejected at staging -------------
    dist_c = base / "distC"
    make_tree(dist_c, "0.4.0", "v1")
    (dist_c / ".echo").mkdir(exist_ok=True)
    (dist_c / ".echo/manifest.json").write_text(json.dumps(man_a),
                                                encoding="utf-8")
    bad = bytearray((base / "pack.zip").read_bytes())
    with zipfile.ZipFile(base / "pack.zip") as z:
        names = z.namelist()
    # rebuild the zip with one staged file's bytes altered
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(base / "pack.zip") as zin, \
            zipfile.ZipFile(buf, "w") as zout:
        for name in names:
            data = zin.read(name)
            if name == "files/app/echo_app/x.py":
                data = b"tampered"
            zout.writestr(name, data)
    blobs["mem://pack.zip"] = buf.getvalue()
    info_c = hotupdate.check_hot_update(release, dist_c)
    try:
        hotupdate.download_and_stage(dist_c, info_c)
        check("corrupt pack rejected", False)
    except hotupdate.HotUpdateError:
        check("corrupt pack rejected", True)
    check("corrupt pack applied nothing",
          (dist_c / "app/echo_app/x.py").read_text(encoding="utf-8") == "v1")

    # -- failure 2: mid-apply failure rolls back --------------------
    dist_d = base / "distD"
    make_tree(dist_d, "0.4.0", "v1")
    staging = dist_d / ".echo/staging"
    (staging / "files/app/echo_app").mkdir(parents=True)
    import hashlib
    good = b"v2"
    plan = {"from_version": "0.4.0", "to_version": "0.5.0",
            "to_change": [], "to_delete": [],
            "files": {"app/echo_app/x.py":
                      {"sha256": hashlib.sha256(good).hexdigest(),
                       "size": len(good)},
                      "app/echo_app/ghost.py":
                      {"sha256": hashlib.sha256(b"g").hexdigest(),
                       "size": 1}},
            "steps": [{"op": "replace", "path": "app/echo_app/x.py"},
                      {"op": "replace", "path": "app/echo_app/ghost.py"}]}
    (staging / "files/app/echo_app/x.py").write_bytes(good)
    (staging / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (staging / "target-manifest.json").write_text(
        json.dumps({"version": "0.5.0", "files": {}}), encoding="utf-8")
    (dist_d / ".echo/pending.json").write_text("{}", encoding="utf-8")
    result = hotapply.apply(dist_d)
    check("mid-apply failure reported", result.get("ok") is False)
    check("mid-apply rolled back",
          (dist_d / "app/echo_app/x.py").read_text(encoding="utf-8") == "v1")

    # -- failure 3: interrupted apply repaired on next start --------
    backup = dist_d / ".echo/backup"
    (backup / "app/echo_app").mkdir(parents=True, exist_ok=True)
    (backup / "app/echo_app/x.py").write_text("v1", encoding="utf-8")
    (dist_d / "app/echo_app/x.py").write_text("half", encoding="utf-8")
    (dist_d / ".echo/result.json").unlink(missing_ok=True)
    report = hotapply.rollback_interrupted(dist_d)
    check("interrupted apply repaired",
          report is not None
          and (dist_d / "app/echo_app/x.py")
          .read_text(encoding="utf-8") == "v1")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        run(Path(td))
    print("FAILURES:", len(FAIL))
    sys.exit(1 if FAIL else 0)
