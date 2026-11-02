"""Benches for DS batch 6 quick-line kernels: hand fixtures plus a
full-scale replay of the real v0.4.0 -> v0.5.0 release (the first
live hot update, built by hand on 2026-11-02). The replay is the
acceptance gate: the kernels must reproduce that release's change
set and target manifest exactly from the raw inputs."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echo_super.ds_batch6 import align_crlf, looks_text, merge_manifest, \
    select_changes

FAIL = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAIL.append(name)


def H(ch):
    return ch * 64


def E(sha, size):
    return {"sha256": sha, "size": size}


# -- merge_manifest ---------------------------------------------------
base = {"version": "0.4.0", "files": {"app/a.py": E(H("a"), 3),
                                     "app/b.py": E(H("b"), 4)}}
snapshot = json.loads(json.dumps(base))
out = merge_manifest(base, {"app/b.py": E(H("c"), 9),
                            "app/d.py": E(H("d"), 1),
                            "app/a.py": None}, "0.5.0")
check("merge replaces version", out["version"] == "0.5.0")
check("merge replaces entry", out["files"]["app/b.py"] == E(H("c"), 9))
check("merge adds entry", out["files"]["app/d.py"] == E(H("d"), 1))
check("merge deletes entry", "app/a.py" not in out["files"])
check("merge does not mutate base", base == snapshot)
for bad in ({"../x": E(H("a"), 1)}, {"app\\x": E(H("a"), 1)},
            {"app/x": E("zz" * 32, 1)}, {"app/x": E(H("a"), -1)},
            {"app/x": E(H("a"), True)}):
    try:
        merge_manifest(base, bad, "0.5.0")
        check(f"merge rejects {list(bad.values())[0]}", False)
    except ValueError:
        check(f"merge rejects {list(bad)[0]} bad entry", True)

# -- looks_text / align_crlf ------------------------------------------
check("text: plain", looks_text(b"hello\nworld\n"))
check("text: utf-8 chinese", looks_text("回声\n".encode("utf-8")))
check("text: empty", looks_text(b""))
check("binary: nul", not looks_text(b"a\x00b"))
check("binary: bad utf-8", not looks_text(b"\xff\xfe\x01"))
check("crlf idempotent", align_crlf(align_crlf(b"a\r\nb\nc\rd")) ==
      align_crlf(b"a\r\nb\nc\rd"))
check("crlf normalizes", align_crlf(b"a\r\nb\nc\rd") ==
      b"a\r\nb\r\nc\r\nd")
check("crlf keeps trailing", align_crlf(b"x\n") == b"x\r\n")

# -- select_changes ----------------------------------------------------
bf = {"app/keep.py": E(H("a"), 3), "app/keep2.py": E(H("b"), 4),
      "app/edit.py": E(H("c"), 5), "app/gone.py": E(H("d"), 6),
      "EchoScene.exe": E(H("e"), 7)}
cand = {
    "app/keep.py": {"text": True, "lf": E(H("a"), 3), "crlf": E(H("1"), 4)},
    "app/keep2.py": {"text": True, "lf": E(H("2"), 5), "crlf": E(H("b"), 4)},
    "app/edit.py": {"text": True, "lf": E(H("f"), 5), "crlf": E(H("3"), 6)},
    "app/new.bin": {"text": False, "lf": E(H("4"), 2), "crlf": E(H("4"), 2)},
}
sel = select_changes(bf, cand)
check("select: lf-unchanged kept", "app/keep.py" not in sel["changed"])
check("select: crlf-unchanged kept", "app/keep2.py" not in sel["changed"])
check("select: changed found", sel["changed"] ==
      ["app/edit.py", "app/new.bin"])
check("select: deleted payload only", sel["deleted"] == ["app/gone.py"])
check("select: ship forms", sel["ship"] ==
      {"app/edit.py": "crlf", "app/new.bin": "raw"})

# -- the v0.5.0 replay --------------------------------------------------
artifacts = Path.home() / "workspace" / ".tmp"
base_p = artifacts / "manifest_v040.json"
target_p = artifacts / "manifest_v050_local.json"
app_root = Path(__file__).resolve().parent.parent
if base_p.exists() and target_p.exists():
    base_m = json.loads(base_p.read_text(encoding="utf-8"))
    target_m = json.loads(target_p.read_text(encoding="utf-8"))
    candidates = {}
    for manifest_path, entry in base_m["files"].items():
        if not manifest_path.startswith("app/"):
            continue
        local = app_root / manifest_path[len("app/"):]
        if not local.is_file():
            continue
        raw = local.read_bytes()
        aligned = align_crlf(raw)
        candidates[manifest_path] = {
            "text": looks_text(raw),
            "lf": {"sha256": hashlib.sha256(raw).hexdigest(),
                   "size": len(raw)},
            "crlf": {"sha256": hashlib.sha256(aligned).hexdigest(),
                     "size": len(aligned)},
        }
    blind = sorted(p for p in base_m["files"]
                   if p.startswith("app/") and p not in candidates)
    check("replay: no payload file is invisible locally", blind == [])
    visible_base = {p: e for p, e in base_m["files"].items()
                    if not p.startswith("app/") or p in candidates}
    raw_sel = select_changes(visible_base, candidates)
    check("replay: pinned WIP preview.py would ship without the guard",
          "app/echo_scene/preview.py" in raw_sel["changed"])
    # The snapshot pusher pins preview.py to its committed version
    # (local WIP must never ride a release); the quick line excludes
    # it the same way before selecting changes.
    pinned = {"app/echo_scene/preview.py"}
    candidates = {p: c for p, c in candidates.items() if p not in pinned}
    visible_base = {p: e for p, e in visible_base.items()
                    if p not in pinned}
    sel = select_changes(visible_base, candidates)
    expect = ["app/installer/version.inc", "app/web_app.py",
              "app/web_ui/app.js", "app/web_ui/index.html",
              "app/web_ui/style.css"]
    check("replay: exact v0.5 change set", sel["changed"] == expect)
    check("replay: no deletions", sel["deleted"] == [])
    check("replay: all ship crlf",
          all(sel["ship"][p] == "crlf" for p in expect))
    overrides = {}
    for path in sel["changed"]:
        overrides[path] = candidates[path]["crlf"]
    rebuilt = merge_manifest(base_m, overrides, "0.5.0")
    check("replay: target manifest reproduced", rebuilt == target_m)
    check("replay: align_crlf == the bytes the hand-built pack shipped",
          all(align_crlf((app_root / p[len("app/"):]).read_bytes()) ==
              (Path.home() / "workspace" / ".tmp" / "dist_v05" / p
               ).read_bytes() for p in expect))
else:
    print("SKIP replay (v0.4/v0.5 manifest artifacts not present)")

print("FAILURES:", len(FAIL), FAIL)
sys.exit(1 if FAIL else 0)
