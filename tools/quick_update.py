"""One-command quick release line (the local fast line).

    python tools/quick_update.py <new_version> --notes-file <path> [--yes]

Ships a payload-only release as a hot-update pack WITHOUT a cloud
build, in minutes:

  1. fetch the latest release's manifest (the lineage clients hold)
  2. diff the local payload tree against it (DS batch-6 kernels:
     select_changes / merge_manifest / align_crlf / looks_text),
     honouring the pinned preview.py exclusion
  3. build the pack with tools/build_update_pack.py
  4. prove it on a fake install with the REAL client code
     (hotupdate.check -> stage -> hotapply.apply)
  5. commit the app repo, push the snapshot, stage the pack +
     target manifest under release-assets/v<X.Y.Z>/ on the branch
  6. create the GitHub release and dispatch release-attach (the
     installer/zip attached there come from the most recent cloud
     build -- they are the fresh-install fallback; the release body
     must say so), then verify the release assets.

Exits non-zero at the first failed gate; nothing is published
before step 5 completes locally.
"""
import argparse
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import (  # noqa: E402
    add_surrogate_to_request, read_json_response)

from echo_super.ds_batch6 import (  # noqa: E402
    align_crlf, looks_text, merge_manifest, select_changes)

REPO = "REDrighthand05/echo4d"
BRANCH = "product/echo-scene"
PAYLOAD_DIRS = ("echo_scene", "echo_app", "echo_super", "web_ui")
PAYLOAD_FILES = ("app.py", "web_app.py", "installer/version.inc")
PINNED = {"app/echo_scene/preview.py"}  # snapshot pins it to HEAD
PUSHER = Path.home() / "workspace" / "echo4d_supervisor" / \
    "upload_snapshot_postv03.py"

# Appended to every quick-line release body: the manual route must
# be written on the release itself, because the users who need it
# are exactly the ones the in-app prompt cannot reach.
STANDARD_FOOTER = """

---

【如果软件里没出现更新提示】多半是网络连不上 GitHub，可以手动升：
下载本页的 EchoScene-Update.zip → 关掉软件（任务栏右下角托盘
右键退出）→ 解压，把 files 里的 app 文件复制进安装文件夹的
app 里替换 → 把 target-manifest.json 改名成 manifest.json，
放进安装文件夹的 .echo 里替换 → 重新打开软件即可。

【新装用户】下方完整安装包是上一版的底座，装好后打开软件会
自动热更新到本版。"""


def api(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request("https://api.github.com" + path,
                                 data=data, method=method)
    req.add_header("Accept", "application/vnd.github+json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    add_surrogate_to_request(req, "custom.github-admin",
                             allowed_hosts=["api.github.com"])
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, read_json_response(r)
    except urllib.error.HTTPError as e:
        return e.code, {"error_detail": e.read().decode()[:400]}


def download_asset(asset_id):
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/releases/assets/{asset_id}")
    req.add_header("Accept", "application/octet-stream")
    add_surrogate_to_request(req, "custom.github-admin",
                             allowed_hosts=["api.github.com"])

    class NoRedir(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a):
            return None

    try:
        urllib.request.build_opener(NoRedir()).open(req)
        raise RuntimeError("asset download did not redirect")
    except urllib.error.HTTPError as e:
        loc = e.headers.get("Location")
        out = subprocess.run(["curl", "-sL", loc], capture_output=True,
                             check=True)
        return out.stdout


def sha(data: bytes):
    return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def payload_candidates():
    """{manifest_path: candidate} for every local payload file."""
    cands = {}
    paths = []
    for d in PAYLOAD_DIRS:
        for f in sorted((ROOT / d).rglob("*")):
            if not f.is_file() or "__pycache__" in f.parts:
                continue
            # DS scratch (raw worker outputs) is not product code
            # and must never ride a release.
            if "ds_out" in f.parts or f.name.startswith("ds_out"):
                continue
            paths.append(f.relative_to(ROOT).as_posix())
    paths += [p for p in PAYLOAD_FILES if (ROOT / p).is_file()]
    for rel in paths:
        raw = (ROOT / rel).read_bytes()
        cands["app/" + rel] = {
            "text": looks_text(raw),
            "lf": sha(raw),
            "crlf": sha(align_crlf(raw)),
        }
    return cands


def tracked(rel: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel],
        capture_output=True).returncode == 0


def bump_version_inc(version: str):
    p = ROOT / "installer" / "version.inc"
    text = p.read_text(encoding="utf-8")
    import re
    text = re.sub(r'#define AppVersion "[0-9.]+"',
                  f'#define AppVersion "{version}"', text)
    text = re.sub(r'#define AppVersionInfo "[0-9.]+"',
                  f'#define AppVersionInfo "{version}.0"', text)
    p.write_text(text, encoding="utf-8")


def simulate(base_manifest, target_manifest, pack_bytes, stage_root):
    """Run the real client chain against a fake install root."""
    from echo_app import hotapply, hotupdate

    root = Path(tempfile.mkdtemp(prefix="quicksim-"))
    (root / ".echo").mkdir(parents=True)
    (root / ".echo" / "manifest.json").write_text(
        json.dumps(base_manifest), encoding="utf-8")
    plan = json.loads(zipfile.ZipFile(io.BytesIO(pack_bytes))
                      .read("plan.json"))
    for c in plan["to_change"]:
        dest = root / c["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"old")
    for rel in plan["to_delete"]:
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"old")
    blobs = {"mem://target": json.dumps(target_manifest).encode(),
             "mem://pack": pack_bytes}
    hotupdate._fetch = lambda url: blobs[url]
    release = {"tag_name": "v" + target_manifest["version"],
               "draft": False, "prerelease": False,
               "assets": [
                   {"name": "echoscene-manifest.json",
                    "browser_download_url": "mem://target"},
                   {"name": "EchoScene-Update.zip",
                    "browser_download_url": "mem://pack"}]}
    info = hotupdate.check_hot_update(release, root)
    assert info is not None, "client judged the pack ineligible"
    hotupdate.download_and_stage(root, info)
    result = hotapply.apply(root)
    assert result.get("ok"), f"apply failed: {result}"
    for c in plan["to_change"]:
        want = (stage_root / c["path"]).read_bytes()
        got = (root / c["path"]).read_bytes()
        assert got == want, f"post-apply bytes differ: {c['path']}"
    for rel in plan["to_delete"]:
        assert not (root / rel).exists(), f"not deleted: {rel}"
    advanced = hotupdate.local_manifest(root)
    assert advanced["version"] == target_manifest["version"]
    print("simulation: client check -> stage -> apply ALL OK")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("version", help="new version, e.g. 0.5.1")
    ap.add_argument("--notes-file", required=True)
    ap.add_argument("--yes", action="store_true",
                    help="skip the plan confirmation prompt")
    args = ap.parse_args(argv)
    version = args.version.lstrip("v")
    notes = Path(args.notes_file).read_text(encoding="utf-8")

    st, rel = api("GET", f"/repos/{REPO}/releases/latest")
    assert st == 200, f"latest release fetch failed: {st} {rel}"
    base_tag = rel["tag_name"]
    man_asset = next(a for a in rel["assets"]
                     if a["name"] == "echoscene-manifest.json")
    base = json.loads(download_asset(man_asset["id"]))
    print(f"base: {base_tag} (manifest v{base['version']}, "
          f"{len(base['files'])} files) -> target v{version}")
    assert base["version"] != version, "already at that version"

    bump_version_inc(version)
    cands = payload_candidates()
    # Visible universe: local candidates + base payload entries that
    # are tracked locally (so genuine deletions register). Foreign
    # base entries (branch-only files) stay untouched.
    visible = {}
    for path, entry in base["files"].items():
        if "ds_out" in path.split("/"):
            continue  # DS scratch, wherever it lives: never ours to
            # change or delete through the quick line
        if not path.startswith("app/"):
            visible[path] = entry
        elif path in cands or tracked(path[len("app/"):]):
            visible[path] = entry
    for pin in PINNED:
        cands.pop(pin, None)
        visible.pop(pin, None)
    sel = select_changes(visible, cands)
    if PINNED & set(sel["changed"]):
        raise AssertionError("pinned file leaked into the change set")
    print(f"changed: {sel['changed']}")
    print(f"deleted: {sel['deleted']}")
    if not sel["changed"] and not sel["deleted"]:
        print("nothing to ship")
        return 1
    if not args.yes:
        try:
            answer = input("ship it? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer != "y":
            print("aborted")
            return 1

    overrides = {}
    stage = Path(tempfile.mkdtemp(prefix="quickstage-"))
    for path in sel["changed"]:
        rel = path[len("app/"):]
        raw = (ROOT / rel).read_bytes()
        data = align_crlf(raw) if sel["ship"][path] == "crlf" else raw
        dest = stage / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        overrides[path] = sha(data)
    for path in sel["deleted"]:
        overrides[path] = None
    target = merge_manifest(base, overrides, version)
    work = Path(tempfile.mkdtemp(prefix="quickwork-"))
    target_p = work / "target-manifest.json"
    target_p.write_text(json.dumps(target, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    base_p = work / "base-manifest.json"
    base_p.write_text(json.dumps(base, ensure_ascii=False),
                      encoding="utf-8")
    from tools.build_update_pack import main as pack_main
    pack_p = work / "EchoScene-Update.zip"
    rc = pack_main(["x", str(base_p), str(stage), str(target_p),
                    str(pack_p)])
    assert rc == 0, f"pack builder exited {rc}"
    pack_bytes = pack_p.read_bytes()
    print(f"pack: {len(pack_bytes)} bytes")
    simulate(base, target, pack_bytes, stage)

    msg = (f"v{version}: quick-line payload update "
           f"({len(sel['changed'])} changed, {len(sel['deleted'])} "
           f"deleted) -- see the GitHub release notes")
    # Stage ONLY the release's own paths -- never a blanket add:
    # the working tree carries unrelated WIP (above all the pinned
    # echo_scene/preview.py, whose local state must never move the
    # pin the snapshot pusher reads from HEAD).
    stage_paths = ["echo_scene", "echo_app", "echo_super", "web_ui",
                   "app.py", "web_app.py", "installer/version.inc",
                   "tools/quick_update.py", "tests/test_hot_update.py"]
    subprocess.run(["git", "-C", str(ROOT), "add", "--"] + stage_paths,
                   check=True)
    subprocess.run(["git", "-C", str(ROOT), "reset", "-q", "--",
                    "echo_scene/preview.py"], check=True)
    subprocess.run(["git", "-C", str(ROOT), "commit", "-m", msg],
                   check=True)
    env = dict(os.environ, QUICK_MSG=msg)
    subprocess.run([sys.executable, str(PUSHER)], check=True, env=env)

    st, ref = api("GET", f"/repos/{REPO}/git/ref/heads/{BRANCH}")
    head = ref["object"]["sha"]
    st, commit = api("GET", f"/repos/{REPO}/git/commits/{head}")
    tree_items = []
    for path, data in (
            (f"release-assets/v{version}/echoscene-manifest.json",
             target_p.read_bytes()),
            (f"release-assets/v{version}/EchoScene-Update.zip",
             pack_bytes)):
        st, blob = api("POST", f"/repos/{REPO}/git/blobs",
                       {"content": base64.b64encode(data).decode(),
                        "encoding": "base64"})
        tree_items.append({"path": path, "mode": "100644",
                           "type": "blob", "sha": blob["sha"]})
    st, tree = api("POST", f"/repos/{REPO}/git/trees",
                   {"base_tree": commit["tree"]["sha"],
                    "tree": tree_items})
    st, nc = api("POST", f"/repos/{REPO}/git/commits",
                 {"message": f"stage v{version} quick-line assets",
                  "tree": tree["sha"], "parents": [head]})
    api("PATCH", f"/repos/{REPO}/git/refs/heads/{BRANCH}",
        {"sha": nc["sha"]})
    print("staged assets committed:", nc["sha"][:12])

    if "手动升" not in notes:
        notes += STANDARD_FOOTER
    st, rel_new = api("POST", f"/repos/{REPO}/releases", {
        "tag_name": f"v{version}", "target_commitish": nc["sha"],
        "name": f"EchoScene v{version}", "body": notes})
    assert st == 201, f"release create failed: {st} {rel_new}"
    print("release:", rel_new["html_url"])

    st, runs = api("GET", f"/repos/{REPO}/actions/workflows/"
                           f"build-windows.yml/runs?branch={BRANCH}"
                           f"&status=success&per_page=1")
    build_run = runs["workflow_runs"][0]["id"]
    try:
        api("POST", f"/repos/{REPO}/actions/workflows/"
                    f"release-attach.yml/dispatches",
            {"ref": BRANCH, "inputs": {"tag": f"v{version}",
                                       "build_run": str(build_run)}})
    except Exception:
        pass  # 204 No Content trips the JSON reader; dispatch landed
    print(f"release-attach dispatched (build run {build_run}); "
          f"waiting for it to finish...")

    import time
    attach_id = None
    for _ in range(10):
        time.sleep(6)
        st, runs = api("GET", f"/repos/{REPO}/actions/workflows/"
                              f"release-attach.yml/runs?per_page=5")
        for r in runs.get("workflow_runs", []):
            if r["event"] == "workflow_dispatch" and \
                    r["status"] != "completed":
                attach_id = r["id"]
                break
        if attach_id:
            break
    assert attach_id, "attach run never appeared"
    conclusion = None
    for _ in range(24):
        st, run = api("GET", f"/repos/{REPO}/actions/runs/{attach_id}")
        if run["status"] == "completed":
            conclusion = run["conclusion"]
            break
        time.sleep(30)
    assert conclusion == "success", f"attach run ended: {conclusion}"
    st, rel_done = api("GET", f"/repos/{REPO}/releases/tags/v{version}")
    assets = {a["name"]: a["size"] for a in rel_done["assets"]}
    expect = {"EchoScene-Update.zip": len(pack_bytes),
              "echoscene-manifest.json": len(target_p.read_bytes())}
    for name, size in expect.items():
        assert assets.get(name) == size, \
            f"asset {name}: got {assets.get(name)}, want {size}"
    for name in ("EchoScene-Setup.exe", "EchoScene-windows.zip"):
        assert name in assets, f"fallback asset missing: {name}"
    print("release assets verified:", assets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
