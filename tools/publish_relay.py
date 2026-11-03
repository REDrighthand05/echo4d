#!/usr/bin/env python3
"""Publish EchoScene update assets to the relay site.

The relay is a plain static site (GitHub Pages on the dedicated
``echo4d-update`` repo) mirroring, per release, the two small files
the hot-update path needs -- ``echoscene-manifest.json`` and
``EchoScene-Update.zip`` -- plus an ``index.json`` describing the
hop chain. The app checks the relay first and only falls back to
the GitHub API when the relay cannot answer, because the API's
unauthenticated rate limit is shared per egress IP and kept
locking real users out (seen live 2026-11-03: 403 rate limit on
the user's shared exit IP while plain downloads worked fine).

Usage: publish_relay.py 0.5.5 0.5.6 [...]   (versions ascending)
Each version's "from" is the previous published version (or the
previous argument). Release assets are read from the echo4d
releases through the authenticated API helper.
"""
import base64
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import (  # noqa: E402
    add_surrogate_to_request, read_json_response)

SRC_REPO = "REDrighthand05/echo4d"
RELAY_REPO = "REDrighthand05/echo4d-update"


def api(method, path, body=None, repo=None, raw=False):
    url = "https://api.github.com" + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/vnd.github+json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    add_surrogate_to_request(req, "custom.github-admin",
                             allowed_hosts=["api.github.com"])
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            if raw:
                return r.status, r.read()
            return r.status, read_json_response(r)
    except urllib.error.HTTPError as e:
        return e.code, {"error_detail": e.read().decode()[:400]}


def download_asset(repo, asset_id):
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/releases/assets/{asset_id}")
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


def ensure_repo():
    st, info = api("GET", f"/repos/{RELAY_REPO}")
    if st == 404:
        st, info = api("POST", "/user/repos",
                       {"name": "echo4d-update",
                        "description": "EchoScene update relay "
                                       "(manifests + hot-update packs)",
                        "private": False, "auto_init": True})
        print("repo created:", st)
    st, pages = api("GET", f"/repos/{RELAY_REPO}/pages")
    if st == 404:
        st, pages = api("POST", f"/repos/{RELAY_REPO}/pages",
                        {"source": {"branch": "main", "path": "/"}})
        print("pages enabled:", st)


def release_assets(version):
    st, rel = api("GET", f"/repos/{SRC_REPO}/releases/tags/v{version}")
    if st != 200:
        raise RuntimeError(f"release v{version}: {st} {rel}")
    out = {}
    for a in rel["assets"]:
        out[a["name"]] = a
    return out


def main(argv):
    versions = argv[1:]
    if not versions:
        print(__doc__)
        return 2
    ensure_repo()
    # current index, if any
    st, cur = api("GET", f"/repos/{RELAY_REPO}/contents/index.json")
    entries = []
    if st == 200:
        entries = json.loads(base64.b64decode(
            cur["content"]).decode("utf-8"))["versions"]
    by_ver = {e["version"]: e for e in entries}
    blobs = {}  # path -> bytes to write this run
    prev = entries[-1]["version"] if entries else None
    for ver in versions:
        assets = release_assets(ver)
        man = download_asset(SRC_REPO,
                             assets["echoscene-manifest.json"]["id"])
        pack = download_asset(SRC_REPO,
                              assets["EchoScene-Update.zip"]["id"])
        # The pack itself states its base version in plan.json --
        # trust that over chain bookkeeping.
        import io
        import zipfile
        with zipfile.ZipFile(io.BytesIO(pack)) as z:
            pack_from = json.loads(z.read("plan.json"))["from_version"]
        frm = pack_from or by_ver.get(ver, {}).get("from") or prev
        entry = {"version": ver, "from": frm,
                 "manifest": f"v{ver}/echoscene-manifest.json",
                 "pack": f"v{ver}/EchoScene-Update.zip",
                 "setup_size": assets["EchoScene-Setup.exe"]["size"]}
        by_ver[ver] = entry
        blobs[entry["manifest"]] = man
        blobs[entry["pack"]] = pack
        prev = ver
        print(f"v{ver}: manifest {len(man)} B, pack {len(pack)} B, "
              f"from {frm}")
    ordered = sorted(by_ver.values(),
                     key=lambda e: [int(p) for p in e["version"].split(".")])
    index = {"product": "EchoScene", "latest": ordered[-1]["version"],
             "versions": ordered}
    blobs["index.json"] = json.dumps(
        index, ensure_ascii=False, indent=1).encode("utf-8")

    # one commit via the git data API, preserving untouched files
    st, ref = api("GET", f"/repos/{RELAY_REPO}/git/ref/heads/main")
    base_commit = ref["object"]["sha"]
    st, commit = api("GET",
                     f"/repos/{RELAY_REPO}/git/commits/{base_commit}")
    base_tree = commit["tree"]["sha"]
    tree_items = []
    for path, data in blobs.items():
        st, blob = api("POST", f"/repos/{RELAY_REPO}/git/blobs",
                       {"content": base64.b64encode(data).decode(),
                        "encoding": "base64"})
        tree_items.append({"path": path, "mode": "100644",
                           "type": "blob", "sha": blob["sha"]})
        print("blob:", path)
    st, tree = api("POST", f"/repos/{RELAY_REPO}/git/trees",
                   {"base_tree": base_tree, "tree": tree_items})
    st, new_commit = api(
        "POST", f"/repos/{RELAY_REPO}/git/commits",
        {"message": "relay: publish " + ", ".join("v" + v
                                                  for v in versions),
         "tree": tree["sha"], "parents": [base_commit]})
    st, upd = api("PATCH", f"/repos/{RELAY_REPO}/git/refs/heads/main",
                  {"sha": new_commit["sha"]})
    print("relay main ->", new_commit["sha"][:12], "update:", st)
    print("index:", json.dumps(index)[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
