#!/usr/bin/env python3
"""Publish an EchoCollector APK to the update relay.

The relay (GitHub Pages on REDrighthand05/echo4d-update) gains a
``collector/`` section: one folder per version holding the APK,
plus ``collector/index.json`` listing every published version
with its versionCode and sha256. The app's updater checks that
index (Pages first, jsDelivr mirror second), downloads the APK
and verifies the hash before handing it to the system
installer -- the same relay-first doctrine as the desktop app,
because the GitHub API's shared rate limit locks real users
out.

Usage: publish_collector.py 0.7 "更新说明"
The APK is read from android_collector/EchoCollector.apk; its
versionCode is read from the APK badging, not from arguments.
"""
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.publish_relay import api  # noqa: E402  (authenticated)

RELAY = "REDrighthand05/echo4d-update"
AAPT2 = (Path.home() / "workspace/android-tools/sdk/build-tools"
         / "34.0.0/aapt2")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    version = argv[1].lstrip("v")
    notes = argv[2] if len(argv) > 2 else ""
    apk = ROOT / "android_collector" / "EchoCollector.apk"
    data = apk.read_bytes()
    badging = subprocess.run(
        [str(AAPT2), "dump", "badging", str(apk)],
        capture_output=True, text=True, check=True).stdout
    code = None
    for token in badging.split():
        if token.startswith("versionCode="):
            code = int(token.split("=")[1].strip("'"))
    assert code, f"no versionCode in badging: {badging[:200]}"
    sha = hashlib.sha256(data).hexdigest()
    print(f"v{version} code {code}: {len(data)} B sha256 {sha[:16]}…")

    # current index, if any
    entries = []
    st, cur = api("GET", f"/repos/{RELAY}/contents/collector/index.json")
    if st == 200:
        entries = json.loads(base64.b64decode(
            cur["content"]).decode("utf-8"))["versions"]
    entries = [e for e in entries if e["version"] != version]
    entries.append({
        "version": version,
        "code": code,
        "apk": f"v{version}/EchoCollector.apk",
        "sha256": sha,
        "notes": notes,
    })
    entries.sort(key=lambda e: e["code"])
    index = {"product": "EchoCollector", "latest": version,
             "versions": entries}

    st, ref = api("GET", f"/repos/{RELAY}/git/ref/heads/main")
    head = ref["object"]["sha"]
    st, commit = api("GET", f"/repos/{RELAY}/git/commits/{head}")
    tree_items = []
    for path, blob_data in (
            (f"collector/v{version}/EchoCollector.apk", data),
            ("collector/index.json",
             json.dumps(index, ensure_ascii=False, indent=1)
             .encode("utf-8"))):
        st, blob = api("POST", f"/repos/{RELAY}/git/blobs",
                       {"content": base64.b64encode(blob_data).decode(),
                        "encoding": "base64"})
        assert st == 201, f"blob {path}: {st} {blob}"
        tree_items.append({"path": path, "mode": "100644",
                           "type": "blob", "sha": blob["sha"]})
    st, tree = api("POST", f"/repos/{RELAY}/git/trees",
                   {"base_tree": commit["tree"]["sha"],
                    "tree": tree_items})
    st, nc = api("POST", f"/repos/{RELAY}/git/commits",
                 {"message": f"collector v{version}",
                  "tree": tree["sha"], "parents": [head]})
    api("PATCH", f"/repos/{RELAY}/git/refs/heads/main",
        {"sha": nc["sha"]})
    print("relay commit:", nc["sha"][:12])
    print("index:", f"https://redrighthand05.github.io/"
                    f"echo4d-update/collector/index.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
