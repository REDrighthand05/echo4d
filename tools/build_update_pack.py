"""Build the hot-update pack between two releases.

    python tools/build_update_pack.py <old_manifest.json> <dist_dir> \
        <new_manifest.json> <out.zip>

Diffs the old release's manifest against the new dist's manifest
with the batch-5 kernels. A pack is produced ONLY when every changed
path lives under app/ (the loose payload): if a base file changed,
there is no pack and the release stays installer-only -- the script
says so and exits 3 so CI can tell the difference between "no pack
needed" (exit 0, empty diff) and "pack impossible" (exit 3).

Pack layout (what echo_app.hotupdate expects):
  plan.json             kernel plan + ordered steps + per-file hashes
  target-manifest.json  the new release's manifest
  files/<relpath>       the new content of every to_change file
"""
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echo_super.ds_batch5 import order_apply_steps, plan_update  # noqa: E402

PAYLOAD_PREFIX = "app/"


def main(argv) -> int:
    if len(argv) < 5:
        print(__doc__)
        return 2
    old = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    dist = Path(argv[2])
    new = json.loads(Path(argv[3]).read_text(encoding="utf-8"))
    out = Path(argv[4])
    plan = plan_update(old, new)
    changed = [c["path"] for c in plan["to_change"]] + plan["to_delete"]
    if not changed:
        print("pack: no changes, nothing to build")
        return 0
    outside = [p for p in changed if not p.startswith(PAYLOAD_PREFIX)]
    if outside:
        print(f"pack: IMPOSSIBLE, base files changed: {outside[:5]}")
        return 3
    full_plan = dict(plan)
    full_plan["steps"] = order_apply_steps(plan)
    full_plan["files"] = {c["path"]: {"sha256": c["sha256"],
                                      "size": c["size"]}
                          for c in plan["to_change"]}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("plan.json", json.dumps(full_plan, ensure_ascii=False,
                                           indent=1))
        z.writestr("target-manifest.json",
                   json.dumps(new, ensure_ascii=False, indent=1))
        for change in plan["to_change"]:
            z.write(dist / change["path"], f"files/{change['path']}")
    print(f"pack: {len(plan['to_change'])} changed, "
          f"{len(plan['to_delete'])} deleted, "
          f"{plan['total_bytes']} bytes -> {out} ({out.stat().st_size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
