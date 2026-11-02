"""Benches for DS batch 5 hot-update kernels (hand fixtures)."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echo_super.ds_batch5 import (manifest_digest, order_apply_steps,
                                  plan_update, verify_listing)

FAIL = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAIL.append(name)


def H(ch):
    return ch * 64


def M(files, version="0.4.0"):
    return {"version": version, "files": files}


def E(sha, size):
    return {"sha256": sha, "size": size}


# digest: independent recomputation + key-order independence
m = M({"b.txt": E(H("b"), 2), "a.txt": E(H("a"), 1)})
canon = json.dumps({"version": "0.4.0", "files": m["files"]},
                   sort_keys=True, separators=(",", ":"), ensure_ascii=False)
check("digest matches independent sha256",
      manifest_digest(m) == hashlib.sha256(canon.encode()).hexdigest())
m2 = {"files": {"a.txt": E(H("a"), 1), "b.txt": E(H("b"), 2)}, "version": "0.4.0"}
check("digest independent of key order", manifest_digest(m) == manifest_digest(m2))

# plan_update: change + add + delete + unchanged
old = M({"a": E(H("a"), 1), "b": E(H("b"), 2), "c": E(H("c"), 3)}, "0.4.0")
new = M({"a": E(H("a"), 1), "b": E(H("d"), 5), "d": E(H("e"), 7)}, "0.5.0")
p = plan_update(old, new)
check("plan versions", p["from_version"] == "0.4.0" and p["to_version"] == "0.5.0")
check("plan to_change sorted+exact",
      [c["path"] for c in p["to_change"]] == ["b", "d"])
check("plan to_delete", p["to_delete"] == ["c"])
check("plan unchanged", p["unchanged"] == 1)
check("plan total_bytes", p["total_bytes"] == 12)
p0 = plan_update(old, old)
check("plan no-op", p0["to_change"] == [] and p0["to_delete"] == []
      and p0["unchanged"] == 3 and p0["total_bytes"] == 0)

# verify_listing: missing / mismatched / extra semantics
man = M({"a": E(H("a"), 1), "b": E(H("b"), 2), "c": E(H("c"), 3)})
r = verify_listing(man, {"a": E(H("a"), 1), "b": E(H("f"), 2), "z": E(H("0"), 9)})
check("verify missing", r["missing"] == ["c"])
check("verify mismatched", r["mismatched"] == ["b"])
check("verify extra", r["extra"] == ["z"])
check("verify not ok", r["ok"] is False)
r2 = verify_listing(man, {"a": E(H("a"), 1), "b": E(H("b"), 2),
                          "c": E(H("c"), 3), "user.txt": E(H("1"), 4)})
check("verify ok with extra only", r2["ok"] is True and r2["extra"] == ["user.txt"])

# order_apply_steps: version stamp last among replaces, deletes after
plan = {"to_change": [{"path": "installer/version.inc"}, {"path": "b.py"},
                      {"path": "a.py"}],
        "to_delete": ["old.dat", "a_old.dat"]}
steps = order_apply_steps(plan)
seq = [(s["op"], s["path"]) for s in steps]
check("steps order", seq == [("replace", "a.py"), ("replace", "b.py"),
                             ("replace", "installer/version.inc"),
                             ("delete", "a_old.dat"), ("delete", "old.dat")])

# validation: unsafe paths and bad shapes raise ValueError
for bad in (lambda: plan_update(M({"../evil": E(H("a"), 1)}), new),
            lambda: plan_update(old, M({"/abs": E(H("a"), 1)})),
            lambda: verify_listing(M({"a": {"sha256": "zz", "size": 1}}), {}),
            lambda: order_apply_steps({"to_change": [{"path": "a\\b"}], "to_delete": []}),
            lambda: manifest_digest({"version": "1"})):
    try:
        bad()
        check("validation raises", False)
    except ValueError:
        check("validation raises", True)

print("FAILURES:", len(FAIL))
sys.exit(1 if FAIL else 0)
