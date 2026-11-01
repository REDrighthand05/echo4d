#!/usr/bin/env python3
"""True-scale stage regression locks.

Layers:
  1. the anchor catalogue is data with the fields the stage consumes,
     and v1 ships the 86-panel family only (a door leaf was never
     measured by the automatic chain -- Pioneer A -- so it must not
     sneak in unproven);
  2. rectangle discovery finds a drawn plate on a wall-like image and
     stays silent on a blank one;
  3. the decision logic: corroborated clusters at one scale win,
     absurd implied rooms are vetoed, single uncorroborated clusters
     do not set a scale;
  4. application: points/poses are multiplied exactly, the record is
     written, and a re-run never scales twice;
  5. degenerate jobs (no poses / no sfm anchors) pass through with
     status unknown and byte-identical points -- the stage never
     blocks a job.

Run: .venv/bin/python tests/test_scale.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from echo_scene import scale as scale_mod          # noqa: E402
from echo_scene.fuse import _write_points_bin      # noqa: E402
from echo_scene.pipeline import stage_scale        # noqa: E402

FAILURES = []


def _check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------- catalogue
print("== catalogue")
for entry in scale_mod.ANCHOR_CATALOG:
    _check(f"catalogue entry {entry.get('id')} complete",
           all(k in entry for k in ("id", "label", "kind", "unit_m",
                                    "max_gang", "agree_tol", "description"))
           and entry["unit_m"] > 0 and entry["max_gang"] >= 1)
ids = [e["id"] for e in scale_mod.ANCHOR_CATALOG]
_check("v1 ships the 86-panel family only", ids == ["cn86_panel"], str(ids))
panel = scale_mod.ANCHOR_CATALOG[0]
_check("86-panel unit is 86 mm", abs(panel["unit_m"] - 0.086) < 1e-9)

# ---------------------------------------------------------------- discovery
print("== rectangle discovery")
import cv2  # noqa: E402

img = np.full((900, 1200), 205, np.uint8)          # blank wall
cv2.rectangle(img, (420, 330), (780, 570), 90, -1)  # one dark plate
quads = scale_mod._detect_rectangles(img)
_check("drawn plate found",
       any(abs(q[1] - 360) < 26 and abs(q[2] - 240) < 26 for q in quads),
       f"{[(round(q[1]), round(q[2])) for q in quads]}")
blank = np.full((900, 1200), 205, np.uint8)
_check("blank wall stays silent", scale_mod._detect_rectangles(blank) == [])

# ---------------------------------------------------------------- decision
print("== scale decision")
REFS = (10.0, 10.5, 11.8, 73.0)  # height, plan a, plan b, route (units)


def _cluster(median, n_frames=3, n_samples=12, cen=(0.0, 0.0, 0.0)):
    return {"median": median,
            "samples": [median] * n_samples,
            "frames": {f"f{i}" for i in range(n_frames)},
            "centroid": np.array(cen), "gangs": [1]}


win = scale_mod._decide_scale(
    [_cluster(0.25, cen=(0, 0, 0)), _cluster(0.26, cen=(3, 0, 0))], REFS)
_check("corroborated group wins", win is not None
       and abs(win["scale"] - 0.255) < 0.01,
       f"{None if win is None else round(win['scale'], 3)}")
_check("absurd scale vetoed by the room gate",
       scale_mod._decide_scale(
           [_cluster(0.03, cen=(0, 0, 0)), _cluster(0.031, cen=(3, 0, 0))],
           REFS) is None)
_check("lone single-frame cluster cannot set scale",
       scale_mod._decide_scale([_cluster(0.25, n_frames=1)], REFS) is None)
_check("no clusters -> no scale",
       scale_mod._decide_scale([], REFS) is None)

# ---------------------------------------------------------------- application
print("== application + idempotency")
with tempfile.TemporaryDirectory() as td:
    job = Path(td)
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-1.0, 0.5, 2.0]])
    cols = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], np.uint8)
    _write_points_bin(pts, cols, job / "points.bin")
    poses = {"frame_00001.jpg": {
        "cam_from_world_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "cam_from_world_t": [2.0, -1.0, 0.5],
        "camera": {"model": "SIMPLE_RADIAL", "width": 10, "height": 10,
                   "params": [100.0, 5.0, 5.0, 0.0]}}}
    (job / "poses.json").write_text(json.dumps(poses))
    applied = scale_mod._apply_scale(job, 0.25)
    raw = np.fromfile(job / "points.bin", dtype=np.uint8).reshape(-1, 15)
    got = raw[:, :12].copy().view(np.float32).reshape(-1, 3).astype(np.float64)
    _check("points multiplied by the scale",
           np.allclose(got, pts * 0.25, atol=1e-6), str(applied))
    _check("colours untouched", (raw[:, 12:15] == cols).all())
    new_t = json.loads((job / "poses.json").read_text())[
        "frame_00001.jpg"]["cam_from_world_t"]
    _check("pose translations scaled", new_t == [0.5, -0.25, 0.125])

    # a written set+applied record makes the stage a no-op
    (job / "scale.json").write_text(json.dumps({
        "status": "set", "scale_to_meters": 0.25, "evidence": "估计",
        "error_pct": 10, "anchor": {"id": "cn86_panel", "label": "墙面开关面板"},
        "applied": True, "applied_to": applied}))
    before = (job / "points.bin").read_bytes()
    info = stage_scale(job)
    _check("re-run does not scale twice",
           (job / "points.bin").read_bytes() == before
           and info["scale_applied"] is True)

# ---------------------------------------------------------------- degenerate
print("== degenerate jobs pass through")
with tempfile.TemporaryDirectory() as td:
    job = Path(td)
    pts = np.random.default_rng(0).normal(size=(500, 3))
    cols = np.zeros((500, 3), np.uint8)
    _write_points_bin(pts, cols, job / "points.bin")
    before = (job / "points.bin").read_bytes()
    info = stage_scale(job)  # no poses.json, no sfm: must not raise
    rec = json.loads((job / "scale.json").read_text())
    _check("unknown scale recorded, job unblocked",
           info["ok"] is True and info["scale_applied"] is False
           and rec["status"] == "unknown" and rec["evidence"] == "未知")
    _check("points byte-identical", (job / "points.bin").read_bytes() == before)

print()
if FAILURES:
    print(f"FAILED: {FAILURES}")
    raise SystemExit(1)
print("ALL SCALE CHECKS PASSED")
