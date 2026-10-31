#!/usr/bin/env python3
"""Continuity-cleanup regression locks.

Two layers:
  1. synthetic: two separated bodies -> only the larger survives, and
     the result is deterministic across repeated runs;
  2. room regression (when the local acceptance data is present): the
     production stage on the room job's fused cloud must reproduce the
     validated experiment exactly -- same chosen cell size, same kept
     count (208,666 -> 183,691 at strictness 0.80), byte-identical
     output to experiments/continuity_clean.py's points_q80.bin.

Run: .venv/bin/python tests/test_continuity_clean.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from echo_scene.clean import clean_points, stage_clean  # noqa: E402

FAILURES = []


def _check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        FAILURES.append(name)


def synthetic():
    print("[synthetic two-body cloud]")
    rng = np.random.default_rng(7)
    # main body: a dense 12x12x12 lattice (1,728 points, spacing 0.1)
    g = np.stack(np.meshgrid(*[np.arange(12) * 0.1] * 3), axis=-1).reshape(-1, 3)
    # stray fragment: 9 points far away
    frag = rng.normal(loc=[30.0, 30.0, 30.0], scale=0.05, size=(9, 3))
    pts = np.concatenate([g, frag])
    keep, stats = clean_points(pts, 0.80)
    _check("keeps the whole main body", bool(keep[: len(g)].all()),
           f"kept {int(keep[:len(g)].sum())}/{len(g)}")
    _check("drops the whole fragment", not bool(keep[len(g):].any()),
           f"dropped {int((~keep[len(g):]).sum())}/9")
    keep2, _ = clean_points(pts, 0.80)
    _check("deterministic", bool((keep == keep2).all()))


def room_regression():
    src = ROOT / "jobs" / "room2_trim_acceptance" / "points.bin"
    ref = ROOT / "jobs" / "room2_clean_exp" / "points_q80.bin"
    if not src.exists():
        print("[room regression] acceptance data absent -- skipped")
        return
    print("[room regression: production stage vs validated experiment]")
    with tempfile.TemporaryDirectory() as td:
        job = Path(td)
        shutil.copyfile(src, job / "points.bin")
        info = stage_clean(job)
        _check("stage ok", info.get("ok") is True, str(info.get("reason", "")))
        _check("points_before 208,666", info.get("points_before") == 208666,
               f"got {info.get('points_before')}")
        _check("points_after 183,691", info.get("points_after") == 183691,
               f"got {info.get('points_after')}")
        _check("cell size matches experiment",
               info.get("cell_size") == 0.05415087325639512,
               f"got {info.get('cell_size')}")
        _check("rewritten points.bin size",
               (job / "points.bin").stat().st_size == 183691 * 15)
        if ref.exists():
            _check("byte-identical to experiment points_q80.bin",
                   (job / "points.bin").read_bytes() == ref.read_bytes())


def main() -> int:
    synthetic()
    room_regression()
    print("CONTINUITY TESTS:", "ALL PASS" if not FAILURES else f"FAILED {FAILURES}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
