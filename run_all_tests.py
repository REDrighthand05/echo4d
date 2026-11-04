#!/usr/bin/env python3
"""Unified test entry — one command runs every layer.

    .venv/bin/python run_all_tests.py [--smoke]

Layers (any red layer fails the run):
  parts   per-kernel unit tests        (echo_super/test_real_kernels.py,
                                          echo_super/test_ds_batch5.py)
  hotupdate  hot-update chain e2e      (tests/test_hot_update.py)
  routes  routing / skip / reroute     (tests/test_echo_super.py)
  frozen  locked features must not drift (tests/test_voxel_frozen.py)
  smoke   whole-machine video runs     (only with --smoke; slow, TODO hook)

Exit code 0 = all green.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SUITES = [
    ("parts", [sys.executable, "-m", "unittest", "echo_super.test_real_kernels", "-v"]),
    ("routes", [sys.executable, "-m", "unittest", "tests.test_echo_super", "-v"]),
    ("frozen", [sys.executable, "tests/test_voxel_frozen.py"]),
    ("align", [sys.executable, "tests/test_up_alignment.py"]),
    ("fuse", [sys.executable, "tests/test_voxel_pick_rule.py"]),
    ("poses", [sys.executable, "tests/test_pose_chain.py"]),
    ("clean", [sys.executable, "tests/test_continuity_clean.py"]),
    ("replay", [sys.executable, "tests/test_replay.py"]),
    ("replay", [sys.executable, "tests/test_replay_walk.py"]),
    ("scale", [sys.executable, "tests/test_scale.py"]),
    ("desktop", [sys.executable, "tests/test_desktop.py"]),
    ("webshell", [sys.executable, "tests/test_web_bridge.py"]),
    ("webshell", [sys.executable, "tests/test_appearance_settings.py"]),
    ("parts", [sys.executable, "echo_super/test_ds_batch5.py"]),
    ("hotupdate", [sys.executable, "tests/test_hot_update.py"]),
    ("sensor", [sys.executable, "tests/test_sensor_record.py"]),
    ("tethered", [sys.executable, "tests/test_tethered.py"]),
    ("paths", [sys.executable, "tests/test_paths.py"]),
    ("paths", [sys.executable, "tests/test_depth_progress.py"]),
    ("webshell", [sys.executable, "tests/test_win_controls.py"]),
    ("webshell", [sys.executable, "tests/test_window_drag.py"]),
    ("parts", [sys.executable, "tests/test_hwaccel.py"]),
    ("parts", [sys.executable, "tests/test_depth_batch.py"]),
    ("parts", [sys.executable, "tests/test_frame_feed.py"]),
    ("cli", [sys.executable, "tests/test_cli.py"]),
    ("cli", [sys.executable, "tests/test_cli_appctl.py"]),
    ("parts", [sys.executable, "tests/test_voxel_export.py"]),
    ("parts", [sys.executable, "tests/test_gaussian_adapter.py"]),
    ("parts", [sys.executable, "tests/test_photo_discovery.py"]),
    ("ledger", [sys.executable, "tests/test_versions_ledger.py"]),
    ("desktop", [sys.executable, "tests/test_version.py"]),
]


def main() -> int:
    smoke = "--smoke" in sys.argv
    results = []
    for layer, cmd in SUITES:
        print(f"\n=== [{layer}] {' '.join(cmd[1:])}", flush=True)
        r = subprocess.run(cmd, cwd=ROOT)
        results.append((layer, cmd[-2] if cmd[-1] == "-v" else cmd[-1], r.returncode == 0))
    if smoke:
        print("\n=== [smoke] not wired yet — whole-machine runs stay manual "
              "until the push starts")
    print("\n---- summary ----")
    ok = True
    for layer, name, passed in results:
        print(f"{layer:7s} {name:40s} {'PASS' if passed else 'FAIL'}")
        ok = ok and passed
    print("ALL GREEN" if ok else "SOME LAYER FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
