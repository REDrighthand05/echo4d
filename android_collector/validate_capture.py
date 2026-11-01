#!/usr/bin/env python3
"""Validate an Android-collector capture folder against the format
contract: the unchanged Python reader (echo_scene/sensor_record.py)
must accept it, and the payoff importer must reproduce the simulated
references when the folder is the harness output.

    .venv/bin/python android_collector/validate_capture.py <folder> \
        [--job jobs/room2] [--sim experiments/sensor_sim/capture_room2_sim]

Checks:
  1. sr.read_capture parses the folder (headers, strictly increasing
     clocks, format_version) — the contract itself.
  2. If --sim is given: stream-by-stream losslessness against the sim
     capture (frame times, IMU rows, ToF zones must match to float
     formatting precision).
  3. Payoff: TRUE SCALE from ToF vs the Pioneer A/D anchor consensus
     (0.25 m/unit, honest +/-5% band) and LEVEL from IMU vs the
     pipeline's up estimate for the job (expect ~0.13 deg, the sim
     proof's number; pass bar 1.0 deg).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echo_scene import sensor_record as sr  # noqa: E402
from echo_scene.fuse import _estimate_world_up  # noqa: E402

ANCHOR_SCALE = 0.25
ANCHOR_ERR = 0.05


def load_points(job: Path) -> np.ndarray:
    raw = np.fromfile(job / "points.bin", dtype=np.uint8)
    raw = raw[: raw.size // 15 * 15].reshape(-1, 15)
    return raw[:, :12].copy().view(np.float32).reshape(-1, 3).astype(np.float64)


def angle_deg(a, b) -> float:
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--job", default=str(ROOT / "jobs" / "room2"))
    ap.add_argument("--sim", default=None)
    args = ap.parse_args()
    folder = Path(args.folder)
    ok = True

    cap = sr.read_capture(folder)
    print(f"[reader] OK: {len(cap.frame_t)} frames, {len(cap.imu_t)} imu, "
          f"{len(cap.tof_t)} tof rows; format_version "
          f"{cap.meta.get('format_version')}")

    if args.sim:
        sim = sr.read_capture(Path(args.sim))
        assert len(cap.frame_t) == len(sim.frame_t), "frame count differs"
        d_frame = float(np.max(np.abs(cap.frame_t - sim.frame_t)))
        d_imu = float(np.max(np.abs(cap.imu_quat - sim.imu_quat)))
        d_imu_t = float(np.max(np.abs(cap.imu_t - sim.imu_t)))
        tof_same = bool(np.array_equal(cap.tof_mm, sim.tof_mm))
        print(f"[lossless] frame_t max dev {d_frame:.2e}s, imu_t {d_imu_t:.2e}s, "
              f"imu_quat {d_imu:.2e}, tof identical: {tof_same}")
        ok &= d_frame < 1e-6 and d_imu_t < 1e-6 and d_imu < 1e-5 and tof_same

    job = Path(args.job)
    poses = json.loads((job / "poses.json").read_text())
    points = load_points(job)
    scale = sr.estimate_scale_from_tof(cap, poses, points)
    s = scale["scale_m_per_unit"]
    dev = (s / ANCHOR_SCALE - 1.0) * 100.0
    in_band = abs(dev) <= ANCHOR_ERR * 100.0
    print(f"[scale] {s:.4f} m/unit vs anchor {ANCHOR_SCALE} "
          f"({dev:+.2f}%, {'inside' if in_band else 'OUTSIDE'} the +/-5% band; "
          f"pairs {scale['pairs']})")
    ok &= in_band

    R_align = sr.register_imu_frame(cap, poses, cap.cam_from_device)
    up_imu = sr.estimate_up_from_imu(cap, R_align)
    ang = angle_deg(up_imu, _estimate_world_up(poses))
    print(f"[level] IMU up vs pipeline estimate: {ang:.2f} deg "
          f"(pass bar 1.0 deg)")
    ok &= ang < 1.0

    print("VALID" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
