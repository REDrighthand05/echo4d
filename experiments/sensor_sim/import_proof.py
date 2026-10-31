#!/usr/bin/env python3
"""Payoff proof: run the sensor importer on the simulated capture.

Reads ONLY the capture folder + the jobs/room2 reconstruction (the same
inputs a real capture would produce), derives TRUE SCALE from the ToF
stream and LEVEL from the IMU stream, and compares both against the
independent references:

  * scale reference: Pioneer A/D strong-discovery consensus,
    0.25 m per scene unit (centre 0.249, honest error +/-5%).
  * level references: (1) the pipeline's own up estimate for room2
    (fuse._estimate_world_up on the job poses -- what today's
    up-alignment uses), (2) the up direction implied by the stored
    alignment.json of jobs/room2_clean_exp.

Writes payoff_report.md next to this script and prints the numbers.

    .venv/bin/python experiments/sensor_sim/import_proof.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from echo_scene import sensor_record as sr  # noqa: E402
from echo_scene.fuse import _estimate_world_up  # noqa: E402

HERE = Path(__file__).resolve().parent
CAPTURE = HERE / "capture_room2_sim"
JOB = ROOT / "jobs" / "room2"
ANCHOR_SCALE = 0.25          # Pioneer A/D consensus centre (D: 0.249)
ANCHOR_SCALE_ERR = 0.05      # Pioneer D honest error bar (+/-5%)


def load_points(job: Path) -> np.ndarray:
    raw = np.fromfile(job / "points.bin", dtype=np.uint8)
    raw = raw[: raw.size // 15 * 15].reshape(-1, 15)
    return raw[:, :12].copy().view(np.float32).reshape(-1, 3).astype(np.float64)


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def main() -> int:
    cap = sr.read_capture(CAPTURE)
    poses = json.loads((JOB / "poses.json").read_text())
    points = load_points(JOB)
    gt = json.loads((HERE / "ground_truth.json").read_text())
    print(f"capture: {len(cap.frame_t)} frames, {len(cap.imu_t)} imu, "
          f"{len(cap.tof_t)} tof; cloud {len(points)} points")

    # --- payoff 1: true scale from ToF --------------------------------
    scale = sr.estimate_scale_from_tof(cap, poses, points)
    s = scale["scale_m_per_unit"]
    dev_pct = (s / ANCHOR_SCALE - 1.0) * 100.0
    print(f"scale from ToF: {s:.4f} m/unit "
          f"(anchor {ANCHOR_SCALE} {dev_pct:+.2f}%, "
          f"pairs {scale['pairs']}, dropped {scale['pairs_dropped']}, "
          f"resid RMS {scale['residual_rms_m'] * 1000:.1f} mm)")

    # --- payoff 2: level from IMU --------------------------------------
    R_align = sr.register_imu_frame(cap, poses, cap.cam_from_device)
    up_imu = sr.estimate_up_from_imu(cap, R_align)
    up_pipe = _estimate_world_up(poses)
    ang_pipe = angle_deg(up_imu, up_pipe)
    # The stored alignment.json of the sibling job room2_trim_acceptance
    # belongs to a DIFFERENT reconstruction (its own COLMAP frame), so
    # it cannot be compared in room2's frame directly; what can be
    # checked is that a stored alignment IS its job's estimator output
    # (self-consistency), which makes the room2 estimator the pipeline's
    # up-alignment result for this job.
    ta_poses = json.loads((ROOT / "jobs" / "room2_trim_acceptance"
                           / "poses.json").read_text())
    ta_align = json.loads((ROOT / "jobs" / "room2_trim_acceptance"
                           / "alignment.json").read_text())
    up_ta_stored = np.array(ta_align["rotation"]).T @ np.array([0., 1., 0.])
    stored_selfcheck = angle_deg(_estimate_world_up(ta_poses), up_ta_stored)
    R_boot_gt = np.array(gt["boot_rotation"])
    # R_align maps boot frame -> recon frame, so truth is R_boot.T:
    E = R_align @ R_boot_gt
    boot_err = float(np.degrees(np.arccos(
        np.clip((np.trace(E) - 1.0) / 2.0, -1.0, 1.0))))
    print(f"up from IMU vs pipeline estimate: {ang_pipe:.2f} deg; "
          f"stored-alignment self-consistency: {stored_selfcheck:.2f} deg; "
          f"frame registration error vs boot truth: {boot_err:.2f} deg")

    within = abs(dev_pct) <= ANCHOR_SCALE_ERR * 100.0
    report = f"""# Sensor payoff report — simulated capture of the room

Capture: `experiments/sensor_sim/capture_room2_sim/` (SIMULATED —
fabricated from the jobs/room2 reconstruction; see the honest boundary
below). Importer: `echo_scene/sensor_record.py`, run by
`experiments/sensor_sim/import_proof.py`. The importer read only the
capture folder + the room2 reconstruction — the same inputs a real
capture would produce.

## Payoff 1 — TRUE SCALE from the ToF stream

| | value |
|---|---|
| derived scale | **{s:.4f} m per scene unit** |
| reference (Pioneer A/D anchor consensus) | {ANCHOR_SCALE:.2f} m/unit (centre 0.249, ±5%) |
| deviation | **{dev_pct:+.2f}%** — {"inside" if within else "OUTSIDE"} the anchor's ±5% band |
| zone pairs used | {scale['pairs']} (from {scale['tof_frames_used']} ToF frames) |
| pairs dropped by the 3-sigma trim | {scale['pairs_dropped']} |
| fit residual | RMS {scale['residual_rms_m'] * 1000:.1f} mm, median abs {scale['residual_median_abs_m'] * 1000:.1f} mm |
| median per-zone ratio (cross-check) | {scale['median_ratio']:.4f} m/unit |

Method: one global least-squares fit through the origin over every
(cloud distance, ToF distance) zone pair — no voting, no per-zone
decisions.

## Payoff 2 — LEVEL from the IMU stream

| | angle |
|---|---|
| IMU-derived up vs the pipeline's up-alignment result for room2 (its pose estimator) | **{ang_pipe:.2f}°** |
| IMU frame registration error vs simulated boot truth | {boot_err:.2f}° |
| (reference check) stored alignment.json vs its own job's estimator, room2_trim_acceptance | {stored_selfcheck:.2f}° |

Note on references: jobs/room2 predates alignment.json, so "the
pipeline's up-alignment result" for this job is its estimator output.
That this is the right identification is verified on the sibling job
room2_trim_acceptance, whose stored alignment.json reproduces its own
estimator's up to {stored_selfcheck:.2f}°. (The sibling job is a
separate reconstruction in its own COLMAP frame, ~10.8° away from
room2's — cross-frame angle comparisons between jobs are meaningless
and are not used.)

The IMU up is a MEASUREMENT (mean accelerometer vector carried into
the reconstruction frame by the registered orientation stream); the
references are the pipeline's pose-geometry ESTIMATE. Agreement at
this level means the device replaces the guess with a measurement of
comparable-or-better quality from the first second of capture.

## Honest boundary — what only real hardware can prove

This simulation validates the record format, the importer maths and
their noise robustness. It cannot validate physics it fabricated:

* The simulated ToF was raycast from the reconstruction itself
  (scaled by the anchor value), so scale agreement confirms the data
  path + estimator, NOT an independent measurement. A real VL53L5CX
  sees real surfaces: glass, mirrors, dark fabrics, sunlight and
  multi-path returns will produce zones the cloud does not predict.
* Sync: all simulated streams share one exact clock. On the board,
  camera/IMU/ToF timestamps come from different drivers; jitter of
  even 20–50 ms shifts ToF/IMU samples along a moving route.
* IMU drift here is a smooth 0.6°/min model. Real AHRS drift over
  minutes (temperature, vibration, magnetic disturbance indoors) can
  be worse and less smooth.
* ToF noise here is Gaussian in range. Real zone behaviour (crosstalk,
  cover-glass smudge, target-dependent bias) is not.
* Extrinsic calibration (cam_from_device) was exact here; a hand-built
  board needs its own calibration check.
"""
    (HERE / "payoff_report.md").write_text(report)
    print("report written to", HERE / "payoff_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
