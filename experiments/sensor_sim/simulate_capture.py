#!/usr/bin/env python3
"""Fabricate a simulated ECHO Collector capture of the user's room.

Ground truth is the jobs/room2 reconstruction (230 poses + fused cloud,
COLMAP world frame). From it we synthesise what the handheld device
WOULD have recorded on that walk:

  * frames.csv -- the 320 extracted frames spread over the real video
    duration (159.496 s) on one monotonic clock.
  * imu.csv (100 Hz) -- orientations by slerping the route's
    world-from-device rotations into a random boot frame, with slow
    AHRS drift added to the REPORTED orientations; accelerometer =
    true gravity (direction = the pipeline's own up estimate for
    room2) + walking bob/sway + constant bias + noise; gyro from the
    true angular velocity + noise/bias.
  * tof.csv (5 Hz) -- 8x8 zones raycast from interpolated route poses
    against the fused cloud (sensor_record.zone_distances_from_points),
    converted to metres with the TRUE scale (0.25 m/unit -- used ONLY
    here, to fabricate measurements; the capture folder never contains
    it), then VL53L5CX-class noise, 1 mm quantisation, a 4 m range cap
    and random dropouts.

Ground truth values (true scale, true up, boot rotation, noise model)
go to ground_truth.json NEXT TO the capture folder, for the payoff
report and tests -- never inside it. The importer must derive
everything from the capture + reconstruction alone.

    .venv/bin/python experiments/sensor_sim/simulate_capture.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from echo_scene import sensor_record as sr  # noqa: E402
from echo_scene.fuse import _estimate_world_up, _quat_xyzw_to_matrix  # noqa: E402

JOB = ROOT / "jobs" / "room2"
OUT = Path(__file__).resolve().parent / "capture_room2_sim"
GT_PATH = Path(__file__).resolve().parent / "ground_truth.json"

TRUE_SCALE = 0.25            # m per scene unit (Pioneer A/D anchor value)
DURATION = 159.496           # s, the real input.mp4 duration
IMU_HZ = 100.0
TOF_HZ = 5.0
TOF_FOV_DEG = 45.0
TOF_MAX_RANGE_M = 4.0
GRAVITY = 9.80665
SEED = 20261031

# Device extrinsic: board frame (x right, y up, z back) -> camera frame
# (OpenCV: x right, y down, z forward): 180 deg about x.
R_CAM_DEV = np.diag([1.0, -1.0, -1.0])


def load_points(job: Path) -> np.ndarray:
    raw = np.fromfile(job / "points.bin", dtype=np.uint8)
    raw = raw[: raw.size // 15 * 15].reshape(-1, 15)
    return raw[:, :12].copy().view(np.float32).reshape(-1, 3).astype(np.float64)


def main() -> int:
    rng = np.random.default_rng(SEED)
    poses = json.loads((JOB / "poses.json").read_text())
    points = load_points(JOB)
    up_true = _estimate_world_up(poses)
    assert up_true is not None
    print(f"ground-truth up (COLMAP frame): {np.round(up_true, 4)}")

    # --- frames on the shared clock -------------------------------------
    frame_files = sorted(p.name for p in (JOB / "frames").glob("*.jpg"))
    frame_t = np.linspace(0.0, DURATION, len(frame_files))
    frame_time = dict(zip(frame_files, frame_t))

    # --- route keyframes -------------------------------------------------
    names = sorted(n for n in poses if n in frame_time)
    key_t = np.array([frame_time[n] for n in names])
    centers, R_wd_key = [], []
    for n in names:
        pose = poses[n]
        R_cw = _quat_xyzw_to_matrix(pose["cam_from_world_quat_xyzw"])
        centers.append(-R_cw.T @ np.array(pose["cam_from_world_t"]))
        R_wd_key.append(R_cw.T @ R_CAM_DEV)  # world-from-device
    centers = np.array(centers)
    slerp_wd = Slerp(key_t, Rotation.from_matrix(np.stack(R_wd_key)))
    # world-from-camera slerp, directly from the poses:
    R_wc_key = []
    for n in names:
        R_cw = _quat_xyzw_to_matrix(poses[n]["cam_from_world_quat_xyzw"])
        R_wc_key.append(R_cw.T)
    slerp_wc = Slerp(key_t, Rotation.from_matrix(np.stack(R_wc_key)))

    # --- IMU --------------------------------------------------------------
    boot = Rotation.from_matrix(
        Rotation.random(random_state=rng).as_matrix())  # boot frame rotation
    R_boot = boot.as_matrix()
    drift_axis = rng.normal(size=3)
    drift_axis /= np.linalg.norm(drift_axis)
    drift_rate_deg_min = 0.6

    imu_t = np.arange(0.0, DURATION + 1e-9, 1.0 / IMU_HZ)

    def slerp_clamped(slerp, t):
        # poses cover frames 16..320; outside that span the device is
        # held at the end orientations (constant extrapolation)
        return slerp(np.clip(t, key_t[0], key_t[-1]))

    R_wd_true = slerp_clamped(slerp_wd, imu_t).as_matrix()
    drift_ang = np.radians(drift_rate_deg_min * (imu_t / 60.0)
                           + 0.35 * np.sin(2 * np.pi * imu_t / 47.0))
    drift = Rotation.from_rotvec(drift_ang[:, None] * drift_axis[None, :])
    R_reported = np.einsum("ij,njk->nik", R_boot,
                           drift.as_matrix() @ R_wd_true)
    quat_wxyz = Rotation.from_matrix(R_reported).as_quat()[:, [3, 0, 1, 2]]

    g_world = -GRAVITY * up_true
    horiz = np.array([1.0, 0.0, 0.0])
    horiz = horiz - (horiz @ up_true) * up_true
    horiz /= np.linalg.norm(horiz)
    motion = (0.35 * np.sin(2 * np.pi * 1.9 * imu_t))[:, None] * up_true[None, :] \
        + (0.15 * np.sin(2 * np.pi * 0.95 * imu_t + 1.3))[:, None] * horiz[None, :]
    accel_world = g_world[None, :] + motion
    accel_dev = np.einsum("nji,nj->ni", R_wd_true, accel_world)
    accel_dev += np.array([0.03, -0.02, 0.015])[None, :]
    accel_dev += rng.normal(0.0, 0.04, accel_dev.shape)

    dt = 1.0 / IMU_HZ
    rel = slerp_clamped(slerp_wd, imu_t + dt) * slerp_clamped(slerp_wd, imu_t).inv()
    omega_world = rel.as_rotvec() / dt
    gyro_dev = np.einsum("nji,nj->ni", R_wd_true, omega_world)
    gyro_dev += 0.0015 + rng.normal(0.0, 0.004, gyro_dev.shape)

    # --- ToF ----------------------------------------------------------------
    tof_t = np.arange(0.0, DURATION + 1e-9, 1.0 / TOF_HZ)
    centers_tof = np.stack([np.interp(tof_t, key_t, centers[:, a])
                            for a in range(3)], axis=1)
    R_wc_tof = slerp_clamped(slerp_wc, tof_t).as_matrix()
    zones_mm = np.full((len(tof_t), 64), sr.TOF_INVALID, dtype=np.int32)
    for i, (C, R_wc) in enumerate(zip(centers_tof, R_wc_tof)):
        # row-vector form: p_cam = (p_world - C) @ R_wc  (since
        # p_cam = R_cw @ p_world with R_cw = R_wc.T in column form)
        scene = sr.zone_distances_from_points(
            (points - C) @ R_wc, fov_deg=TOF_FOV_DEG)
        metres = scene * TRUE_SCALE
        valid = np.isfinite(metres) & (metres <= TOF_MAX_RANGE_M)
        noisy = metres + rng.normal(0.0, 0.005 + 0.015 * np.nan_to_num(metres))
        mm = np.round(noisy * 1000.0)
        keep = valid & (rng.random(64) > 0.025)
        far = valid & (metres > 3.5)
        keep &= ~(far & (rng.random(64) < 0.20))
        zones_mm[i, keep] = mm[keep].astype(np.int32)
        if i % 200 == 0:
            print(f"  tof {i}/{len(tof_t)}", flush=True)
    print(f"tof valid fraction: {(zones_mm > 0).mean():.2%}")

    # --- write the capture --------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    sr.write_frames(OUT, np.arange(len(frame_files)), frame_t, frame_files)
    sr.write_imu(OUT, imu_t, quat_wxyz, accel_dev, gyro_dev)
    sr.write_tof(OUT, tof_t, zones_mm)
    cam0 = poses[names[0]]["camera"]
    sr.write_meta(OUT, {
        "device": {"name": "ECHO Collector", "board": "simulated",
                   "firmware": "sim-1"},
        "simulated": True,
        "provenance": ("Fabricated from jobs/room2 reconstruction (poses + "
                       "fused cloud) by experiments/sensor_sim/"
                       "simulate_capture.py. No video.mp4: the job's "
                       "extracted frames are the stand-in pictures."),
        "clock": {"type": "monotonic", "unit": "s",
                  "origin_note": "t=0 is capture start on the device "
                                 "monotonic clock"},
        "duration_s": DURATION,
        "streams": {
            "video": {"fps_nominal": len(frame_files) / DURATION,
                      "frames_csv": "frames.csv"},
            "imu": {"hz": IMU_HZ, "quat_order": "wxyz",
                    "accel_unit": "m/s^2 (device frame, incl. gravity)",
                    "gyro_unit": "rad/s (device frame)"},
            "tof": {"hz": TOF_HZ, "grid": [8, 8], "fov_deg": TOF_FOV_DEG,
                    "max_range_mm": int(TOF_MAX_RANGE_M * 1000),
                    "invalid": sr.TOF_INVALID, "unit": "mm"},
        },
        "calibration": {
            "cam_from_device": R_CAM_DEV.tolist(),
            "camera": {"model": str(cam0["model"]), "width": cam0["width"],
                       "height": cam0["height"], "params": cam0["params"]},
        },
    })

    GT_PATH.write_text(json.dumps({
        "true_scale_m_per_unit": TRUE_SCALE,
        "up_true_colmap": up_true.tolist(),
        "boot_rotation": R_boot.tolist(),
        "drift": {"axis": drift_axis.tolist(),
                  "rate_deg_per_min": drift_rate_deg_min},
        "noise": {"accel_sigma": 0.04, "gyro_sigma": 0.004,
                  "tof_sigma_m": "0.005 + 0.015 * range",
                  "tof_dropout": 0.025},
        "source_job": "jobs/room2",
    }, indent=2))
    print(f"capture written to {OUT}")
    print(f"ground truth written to {GT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
