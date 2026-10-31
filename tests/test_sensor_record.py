"""Sensor-record contract (handheld collector, 2026-10-31).

Locks three things:
  * the capture-folder format round-trips (writers -> reader) and the
    reader enforces the one-clock rule (strictly increasing t_s);
  * the Pi capture program (device/capture.py) compiles anywhere and
    its CSV headers match the format module's exactly;
  * the payoff derivations work on the committed SIMULATED capture of
    the room (experiments/sensor_sim/capture_room2_sim): true scale
    from the ToF stream lands on the Pioneer A/D anchor value, and
    level from the IMU stream lands on the pipeline's up estimate --
    against ground truth the importer never sees.
"""
import json
import py_compile
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from echo_scene import sensor_record as sr  # noqa: E402
from echo_scene.fuse import _estimate_world_up  # noqa: E402

SIM = ROOT / "experiments" / "sensor_sim"
CAPTURE = SIM / "capture_room2_sim"
ANCHOR_SCALE = 0.25  # Pioneer A/D consensus centre, m per scene unit


def _load_points(job: Path) -> np.ndarray:
    raw = np.fromfile(job / "points.bin", dtype=np.uint8)
    raw = raw[: raw.size // 15 * 15].reshape(-1, 15)
    return raw[:, :12].copy().view(np.float32).reshape(-1, 3).astype(np.float64)


def _angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(
        np.asarray(a) @ np.asarray(b), -1.0, 1.0))))


def test_round_trip():
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as td:
        folder = Path(td) / "cap"
        t_imu = np.arange(0.0, 2.0, 0.01)
        quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(t_imu), 1))
        accel = np.tile(np.array([0.0, 9.80665, 0.0]), (len(t_imu), 1))
        gyro = np.zeros((len(t_imu), 3))
        t_tof = np.arange(0.0, 2.0, 0.2)
        zones = rng.integers(100, 3000, size=(len(t_tof), 64)).astype(np.int32)
        zones[0, :5] = sr.TOF_INVALID
        sr.write_meta(folder, {
            "device": {"name": "test"}, "simulated": True,
            "calibration": {"cam_from_device": np.diag([1., -1., -1.]).tolist()},
            "tof": {"fov_deg": 45.0},
        })
        sr.write_frames(folder, np.arange(3), np.array([0.0, 0.5, 1.0]),
                        ["frame_00000.jpg", "frame_00001.jpg",
                         "frame_00002.jpg"])
        sr.write_imu(folder, t_imu, quats, accel, gyro)
        sr.write_tof(folder, t_tof, zones)
        cap = sr.read_capture(folder)
        assert cap.meta["device"]["name"] == "test"
        assert cap.frame_time["frame_00002.jpg"] == 1.0
        assert np.allclose(cap.imu_accel, accel, atol=1e-6)
        assert np.array_equal(cap.tof_mm, zones)
        assert np.allclose(cap.cam_from_device, np.diag([1., -1., -1.]))
        # non-monotonic timestamps must be rejected (one-clock rule)
        bad = np.array(t_imu)
        bad[10] = bad[9]
        sr.write_imu(folder, bad, quats, accel, gyro)
        try:
            sr.read_capture(folder)
        except ValueError:
            pass
        else:
            raise AssertionError("reader accepted non-monotonic imu t_s")


def test_zone_model_plane():
    # a fronto-parallel wall 2 units ahead: each zone must report the
    # SLANT range to the wall along its own centre ray (2*sqrt(1+u^2+v^2)
    # in tan space), not the z-depth -- that is what a ToF zone measures
    xs, ys = np.meshgrid(np.linspace(-1.5, 1.5, 241),
                         np.linspace(-1.5, 1.5, 241))
    pts = np.stack([xs.ravel(), ys.ravel(),
                    np.full(xs.size, 2.0)], axis=1)
    d = sr.zone_distances_from_points(pts, fov_deg=45.0).reshape(8, 8)
    half = np.tan(np.radians(45.0) / 2.0)
    expected = np.zeros((8, 8))
    for r in range(8):
        for c in range(8):
            u = ((c + 0.5) / 8 * 2 - 1) * half
            v = ((r + 0.5) / 8 * 2 - 1) * half
            expected[r, c] = 2.0 * np.sqrt(1 + u * u + v * v)
    assert np.isfinite(d).sum() >= 60
    assert np.allclose(d, expected, atol=0.06)


def test_device_program_contract():
    src = ROOT / "device" / "capture.py"
    py_compile.compile(str(src), doraise=True)
    text = src.read_text()
    for header in (sr._FRAMES_HEADER, sr._IMU_HEADER):
        for col in header:
            assert f'"{col}"' in text, f"capture.py missing column {col}"
    # tof columns are generated, not literal: lock the generator itself
    assert '"t_s"' in text and 'z{r * 8 + c}' in text
    assert sr._TOF_HEADER == ["t_s"] + [f"z{r * 8 + c}" for r in range(8)
                                        for c in range(8)]


def test_payoff_on_simulated_capture():
    assert CAPTURE.exists(), "simulated capture missing — run " \
        "experiments/sensor_sim/simulate_capture.py"
    cap = sr.read_capture(CAPTURE)
    assert cap.meta["simulated"] is True
    poses = json.loads((ROOT / "jobs" / "room2" / "poses.json").read_text())
    points = _load_points(ROOT / "jobs" / "room2")
    gt = json.loads((SIM / "ground_truth.json").read_text())

    scale = sr.estimate_scale_from_tof(cap, poses, points, tof_stride=8)
    dev = abs(scale["scale_m_per_unit"] / ANCHOR_SCALE - 1.0)
    assert dev < 0.03, f"scale {scale['scale_m_per_unit']:.4f} off by {dev:.1%}"
    assert scale["pairs"] > 1000

    R_align = sr.register_imu_frame(cap, poses, cap.cam_from_device)
    up_imu = sr.estimate_up_from_imu(cap, R_align)
    ang = _angle_deg(up_imu, _estimate_world_up(poses))
    assert ang < 1.0, f"IMU up off pipeline estimate by {ang:.2f} deg"
    E = R_align @ np.array(gt["boot_rotation"])
    reg_err = float(np.degrees(np.arccos(np.clip(
        (np.trace(E) - 1.0) / 2.0, -1.0, 1.0))))
    assert reg_err < 3.0, f"frame registration off by {reg_err:.2f} deg"
    print(f"  payoff: scale {scale['scale_m_per_unit']:.4f} m/unit "
          f"({scale['pairs']} pairs), up error {ang:.2f} deg, "
          f"registration error {reg_err:.2f} deg")


if __name__ == "__main__":
    test_round_trip()
    print("round-trip: PASS")
    test_zone_model_plane()
    print("zone model: PASS")
    test_device_program_contract()
    print("device program contract: PASS")
    test_payoff_on_simulated_capture()
    print("payoff on simulated capture: PASS")
    print("sensor record contract: ALL PASS")
