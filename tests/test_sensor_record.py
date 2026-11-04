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

Fixture binding (read before touching either side): the simulated
capture is fabricated FROM jobs/room2 by
experiments/sensor_sim/simulate_capture.py, so it only pairs with
the job folder as it stood at fabrication time.  room2 was
regenerated on 2026-11-04 (anchor scale 0.2761 applied to
points/poses, alignment rotation applied, point set re-trimmed:
the job is in metres now) while the committed capture still
encoded the old scene-unit geometry, and the payoff read
0.8593 m/unit -- a stale-fixture artefact, not an estimator error
(the unchanged estimator recovers a scale fabricated from the
current geometry exactly).  Re-run simulate_capture.py whenever
jobs/room2 is regenerated.  ANCHOR_SCALE below is the simulator's
fabrication constant (TRUE_SCALE there), chosen at the Pioneer
A/D anchor value of the original room2 reconstruction; it is the
scale of the SIMULATED world, not a claim about the units of the
current jobs/room2 folder.
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
ANCHOR_SCALE = 0.25  # Pioneer A/D consensus centre, m per scene unit;
# = TRUE_SCALE in simulate_capture.py (see fixture binding note above)


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


def test_collector_v02_streams():
    """Collector v0.2: poses.csv (with and without the tracking
    column) and frame_meta.csv parse as first-class optional streams;
    trajectory length honours the tracking filter; the exposure
    report summarises only rows the device actually reported."""
    with tempfile.TemporaryDirectory() as td:
        folder = Path(td) / "cap"
        sr.write_meta(folder, {"device": {"name": "test"}})
        sr.write_frames(folder, np.arange(3), np.array([0.0, 0.5, 1.0]),
                        ["frame_00000.jpg", "frame_00001.jpg",
                         "frame_00002.jpg"])
        t_imu = np.arange(0.0, 1.0, 0.1)
        sr.write_imu(folder, t_imu,
                     np.tile([1.0, 0.0, 0.0, 0.0], (len(t_imu), 1)),
                     np.tile([0.0, 9.80665, 0.0], (len(t_imu), 1)),
                     np.zeros((len(t_imu), 3)))
        sr.write_tof(folder, np.array([0.0]),
                     np.full((1, 64), 1000, dtype=np.int32))
        # straight 3 m walk, middle row LIMITED
        (folder / "poses.csv").write_text(
            "t_s,tx,ty,tz,qx,qy,qz,qw,tracking\n"
            "0.000000,0.000000,1.600000,0.000000,0.000000,0.000000,"
            "0.000000,1.000000,TRACKING\n"
            "0.500000,1.500000,1.600000,0.000000,0.000000,0.000000,"
            "0.000000,1.000000,LIMITED\n"
            "1.000000,3.000000,1.600000,0.000000,0.000000,0.000000,"
            "0.000000,1.000000,TRACKING\n")
        (folder / "frame_meta.csv").write_text(
            "frame,t_s,exposure_ns,iso,focus_m,ae_state,af_state\n"
            "1,0.000000,8000000,200,2.500000,2,4\n"
            "2,0.500000,20000000,400,2.500000,2,4\n"
            "3,1.000000,-1,-1,-1.000000,-1,-1\n")
        cap = sr.read_capture(folder)
        assert cap.traj_pos.shape == (3, 3)
        assert cap.trajectory_length_m(only_tracking=False) == 3.0
        assert cap.trajectory_length_m(only_tracking=True) == 0.0
        rep = cap.exposure_report()
        assert rep["frames"] == 2  # the -1 row is excluded
        assert rep["median_exposure_ms"] == 14.0
        assert rep["frac_slower_than_1_60"] == 0.5
        # legacy 8-column poses (collector v0.1) still parse
        (folder / "poses.csv").write_text(
            "t_s,tx,ty,tz,qx,qy,qz,qw\n"
            "0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,"
            "0.000000,1.000000\n"
            "1.000000,1.000000,0.000000,0.000000,0.000000,0.000000,"
            "0.000000,1.000000\n")
        cap = sr.read_capture(folder)
        assert cap.traj_tracking == ["UNKNOWN", "UNKNOWN"]
        assert cap.trajectory_length_m(only_tracking=False) == 1.0


def _write_min_capture(folder: Path, meta_extra: dict,
                       cfd: np.ndarray) -> None:
    meta = {"device": {"name": "test"},
            "calibration": {"cam_from_device": cfd.tolist()},
            "tof": {"fov_deg": 45.0}}
    meta.update(meta_extra)
    sr.write_meta(folder, meta)
    sr.write_frames(folder, np.arange(3), np.array([0.0, 0.5, 1.0]),
                    ["frame_00001.jpg", "frame_00002.jpg",
                     "frame_00003.jpg"])
    t_imu = np.arange(0.0, 1.05, 0.1)
    sr.write_imu(folder, t_imu,
                 np.tile([1.0, 0.0, 0.0, 0.0], (len(t_imu), 1)),
                 np.tile([0.0, 9.80665, 0.0], (len(t_imu), 1)),
                 np.zeros((len(t_imu), 3)))
    sr.write_tof(folder, np.array([0.0]),
                 np.full((1, 64), 1000, dtype=np.int32))


def test_stored_frame_convention():
    """The stored-frame roll, locked both ways (orientation audit,
    2026-11-04): with the collector's sensor_orientation_deg=90 the
    raw cam_from_device leaves measured gravity ~90 deg off the
    image-down axis of the posed (stored, landscape) frames, and
    Rz(-90) @ cam_from_device puts it on the axis. Without the
    android record (Pi / simulator) the calibration is untouched."""
    cfd = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    down_dev = np.array([0.0, -1.0, 0.0])   # device held upright
    img_down = np.array([0.0, 1.0, 0.0])
    with tempfile.TemporaryDirectory() as td:
        folder = Path(td) / "cap"
        _write_min_capture(folder, {"android": {
            "sensor_orientation_deg": 90}}, cfd)
        cap = sr.read_capture(folder)
        assert sr.stored_frame_roll_deg(cap) == -90.0
        raw_ang = _angle_deg(cfd @ down_dev, img_down)
        assert raw_ang > 80.0, f"raw convention should fail, got {raw_ang}"
        stored = sr.cam_from_device_stored(cap)
        fixed_ang = _angle_deg(stored @ down_dev, img_down)
        assert fixed_ang < 1.0, f"corrected convention off by {fixed_ang}"
        # the other sign is decisively wrong, not a wash
        a = np.radians(90.0)
        rz_plus = np.array([[np.cos(a), -np.sin(a), 0.0],
                            [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
        assert _angle_deg(rz_plus @ cfd @ down_dev, img_down) > 170.0

        plain = Path(td) / "cap2"
        _write_min_capture(plain, {}, cfd)
        cap2 = sr.read_capture(plain)
        assert sr.stored_frame_roll_deg(cap2) == 0.0
        assert np.array_equal(sr.cam_from_device_stored(cap2), cfd)


def test_import_capture_record():
    """Import landing: a capture's record files land in
    job/sensor_record (the video does NOT), read back through the
    unchanged reader; a non-capture folder lands nothing."""
    cfd = np.diag([1.0, -1.0, -1.0])
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = root / "capture_x"
        _write_min_capture(src, {}, cfd)
        (src / "video.mp4").write_bytes(b"fake")
        (src / "frame_meta.csv").write_text(
            "frame,t_s,exposure_ns,iso,focus_m,ae_state,af_state\n"
            "1,0.000000,8000000,200,2.500000,2,4\n")
        job = root / "job_x"
        job.mkdir()
        info = sr.import_capture_record(src, job)
        assert info is not None and info["has_imu"] is True
        landed = job / sr.JOB_SENSOR_DIRNAME
        for name in ("meta.json", "frames.csv", "imu.csv", "tof.csv",
                     "frame_meta.csv"):
            assert (landed / name).exists(), name
        assert not (landed / "video.mp4").exists()
        cap = sr.read_job_capture(job)
        assert cap is not None
        assert cap.frame_time["frame_00003.jpg"] == 1.0
        assert sr.read_job_capture(root / "no_such_job") is None

        notcap = root / "plain"
        notcap.mkdir()
        (notcap / "video.mp4").write_bytes(b"fake")
        job2 = root / "job_y"
        job2.mkdir()
        assert sr.import_capture_record(notcap, job2) is None
        assert not (job2 / sr.JOB_SENSOR_DIRNAME).exists()


def test_register_frame_times_override():
    """Production pairing: job poses are renumbered extractions, so
    registration must pair by the explicit time map, not by the
    capture's own frame names -- and with reaction-convention accel
    the estimator returns the device up axis itself."""
    from scipy.spatial.transform import Rotation
    cfd = np.diag([1.0, -1.0, -1.0])
    with tempfile.TemporaryDirectory() as td:
        folder = Path(td) / "cap"
        _write_min_capture(folder, {}, cfd)
        cap = sr.read_capture(folder)
        # recon world == boot world, device static: R_wc = cfd makes
        # R_wc @ cfd = I = R_wd, so R_align must come back identity.
        q = Rotation.from_matrix(cfd).as_quat()  # xyzw
        poses = {f"frame_0010{i}.jpg": {
            "cam_from_world_quat_xyzw": q.tolist(),
            "cam_from_world_t": [0.0, 0.0, 0.0]} for i in (1, 2, 3)}
        times = {"frame_00101.jpg": 0.0, "frame_00102.jpg": 0.5,
                 "frame_00103.jpg": 1.0}
        try:
            sr.register_imu_frame(cap, poses, cfd)
        except ValueError:
            pass
        else:
            raise AssertionError("name pairing should find no pairs")
        R_align = sr.register_imu_frame(cap, poses, cfd,
                                       frame_times=times)
        ang = float(np.degrees(np.arccos(np.clip(
            (np.trace(R_align) - 1.0) / 2.0, -1.0, 1.0))))
        assert ang < 0.01, f"R_align off identity by {ang} deg"
        up = sr.estimate_up_from_imu(cap, R_align)
        assert _angle_deg(up, [0.0, 1.0, 0.0]) < 0.01


if __name__ == "__main__":
    test_round_trip()
    print("round-trip: PASS")
    test_collector_v02_streams()
    print("collector v0.2 streams: PASS")
    test_zone_model_plane()
    print("zone model: PASS")
    test_device_program_contract()
    print("device program contract: PASS")
    test_stored_frame_convention()
    print("stored-frame convention: PASS")
    test_import_capture_record()
    print("import capture record: PASS")
    test_register_frame_times_override()
    print("register frame-times override: PASS")
    test_payoff_on_simulated_capture()
    print("payoff on simulated capture: PASS")
    print("sensor record contract: ALL PASS")
