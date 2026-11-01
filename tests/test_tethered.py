"""Tethered collector contract (ESP32-S3 head + PC recorder).

Locks three things:
  * the ECHO-T1 wire protocol (device_tethered/protocol.py): frames
    round-trip, the microsecond tick unwraps across its 2**32 wrap,
    and the parser resynchronises after garbage, partial feeds and
    CRC failures;
  * the recorder's toggle path (head button frame -> record -> head
    button frame -> a valid capture folder on disk);
  * the hardware-free LOOPBACK: the committed simulated capture
    re-emitted as protocol bytes + a synthetic camera feed, recorded
    by device_tethered, passes the UNCHANGED record-format reader
    and reproduces the UNCHANGED importer payoff (scale from ToF on
    the Pioneer A/D anchor, level from IMU on the pipeline estimate).
"""
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from echo_scene import sensor_record as sr  # noqa: E402
from echo_scene.fuse import _estimate_world_up  # noqa: E402
from device_tethered import protocol, simulator  # noqa: E402
from device_tethered.recorder import RecorderApp, cv2  # noqa: E402

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


def test_protocol_roundtrip():
    p = protocol.ProtocolParser()
    stream = b"".join([
        protocol.encode_hello(1000),
        protocol.encode_imu(2000, [1.0, 0.0, 0.0, 0.0],
                            [0.0, 9.81, 0.0], [0.01, -0.02, 0.03]),
        protocol.encode_tof(3000, np.arange(64, dtype=np.int16)),
        protocol.encode_button(4000),
    ])
    frames = p.feed(stream)
    assert [f.ftype for f in frames] == [protocol.TYPE_HELLO,
                                         protocol.TYPE_IMU,
                                         protocol.TYPE_TOF,
                                         protocol.TYPE_BUTTON]
    assert frames[0].hello_text() == protocol.HELLO_TEXT
    quat, accel, gyro = frames[1].imu()
    assert np.allclose(quat, [1, 0, 0, 0], atol=1e-7)
    assert np.allclose(accel, [0, 9.81, 0], atol=1e-6)
    assert np.allclose(gyro, [0.01, -0.02, 0.03], atol=1e-7)
    assert np.array_equal(frames[2].tof(), np.arange(64))
    assert frames[3].button_event() == protocol.BUTTON_PRESS
    assert p.bad_frames == 0 and p.dropped_bytes == 0


def test_protocol_tick_wrap():
    p = protocol.ProtocolParser()
    base = protocol.TICK_MODULUS - 15  # wrap 15 us into the stream
    stream = b"".join(
        protocol.encode_button(base + 10 * i) for i in range(5))
    frames = p.feed(stream)
    ticks = [f.tick_us for f in frames]
    assert ticks == [base + 10 * i for i in range(5)], ticks
    assert frames[-1].tick_raw == 25  # what was on the wire, wrapped


def test_protocol_partial_garbage_crc():
    good1 = protocol.encode_imu(100, [1, 0, 0, 0], [0, 9.81, 0],
                                [0, 0, 0])
    bad = bytearray(protocol.encode_tof(200, np.zeros(64, np.int16)))
    bad[12] ^= 0xFF  # corrupt one payload byte -> CRC must reject it
    good2 = protocol.encode_button(300)
    stream = b"\x13\x37\xa5" + good1 + bytes(bad) + good2
    p = protocol.ProtocolParser()
    frames = []
    for i in range(len(stream)):  # byte-by-byte: partial everywhere
        frames.extend(p.feed(stream[i:i + 1]))
    assert [f.ftype for f in frames] == [protocol.TYPE_IMU,
                                         protocol.TYPE_BUTTON]
    assert p.bad_frames == 1
    assert frames[0].tick_us == 100 and frames[1].tick_us == 300


def test_app_button_toggle():
    if cv2 is None:
        print("  (skipped: cv2 unavailable)")
        return
    with tempfile.TemporaryDirectory() as td:
        app = RecorderApp(Path(td))
        parser = protocol.ProtocolParser()
        for frame in parser.feed(protocol.encode_button(0)):
            app.ingest_frame(frame)  # head button: start
        assert app.recording
        for frame in parser.feed(protocol.encode_imu(
                10, [1, 0, 0, 0], [0, 9.81, 0], [0, 0, 0])):
            app.ingest_frame(frame)
        app.ingest_camera(np.zeros((48, 64, 3), dtype=np.uint8))
        for frame in parser.feed(protocol.encode_button(20)):
            app.ingest_frame(frame)  # head button: stop + write
        assert not app.recording
        folders = list(Path(td).glob("capture_*"))
        assert len(folders) == 1
        cap = sr.read_capture(folders[0])
        assert len(cap.imu_t) == 1 and len(cap.frame_t) == 1
        assert (folders[0] / "mcu_ticks.csv").exists()
        assert (folders[0] / "video.mp4").exists()


def test_loopback_payoff():
    if cv2 is None:
        print("  (skipped: cv2 unavailable)")
        return
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "loopback"
        summary = simulator.run_loopback(out)
        assert summary["button_events"] == 1
        cap = sr.read_capture(out)  # the unchanged reader validates
        assert cap.meta["simulated"] is True
        assert cap.meta["tethered"]["sidecar"] == "mcu_ticks.csv"
        src = sr.read_capture(CAPTURE)  # loopback must be lossless
        assert len(cap.frame_t) == len(src.frame_t) == 320
        assert len(cap.imu_t) == len(src.imu_t)
        assert len(cap.tof_t) == len(src.tof_t)
        assert np.array_equal(cap.tof_mm, src.tof_mm)
        # video really recorded, all frames present
        vc = cv2.VideoCapture(str(out / "video.mp4"))
        n = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        vc.release()
        assert n == 320, n
        # sidecar: one row per sensor sample + the button; the raw
        # tick visibly wraps (starts near 2**32, ends small)
        with open(out / "mcu_ticks.csv", newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["stream", "t_s", "mcu_tick_us"]
        ticks = [int(r[2]) for r in rows[1:]]
        streams = [r[0] for r in rows[1:]]
        assert streams.count("imu") == len(cap.imu_t)
        assert streams.count("tof") == len(cap.tof_t)
        assert streams.count("button") == 1
        assert max(ticks) > protocol.TICK_MODULUS - 20_000_001
        assert min(ticks) < 1_000_000  # post-wrap values present
        # the unchanged importer payoff, same bars as the Pi-era proof
        poses = json.loads((ROOT / "jobs" / "room2"
                            / "poses.json").read_text())
        points = _load_points(ROOT / "jobs" / "room2")
        gt = json.loads((SIM / "ground_truth.json").read_text())
        scale = sr.estimate_scale_from_tof(cap, poses, points,
                                           tof_stride=8)
        dev = abs(scale["scale_m_per_unit"] / ANCHOR_SCALE - 1.0)
        assert dev < 0.03, f"scale {scale['scale_m_per_unit']:.4f} " \
                           f"off by {dev:.1%}"
        R_align = sr.register_imu_frame(cap, poses, cap.cam_from_device)
        up_imu = sr.estimate_up_from_imu(cap, R_align)
        ang = _angle_deg(up_imu, _estimate_world_up(poses))
        assert ang < 1.0, f"IMU up off by {ang:.2f} deg"
        E = R_align @ np.array(gt["boot_rotation"])
        reg_err = float(np.degrees(np.arccos(np.clip(
            (np.trace(E) - 1.0) / 2.0, -1.0, 1.0))))
        assert reg_err < 3.0, f"registration off by {reg_err:.2f} deg"
        print(f"  loopback payoff: scale {scale['scale_m_per_unit']:.4f} "
              f"m/unit ({scale['pairs']} pairs), up error {ang:.2f} deg, "
              f"registration error {reg_err:.2f} deg")


if __name__ == "__main__":
    test_protocol_roundtrip()
    print("protocol roundtrip: PASS")
    test_protocol_tick_wrap()
    print("protocol tick wrap: PASS")
    test_protocol_partial_garbage_crc()
    print("protocol partial/garbage/crc: PASS")
    test_app_button_toggle()
    print("app button toggle: PASS")
    test_loopback_payoff()
    print("loopback payoff: PASS")
    print("tethered collector contract: ALL PASS")
