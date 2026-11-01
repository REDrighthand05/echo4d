#!/usr/bin/env python3
"""ECHO Collector — Pi-side capture program for the handheld device.

STATUS: written and logic-reviewed, NOT YET RUN ON HARDWARE. See
device/README.md for the bring-up checklist (quaternion order, I2C
addresses, axis remap and frame-timestamp jitter must be verified on
the real board before any capture is trusted).

What it does: one big button (GPIO, or Enter in --keyboard mode)
starts/stops a capture. While capturing, three threads record onto
ONE time.monotonic() clock:

  * camera  -> video.mp4 + frames.csv (one timestamp per frame)
  * BNO055  -> imu.csv at 100 Hz (fused quaternion + accel + gyro)
  * VL53L5CX-> tof.csv at 8 Hz (8x8 zones, mm, -1 = no target)

and meta.json is written when the capture stops. The folder format is
specified by echo_scene/sensor_record.py (the PC-side module); the
writers below are self-contained copies of that contract because the
Pi does not import the PC package. tests/test_sensor_record.py locks
the header strings of the two implementations together.

Axis convention (format spec): device frame x right, y up, z back
toward the operator. The BNO055's native frame depends on how the
breakout is mounted; AXIS_REMAP below maps chip vectors into the
device frame and MUST be set to match the physical mounting at
bring-up (default: identity — chip axes already aligned as mounted
flat, connector toward the operator; verify, don't trust).
"""
import argparse
import csv
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VIDEO_SIZE = (1920, 1080)
VIDEO_FPS = 30
IMU_HZ = 100.0
TOF_HZ = 8.0
TOF_FOV_DEG = 45.0
TOF_MAX_RANGE_MM = 4000
BUTTON_GPIO = 17          # BCM numbering, button to GND, internal pull-up
LED_GPIO = 27             # optional status LED
BNO055_ADDRESS = 0x28     # ADR pin low. The VL53L5CX sits at 0x29; do NOT
                          # put the BNO055 at 0x29 as well (bus conflict).

# Chip-frame -> device-frame rotation (rows = device axes in chip coords).
# Identity until the physical mounting says otherwise. Applied to accel
# and gyro vectors directly, and to the orientation quaternion by
# conjugation: q_dev = q_remap * q_chip-as-device... see ImuThread.
AXIS_REMAP = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# Camera extrinsic vs device frame (format spec value for a board where
# the camera looks along device -z... the camera frame is OpenCV
# (x right, y down, z forward); with the lens facing away from the
# operator (device z back), cam = (dev x, -dev y, -dev z).
CAM_FROM_DEVICE = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]

FRAMES_HEADER = ["frame", "t_s", "file"]
IMU_HEADER = ["t_s", "qw", "qx", "qy", "qz", "ax", "ay", "az", "gx", "gy", "gz"]
TOF_HEADER = ["t_s"] + [f"z{r * 8 + c}" for r in range(8) for c in range(8)]


def _fmt(x: float) -> str:
    return f"{x:.6f}"


# ---------------------------------------------------------------------------
# Sensor threads. Every hardware import lives inside the thread that
# needs it, so this file imports cleanly on any machine for review.
# ---------------------------------------------------------------------------

class StopFlag:
    def __init__(self):
        self.event = threading.Event()

    def stop(self):
        self.event.set()

    @property
    def stopped(self):
        return self.event.is_set()


class CameraThread(threading.Thread):
    """picamera2 H.264 recording; a custom output records the monotonic
    timestamp of every encoded frame into frames.csv.

    Honest limitation (bring-up item): the timestamp is taken when the
    encoded frame reaches the output, so it carries delivery jitter of
    a few ms against true exposure time. The PC importer tolerates
    this (it interpolates poses), and the jitter magnitude is one of
    the things the first hardware session must measure.
    """

    def __init__(self, folder: Path, t0: float, flag: StopFlag):
        super().__init__(daemon=True)
        self.folder, self.t0, self.flag = folder, t0, flag
        self.frames = 0
        self.error = None

    def run(self):
        try:
            self._run()
        except Exception as exc:  # surfaced in the status line / meta
            self.error = repr(exc)

    def _run(self):
        from picamera2 import Picamera2
        from picamera2.encoders import H264Encoder
        from picamera2.outputs import FileOutput

        outer = self

        class TimestampedOutput(FileOutput):
            def __init__(self, path, on_frame):
                super().__init__(path)
                self._on_frame = on_frame

            def outputframe(self, frame, keyframe=True, timestamp=None,
                            packet=None, audio=None):
                self._on_frame()
                return super().outputframe(frame, keyframe, timestamp,
                                           packet, audio)

        csv_file = open(self.folder / "frames.csv", "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(FRAMES_HEADER)

        def on_frame():
            t = time.monotonic() - outer.t0
            writer.writerow([outer.frames, _fmt(t),
                             f"frame_{outer.frames:05d}.jpg"])
            outer.frames += 1

        cam = Picamera2()
        config = cam.create_video_configuration(
            main={"size": VIDEO_SIZE, "format": "RGB888"},
            controls={"FrameRate": VIDEO_FPS})
        cam.configure(config)
        encoder = H264Encoder(bitrate=12_000_000)
        output = TimestampedOutput(str(self.folder / "video.mp4"), on_frame)
        cam.start_recording(encoder, output)
        try:
            while not self.flag.stopped:
                time.sleep(0.05)
        finally:
            cam.stop_recording()
            cam.close()
            csv_file.close()


class ImuThread(threading.Thread):
    """BNO055 at IMU_HZ: fused quaternion (the chip's AHRS output --
    the format stores fused orientations, so an IMU without on-chip
    fusion is NOT a drop-in replacement), accel incl. gravity, gyro."""

    def __init__(self, folder: Path, t0: float, flag: StopFlag):
        super().__init__(daemon=True)
        self.folder, self.t0, self.flag = folder, t0, flag
        self.samples = 0
        self.error = None

    def run(self):
        try:
            self._run()
        except Exception as exc:
            self.error = repr(exc)

    def _run(self):
        import board
        import busio
        import adafruit_bno055
        import numpy as np

        remap = np.array(AXIS_REMAP, dtype=float)
        i2c = busio.I2C(board.SCL, board.SDA)
        sensor = adafruit_bno055.BNO055_I2C(i2c, address=BNO055_ADDRESS)

        with open(self.folder / "imu.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(IMU_HEADER)
            period = 1.0 / IMU_HZ
            next_t = time.monotonic()
            while not self.flag.stopped:
                next_t += period
                # adafruit_bno055.quaternion returns (w, x, y, z) --
                # VERIFY at bring-up; some firmware paths differ.
                q = sensor.quaternion
                a = sensor.acceleration   # m/s^2, includes gravity
                g = sensor.gyro           # rad/s
                if q is None or a is None or g is None or q[0] is None:
                    self._sleep_until(next_t)
                    continue
                qw, qx, qy, qz = (float(v) for v in q)
                # remap: q' = R * q as a rotation composition
                # (device-from-world = remap @ chip-from-world applied
                # to vectors; for quaternions q_dev = q_remap (x) q)
                qr = _quat_from_matrix(remap)
                qw, qx, qy, qz = _quat_mul(qr, (qw, qx, qy, qz))
                a_dev = remap @ np.array(a, dtype=float)
                g_dev = remap @ np.array(g, dtype=float)
                t = time.monotonic() - self.t0
                writer.writerow([_fmt(t), _fmt(qw), _fmt(qx), _fmt(qy),
                                 _fmt(qz)]
                                + [_fmt(float(v)) for v in a_dev]
                                + [_fmt(float(v)) for v in g_dev])
                self.samples += 1
                self._sleep_until(next_t)

    @staticmethod
    def _sleep_until(deadline: float):
        delay = deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def _quat_from_matrix(R):
    """3x3 rotation matrix -> (w, x, y, z)."""
    import math
    tr = R[0][0] + R[1][1] + R[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return (0.25 * s, (R[2][1] - R[1][2]) / s,
                (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s)
    i = max(range(3), key=lambda k: R[k][k])
    if i == 0:
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2
        return ((R[2][1] - R[1][2]) / s, 0.25 * s,
                (R[0][1] + R[1][0]) / s, (R[0][2] + R[2][0]) / s)
    if i == 1:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2
        return ((R[0][2] - R[2][0]) / s, (R[0][1] + R[1][0]) / s,
                0.25 * s, (R[1][2] + R[2][1]) / s)
    s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2
    return ((R[1][0] - R[0][1]) / s, (R[0][2] + R[2][0]) / s,
            (R[1][2] + R[2][1]) / s, 0.25 * s)


def _quat_mul(p, q):
    """Hamilton product p (x) q, scalar-first quaternions."""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return (pw * qw - px * qx - py * qy - pz * qz,
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw)


class TofThread(threading.Thread):
    """VL53L5CX at TOF_HZ: 8x8 distance grid in mm, -1 = no target."""

    def __init__(self, folder: Path, t0: float, flag: StopFlag):
        super().__init__(daemon=True)
        self.folder, self.t0, self.flag = folder, t0, flag
        self.samples = 0
        self.error = None

    def run(self):
        try:
            self._run()
        except Exception as exc:
            self.error = repr(exc)

    def _run(self):
        import board
        import busio
        import adafruit_vl53l5cx

        i2c = busio.I2C(board.SCL, board.SDA)
        sensor = adafruit_vl53l5cx.VL53L5CX(i2c)
        sensor.resolution = adafruit_vl53l5cx.RESOLUTION_8X8
        sensor.ranging_freq = int(TOF_HZ)
        sensor.start_ranging()
        try:
            with open(self.folder / "tof.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(TOF_HEADER)
                while not self.flag.stopped:
                    if not sensor.data_ready:
                        time.sleep(0.005)
                        continue
                    # adafruit exposes distance in CENTIMETRES as an
                    # 8x8 nested list, target status separately; a
                    # zone with no valid target reads 0 / status != 5.
                    # VERIFY the exact semantics at bring-up.
                    dist_cm = sensor.distance
                    zones = []
                    for r in range(8):
                        for c in range(8):
                            v = dist_cm[r][c] if dist_cm else None
                            if v is None or v <= 0:
                                zones.append(-1)
                            else:
                                mm = int(round(float(v) * 10.0))
                                zones.append(mm if mm <= TOF_MAX_RANGE_MM
                                             else -1)
                    t = time.monotonic() - self.t0
                    writer.writerow([_fmt(t)] + [str(z) for z in zones])
                    self.samples += 1
        finally:
            sensor.stop_ranging()


# ---------------------------------------------------------------------------
# Capture session
# ---------------------------------------------------------------------------

def board_model() -> str:
    p = Path("/proc/device-tree/model")
    try:
        return p.read_bytes().decode(errors="replace").strip("\x00").strip()
    except OSError:
        return "unknown (not a Pi?)"


def run_capture(out_root: Path, wait_for_stop) -> Path:
    """wait_for_stop: callable blocking until the operator stops the
    capture (button press or Enter, decided by main)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = out_root / f"capture_{stamp}"
    folder.mkdir(parents=True, exist_ok=False)
    flag = StopFlag()
    t0 = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()

    led = None
    try:
        from gpiozero import LED
        led = LED(LED_GPIO)
        led.on()
    except Exception:
        led = None  # LED is decoration; never block a capture on it

    threads = [CameraThread(folder, t0, flag),
               ImuThread(folder, t0, flag),
               TofThread(folder, t0, flag)]
    for th in threads:
        th.start()
    print(f"[echo-collector] recording -> {folder}", flush=True)
    try:
        wait_for_stop()
    except (EOFError, KeyboardInterrupt):
        pass
    flag.stop()

    for th in threads:
        th.join(timeout=10)
    if led is not None:
        led.off()
    duration = time.monotonic() - t0

    errors = {type(th).__name__: th.error for th in threads if th.error}
    meta = {
        "format_version": 1,
        "device": {"name": "ECHO Collector", "board": board_model(),
                   "firmware": "capture.py 1.0"},
        "simulated": False,
        "clock": {"type": "monotonic", "unit": "s",
                  "origin_utc": started_utc,
                  "origin_note": "t=0 is capture start on the device "
                                 "monotonic clock; origin_utc is the "
                                 "wall-clock instant of that start"},
        "duration_s": round(duration, 3),
        "streams": {
            "video": {"file": "video.mp4", "size": list(VIDEO_SIZE),
                      "fps_nominal": VIDEO_FPS, "frames_csv": "frames.csv",
                      "frames_recorded": threads[0].frames},
            "imu": {"hz": IMU_HZ, "quat_order": "wxyz",
                    "accel_unit": "m/s^2 (device frame, incl. gravity)",
                    "gyro_unit": "rad/s (device frame)",
                    "samples_recorded": threads[1].samples},
            "tof": {"hz": TOF_HZ, "grid": [8, 8], "fov_deg": TOF_FOV_DEG,
                    "max_range_mm": TOF_MAX_RANGE_MM, "invalid": -1,
                    "unit": "mm", "samples_recorded": threads[2].samples},
        },
        "calibration": {
            "cam_from_device": CAM_FROM_DEVICE,
            "camera": None,  # filled by camera calibration, off-device
            "note": "cam_from_device assumes the reference mounting in "
                    "capture.py; verify at bring-up (README checklist)",
        },
    }
    if errors:
        meta["thread_errors"] = errors
    (folder / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[echo-collector] stopped. {duration:.1f}s, "
          f"frames={threads[0].frames} imu={threads[1].samples} "
          f"tof={threads[2].samples}"
          + (f" ERRORS: {errors}" if errors else ""), flush=True)
    return folder


def main() -> int:
    ap = argparse.ArgumentParser(description="ECHO Collector capture")
    ap.add_argument("--out", default=str(Path.home() / "echo_captures"),
                    help="root folder for captures")
    ap.add_argument("--keyboard", action="store_true",
                    help="start immediately; Enter stops (no GPIO)")
    args = ap.parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.keyboard:
        print("[echo-collector] keyboard mode — recording starts now, "
              "Enter stops", flush=True)
        run_capture(out_root, wait_for_stop=lambda: input())
        return 0

    # Button mode: one Button instance owns the pin for the whole
    # session; press to start, press again to stop.
    try:
        from gpiozero import Button
        button = Button(BUTTON_GPIO, pull_up=True)
    except Exception as exc:
        print(f"[echo-collector] GPIO unavailable ({exc!r}); "
              "use --keyboard for bench testing", flush=True)
        return 2
    print("[echo-collector] ready — press the button to record",
          flush=True)
    button.wait_for_press()
    time.sleep(0.5)  # debounce: the start press must not stop instantly
    run_capture(out_root, wait_for_stop=button.wait_for_press)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
