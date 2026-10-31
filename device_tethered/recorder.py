"""PC recorder for the tethered ECHO Collector.

The ESP32-S3 head streams IMU + ToF frames over USB serial
(device_tethered/protocol.py); a UVC camera plugs straight into the
PC. This program is the PC half: it stamps every sensor sample on
ARRIVAL on the PC monotonic clock, grabs camera frames, and on stop
writes one capture folder in exactly the record format
echo_scene/sensor_record.py defines (video.mp4 + frames.csv +
imu.csv + tof.csv + meta.json) -- the format did not change when the
collection end moved from a Pi to a tethered head.

One format extension, kept OUT of the locked schemas: next to the
standard files the recorder writes `mcu_ticks.csv` (stream, t_s,
mcu_tick_us) -- each sensor sample's raw MCU microsecond tick beside
its PC arrival stamp. The reader ignores the extra file; it exists
so sync quality (offset, jitter, drift between the two clocks) is
measurable after the fact instead of being unknowable. meta.json
points at it under "tethered".

Start/stop: the ENTER key, or the head's hardware button (a BUTTON
frame over the same serial link). While recording, the recorder
sends the head 'R' / 'S' downlink bytes so its LED shows the state.

Structure: CaptureSession is the hardware-free core (injectable
clock, ingest methods, folder writing) and is what the loopback
simulator and the tests drive. SerialSensorSource / UvcCameraSource
are the thin threaded hardware shells around it; cv2 and pyserial
are imported lazily inside them, so this module imports cleanly on a
machine with neither (tests, review).

Samples are buffered in memory and the CSVs written at stop (the
video is written incrementally). A one-hour capture is ~360k IMU
rows -- a few hundred MB of Python objects worst case; typical
captures are minutes. Streaming CSV writes are a known v1 shortcut,
not a format question.

Run (bench, real hardware):
    python -m device_tethered.recorder --port COM7 --camera 0 \
        --out ~/echo_captures
Requires: opencv-python, pyserial (PC side only).
"""
import argparse
import csv
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echo_scene import sensor_record as sr  # noqa: E402
from device_tethered import protocol  # noqa: E402

try:
    import cv2
except ImportError:  # headless review/test machines
    cv2 = None

SIDECAR_NAME = "mcu_ticks.csv"
_EPS = 1e-6  # strictly-increasing guard step, seconds


def _default_calibration() -> np.ndarray:
    # The format's reference mounting (device/capture.py, the sim):
    # device x right / y up / z back -> camera x right / y down /
    # z forward is a 180-degree rotation about x.
    return np.diag([1.0, -1.0, -1.0])


class CaptureSession:
    """One capture, hardware-free. Feed it parsed sensor frames and
    camera images; finish() writes the folder.

    clock: zero-arg callable returning seconds (time.monotonic in
    production, a manual clock in the loopback). All stamps are
    clock() - t0 with t0 taken at begin(); per-stream stamps are
    forced strictly increasing (the reader requires it) by bumping
    a colliding stamp one microsecond past the previous one.
    """

    def __init__(self, folder, clock=time.monotonic, fps: float = 30.0,
                 cam_from_device=None, tof_fov_deg: float = 45.0,
                 simulated: bool = False, device_board: str =
                 "ESP32-S3 (tethered)", firmware: str = "ECHO-T1"):
        self.folder = Path(folder)
        self._clock = clock
        self._fps = float(fps)
        self._cam_from_device = (_default_calibration()
                                 if cam_from_device is None
                                 else np.asarray(cam_from_device,
                                                 dtype=np.float64))
        self._tof_fov_deg = float(tof_fov_deg)
        self._simulated = bool(simulated)
        self._device_board = device_board
        self._firmware = firmware
        self.on_button = None       # callable(event) or None
        self.hello_text = None
        self._t0 = None
        self._writer = None
        self._frame_idx = 0
        self._imu = []              # (t, quat4, accel3, gyro3)
        self._tof = []              # (t, zones64 int16)
        self._ticks = []            # (stream, t, tick_raw)
        self._last_t = {"imu": -1.0, "tof": -1.0, "frames": -1.0}
        self.button_events = 0

    # -- lifecycle ----------------------------------------------------
    def begin(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        self._t0 = self._clock()
        return self

    @property
    def recording(self) -> bool:
        return self._t0 is not None

    def _stamp(self, stream: str) -> float:
        t = self._clock() - self._t0
        if t <= self._last_t[stream]:
            t = self._last_t[stream] + _EPS
        self._last_t[stream] = t
        return t

    # -- ingest ---------------------------------------------------------
    def ingest_frame(self, frame: protocol.Frame):
        """One parsed protocol frame (IMU / TOF / BUTTON / HELLO)."""
        if frame.ftype == protocol.TYPE_IMU:
            quat, accel, gyro = frame.imu()
            t = self._stamp("imu")
            self._imu.append((t, quat, accel, gyro))
            self._ticks.append(("imu", t, frame.tick_raw))
        elif frame.ftype == protocol.TYPE_TOF:
            t = self._stamp("tof")
            self._tof.append((t, frame.tof()))
            self._ticks.append(("tof", t, frame.tick_raw))
        elif frame.ftype == protocol.TYPE_BUTTON:
            t = self._clock() - self._t0
            self._ticks.append(("button", t, frame.tick_raw))
            self.button_events += 1
            if self.on_button is not None:
                self.on_button(frame.button_event())
        elif frame.ftype == protocol.TYPE_HELLO:
            self.hello_text = frame.hello_text()

    def ingest_camera(self, image) -> str:
        """One camera frame (BGR ndarray). Returns its frames.csv
        file name (pipeline-style frame_00001.jpg, the convention the
        Pi program and the frame extractor share)."""
        if cv2 is None:
            raise RuntimeError("camera ingest needs opencv (cv2)")
        t = self._stamp("frames")
        self._frame_idx += 1
        name = f"frame_{self._frame_idx:05d}.jpg"
        if self._writer is None:
            h, w = image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self.folder / "video.mp4"), fourcc, self._fps, (w, h))
            if not self._writer.isOpened():
                raise RuntimeError("VideoWriter could not open video.mp4")
            self._frame_t = []
            self._frame_names = []
        self._writer.write(image)
        self._frame_t.append(t)
        self._frame_names.append(name)
        return name

    # -- finish ---------------------------------------------------------
    def finish(self) -> dict:
        if self._t0 is None:
            raise RuntimeError("session not begun")
        if not self._imu or not self._frame_idx:
            raise ValueError("empty capture: need >= 1 IMU sample and "
                             ">= 1 camera frame")
        duration = max(self._last_t.values())
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        sr.write_frames(self.folder,
                        np.arange(1, self._frame_idx + 1),
                        np.array(self._frame_t), self._frame_names)
        imu_t = np.array([r[0] for r in self._imu])
        sr.write_imu(self.folder, imu_t,
                     np.stack([r[1] for r in self._imu]),
                     np.stack([r[2] for r in self._imu]),
                     np.stack([r[3] for r in self._imu]))
        if self._tof:
            tof_t = np.array([r[0] for r in self._tof])
            tof_z = np.stack([r[1] for r in self._tof]).astype(np.int32)
        else:  # a capture with no ranging is degenerate but writable
            tof_t = np.zeros(0)
            tof_z = np.zeros((0, sr.TOF_GRID * sr.TOF_GRID), dtype=np.int32)
        sr.write_tof(self.folder, tof_t, tof_z)
        with open(self.folder / SIDECAR_NAME, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stream", "t_s", "mcu_tick_us"])
            for stream, t, tick_raw in self._ticks:
                w.writerow([stream, f"{t:.6f}", int(tick_raw)])
        meta = {
            "device": {"name": "ECHO Collector", "board": self._device_board,
                       "firmware": self.hello_text or self._firmware},
            "clock": {"type": "monotonic", "unit": "s",
                      "origin_utc": datetime.now(timezone.utc).isoformat(),
                      "origin_note": "t=0 is capture start on the PC "
                                     "monotonic clock; sensor samples "
                                     "are stamped on arrival at the PC"},
            "duration_s": round(duration, 3),
            "streams": {
                "video": {"file": "video.mp4",
                          "fps_nominal": self._fps,
                          "frames_csv": "frames.csv"},
                "imu": {"hz": round(len(self._imu) / duration, 1)
                        if duration > 0 else 0.0,
                        "quat_order": "wxyz",
                        "accel_unit": "m/s^2 (device frame, incl. gravity)",
                        "gyro_unit": "rad/s (device frame)"},
                "tof": {"hz": round(len(self._tof) / duration, 1)
                        if duration > 0 else 0.0,
                        "grid": [sr.TOF_GRID, sr.TOF_GRID],
                        "fov_deg": self._tof_fov_deg,
                        "max_range_mm": 4000, "invalid": sr.TOF_INVALID,
                        "unit": "mm"},
            },
            "calibration": {
                "cam_from_device": self._cam_from_device.tolist(),
                "camera": None,  # calibrated off-device, as before
            },
            "tethered": {
                "protocol": "ECHO-T1 v1",
                "sidecar": SIDECAR_NAME,
                "tick_unit": "us",
                "tick_modulus": protocol.TICK_MODULUS,
                "note": "mcu_tick_us is the head's raw microsecond "
                        "tick (wraps at 2**32); beside each sample's "
                        "PC arrival stamp it makes the two clocks' "
                        "offset/jitter/drift measurable after the "
                        "fact. The capture reader ignores this file.",
            },
        }
        if self._simulated:
            meta["simulated"] = True
        sr.write_meta(self.folder, meta)
        return meta


# ---------------------------------------------------------------------------
# Hardware shells (only used by the CLI; lazily import serial / cv2).
# ---------------------------------------------------------------------------

class SerialSensorSource:
    """Reads the head's byte stream in a thread, enqueues parsed
    frames. start() returns False (with .error set) when the port or
    pyserial is unavailable -- the app reports it, never crashes."""

    def __init__(self, port: str, baud: int = 921600):
        self.port, self.baud = port, baud
        self.frames = queue.Queue()
        self.error = None
        self._ser = None
        self._stop = threading.Event()

    def start(self) -> bool:
        try:
            import serial
        except ImportError:
            self.error = "pyserial not installed (pip install pyserial)"
            return False
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.2)
        except Exception as e:  # port busy / absent: a bench fact
            self.error = f"cannot open {self.port}: {e}"
            return False
        threading.Thread(target=self._pump, daemon=True).start()
        return True

    def _pump(self):
        parser = protocol.ProtocolParser()
        self.send(b"H")  # ask the head to (re)send its HELLO
        while not self._stop.is_set():
            try:
                chunk = self._ser.read(4096)
            except Exception as e:
                self.error = f"serial read failed: {e}"
                return
            if chunk:
                for frame in parser.feed(chunk):
                    self.frames.put(frame)

    def send(self, data: bytes):
        if self._ser is not None:
            try:
                self._ser.write(data)
            except Exception:
                pass  # LED downlink is best-effort by design

    def stop(self):
        self._stop.set()
        if self._ser is not None:
            self._ser.close()


class UvcCameraSource:
    """Grabs UVC frames in a thread, enqueues BGR ndarrays."""

    def __init__(self, index: int = 0):
        self.index = index
        self.frames = queue.Queue(maxsize=120)
        self.error = None
        self._cap = None
        self._stop = threading.Event()

    def start(self) -> bool:
        if cv2 is None:
            self.error = "opencv not installed (pip install opencv-python)"
            return False
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            self.error = f"cannot open camera {self.index}"
            return False
        threading.Thread(target=self._pump, daemon=True).start()
        return True

    def _pump(self):
        while not self._stop.is_set():
            ok, image = self._cap.read()
            if not ok:
                self.error = "camera read failed"
                return
            try:
                self.frames.put(image, timeout=0.5)
            except queue.Full:
                pass  # dropping a frame beats blocking the camera

    def stop(self):
        self._stop.set()
        if self._cap is not None:
            self._cap.release()


class RecorderApp:
    """Toggle logic + pump draining around a CaptureSession factory.
    Kept separate from the CLI's stdin handling so the toggle rules
    (ENTER or head button, same path) are unit-testable."""

    def __init__(self, out_root: Path, session_kwargs=None):
        self.out_root = Path(out_root)
        self.session = None
        self._kwargs = session_kwargs or {}
        self.on_state_change = None  # callable(recording: bool)
        self._hello = None           # last HELLO seen, recording or not

    @property
    def recording(self) -> bool:
        return self.session is not None

    def toggle(self):
        if self.session is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session = CaptureSession(
                self.out_root / f"capture_{stamp}", **self._kwargs).begin()
            self.session.hello_text = self._hello
        else:
            self.session.finish()
            self.session = None
        if self.on_state_change is not None:
            self.on_state_change(self.recording)

    def ingest_frame(self, frame: protocol.Frame):
        if frame.ftype == protocol.TYPE_HELLO:
            self._hello = frame.hello_text()
        if frame.ftype == protocol.TYPE_BUTTON and \
                frame.button_event() == protocol.BUTTON_PRESS:
            self.toggle()
        elif self.session is not None:
            self.session.ingest_frame(frame)

    def ingest_camera(self, image):
        if self.session is not None:
            self.session.ingest_camera(image)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ECHO tethered recorder")
    ap.add_argument("--port", required=True, help="head's serial port")
    ap.add_argument("--camera", type=int, default=0, help="UVC index")
    ap.add_argument("--out", default="~/echo_captures")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args(argv)

    app = RecorderApp(Path(args.out).expanduser(),
                      session_kwargs={"fps": args.fps})
    sensors = SerialSensorSource(args.port)
    camera = UvcCameraSource(args.camera)
    if not sensors.start():
        print(f"sensors: {sensors.error}", file=sys.stderr)
        return 1
    if not camera.start():
        print(f"camera: {camera.error}", file=sys.stderr)
        return 1
    app.on_state_change = lambda rec: (
        sensors.send(b"R" if rec else b"S"),
        print("● recording" if rec else "■ stopped — folder written"))
    toggles = queue.Queue()

    def _stdin_pump():
        while True:
            try:
                input()
            except EOFError:
                return
            toggles.put(1)

    threading.Thread(target=_stdin_pump, daemon=True).start()
    print("ENTER or the head button starts/stops a capture; Ctrl-C quits.")
    try:
        while True:
            try:
                frame = sensors.frames.get(timeout=0.02)
                app.ingest_frame(frame)
            except queue.Empty:
                pass
            try:
                app.ingest_camera(camera.frames.get_nowait())
            except queue.Empty:
                pass
            if not toggles.empty():
                toggles.get()
                app.toggle()
            if sensors.error or camera.error:
                print(sensors.error or camera.error, file=sys.stderr)
                if app.recording:
                    app.toggle()  # write out whatever was captured
                return 1
    except KeyboardInterrupt:
        if app.recording:
            app.toggle()
        return 0
    finally:
        sensors.stop()
        camera.stop()


if __name__ == "__main__":
    raise SystemExit(main())
