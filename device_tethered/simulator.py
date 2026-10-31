"""Loopback simulator for the tethered collector -- no hardware.

Re-emits the committed simulated capture of the room
(experiments/sensor_sim/capture_room2_sim) the way the ESP32-S3
firmware would: its IMU and ToF rows become ECHO-T1 protocol frames
on a byte stream (MCU ticks start just before the 2**32 wrap, so
the parser's unwrap is exercised mid-capture), and its camera feed
is the job's extracted frames replayed at their recorded times.
A CaptureSession driven by a manual clock slaved to the frame ticks
records it all into a normal capture folder.

That folder must (and the tests assert it does):
  * pass echo_scene.sensor_record.read_capture validation, and
  * feed the importer to the same payoff as the original simulated
    capture (true scale from ToF, level from IMU).

So the loopback proves the whole tethered path -- protocol bytes,
parser, recorder, record format -- against the same ground truth
the Pi-era proof used. What it cannot prove is listed in the
README (serial jitter, real USB camera timestamps, sensor rates
under real I2C load).

    python -m device_tethered.simulator --out /tmp/loopback_capture
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echo_scene import sensor_record as sr  # noqa: E402
from device_tethered import protocol  # noqa: E402
from device_tethered.recorder import CaptureSession, cv2  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SIM_CAPTURE = ROOT / "experiments" / "sensor_sim" / "capture_room2_sim"
JOB = ROOT / "jobs" / "room2"
# Start the MCU tick 20 s before the wrap: the stream crosses 2**32
# mid-capture, like a head that booted ~71 minutes before recording.
TICK_OFFSET = protocol.TICK_MODULUS - 20_000_000


class ManualClock:
    """A clock the driver sets explicitly (seconds). The recorder
    stamps clock() - t0; the driver advances this to each event's
    stream time before ingesting it -- the loopback's stand-in for
    'the sample arrived at the PC at this instant'."""

    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _tick_us(t_s: float) -> int:
    return TICK_OFFSET + int(round(t_s * 1_000_000))


def build_byte_stream(cap: sr.Capture) -> bytes:
    """The whole sensor stream as the firmware would emit it:
    HELLO at boot, one BUTTON press at 0.05 s, then IMU and ToF
    frames merged in time order, ticks from TICK_OFFSET."""
    # Every frame in nondecreasing tick order, like the firmware
    # emits (the parser's unwrap contract). The button press at
    # 0.05 s is an event in the stream, not a prefix.
    events = [(0.05, protocol.encode_button(_tick_us(0.05)))]
    for t, q, a, g in zip(cap.imu_t, cap.imu_quat,
                          cap.imu_accel, cap.imu_gyro):
        events.append((t, protocol.encode_imu(_tick_us(t), q, a, g)))
    for t, zones in zip(cap.tof_t, cap.tof_mm):
        events.append((t, protocol.encode_tof(_tick_us(t), zones)))
    events.sort(key=lambda e: e[0])
    return protocol.encode_hello(TICK_OFFSET) + b"".join(
        frame for _, frame in events)


def parse_stream(data: bytes, chunk: int = 997) -> list:
    """Feed the stream to a parser in awkward chunk sizes (997 is
    coprime with every frame size, so frames split everywhere)."""
    parser = protocol.ProtocolParser()
    frames = []
    for i in range(0, len(data), chunk):
        frames.extend(parser.feed(data[i:i + chunk]))
    assert parser.bad_frames == 0, f"{parser.bad_frames} bad frames"
    return frames


def run_loopback(out_folder, sim_capture=SIM_CAPTURE, job=JOB,
                 chunk: int = 997) -> dict:
    """Record the simulated stream into out_folder; return counts."""
    if cv2 is None:
        raise RuntimeError("loopback needs opencv (cv2) for the "
                           "synthetic camera feed")
    cap = sr.read_capture(Path(sim_capture))
    frames = parse_stream(build_byte_stream(cap), chunk=chunk)
    tick0 = frames[0].tick_us

    timeline = []  # (t_s, kind, payload)
    for frame in frames:
        t = (frame.tick_us - tick0) / 1e6
        timeline.append((t, "sensor", frame))
    for t, name in zip(cap.frame_t, cap.frame_files):
        timeline.append((float(t), "camera", name))
    timeline.sort(key=lambda e: e[0])

    fps = len(cap.frame_t) / max(cap.frame_t[-1] - cap.frame_t[0], 1e-9)
    clock = ManualClock()
    session = CaptureSession(
        Path(out_folder), clock=clock, fps=fps, simulated=True,
        cam_from_device=cap.cam_from_device,
        tof_fov_deg=cap.tof_fov_deg,
        device_board="simulated (tethered loopback)").begin()
    frames_dir = Path(job) / "frames"
    for t, kind, payload in timeline:
        clock.t = t
        if kind == "sensor":
            session.ingest_frame(payload)
        else:
            image = cv2.imread(str(frames_dir / payload))
            if image is None:
                raise RuntimeError(f"cannot read {frames_dir / payload}")
            session.ingest_camera(image)
    meta = session.finish()
    return {"frames": len(cap.frame_t), "imu": len(cap.imu_t),
            "tof": len(cap.tof_t),
            "button_events": session.button_events, "meta": meta}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="tethered loopback sim")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    summary = run_loopback(Path(args.out))
    print(f"loopback capture written to {args.out}: "
          f"{summary['frames']} frames, {summary['imu']} imu, "
          f"{summary['tof']} tof")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
