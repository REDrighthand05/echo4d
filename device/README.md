# ECHO Collector — handheld collection device (Pi side)

**SUPERSEDED (2026-10-31): the tethered ESP32-S3 design in
`device_tethered/` replaces this Pi program — the PC does compute,
storage and display; the hand holds only sensors + a UVC camera.
Kept for reference; the record format both write is unchanged.**

**Status: software written and logic-reviewed; NOT YET RUN ON HARDWARE.**
`capture.py` has never touched a real Pi, camera, IMU or ToF sensor.
Everything in the "bring-up checklist" below must be verified in the
first hardware session before any capture from this program is trusted
as data. The format it writes, however, is already proven end-to-end on
the PC side with simulated sensor data — see
`experiments/sensor_sim/payoff_report.md` (true scale and level both
recovered from a fabricated capture of the user's room).

## What the device is for

One job: walk a space once, slowly, and bring back a *synchronised*
record — video, fused orientation, and an 8×8 ranging grid, all on one
monotonic clock — in the ECHO record format. The PC software
(`echo_scene/sensor_record.py`, which is also the format spec) turns
that folder into a reconstruction whose **scale is measured** (ToF) and
whose **up is measured** (IMU), instead of estimated after the fact.

## Assumed parts (reference build)

| part | role | notes |
|---|---|---|
| Raspberry Pi 4 / 5 (or Zero 2 W) | the board | Pi 4/5 preferred for the encoder + 3 threads |
| Pi Camera Module 3 (or HQ + lens) | video | picamera2, H.264 1080p30 |
| BNO055 breakout | IMU | **on-chip fusion is required** — the format stores fused quaternions |
| VL53L5CX breakout | 8×8 ToF | I2C, 0x29 |
| Big pushbutton + LED | start/stop, status | button GPIO17 (to GND), LED GPIO27 |
| USB power bank | power | — |

Wiring notes: everything sensor-side is I2C bus 1 (SDA GPIO2 / SCL
GPIO3). **Address plan: VL53L5CX at its default 0x29, BNO055 with ADR
low at 0x28** — both parts can sit at 0x29, which is the classic
first-boot failure; `capture.py` already opens the BNO055 at 0x28.

## Run

```bash
pip install picamera2 adafruit-blinka adafruit-circuitpython-bno055 \
            adafruit-circuitpython-vl53l5cx gpiozero numpy
python3 capture.py                 # button mode: press to start/stop
python3 capture.py --keyboard      # bench mode: starts now, Enter stops
python3 capture.py --out /path     # captures root (default ~/echo_captures)
```

Each capture lands in `capture_YYYYMMDD_HHMMSS/` with `video.mp4`,
`frames.csv`, `imu.csv`, `tof.csv`, `meta.json` — exactly the format
`echo_scene/sensor_record.py` specifies and reads.

## Bring-up checklist (first hardware session)

1. **I2C scan** (`i2cdetect -y 1`): expect 0x28 (BNO055) and 0x29
   (VL53L5CX), nothing else, no ghosts.
2. **BNO055 quaternion order**: `adafruit_bno055.quaternion` is coded
   as (w, x, y, z). Hold the board still, rotate it 90° about each
   axis, and confirm the recorded quaternions say the same. If the
   order or signs differ, fix in `ImuThread` before recording anything.
3. **Axis remap**: `AXIS_REMAP` in capture.py is identity. With the
   board in its final mounting, check that a resting capture reports
   accel ≈ +9.81 on device +y (up) and that the orientation stream
   behaves; set the remap matrix to the physical mounting if not.
4. **VL53L5CX distance semantics**: the code treats the adafruit
   library's `distance` grid as centimetres with ≤0 meaning no target,
   mapped to whole mm with −1 for no target. Verify against a tape
   measure at 0.3 m / 1 m / 2 m / 3 m on a matte wall; check what the
   library reports for a window (glass) and a black cloth — those
   zones must come out −1 or honest, never invented.
5. **Frame timestamps**: frames.csv timestamps are taken when encoded
   frames reach the output (delivery time, not exposure time). Measure
   the jitter (film a stopwatch / flashing LED) and write the number
   here; the importer interpolates poses, so a few ms is fine — a few
   hundred is not.
6. **Clock discipline**: confirm all three CSVs' `t_s` span the same
   interval and increase strictly (the PC reader rejects captures that
   don't).
7. **Thermal/power**: 10-minute continuous capture on the power bank;
   no thread errors in meta.json (`thread_errors` must be absent).

## Known limitations (by design, v1)

* No on-device preview screen yet: status is the console line + LED.
  A small SPI display can mirror the same status later; the scan view
  (live ToF heatmap) belongs to the later live-preview work.
* Camera intrinsics in meta.json are `null` until the camera is
  calibrated off-device; the reconstruction estimates them from the
  video as it does today.
* `cam_from_device` assumes the reference mounting in capture.py;
  re-derive it if the board layout changes (checklist item 3 covers
  the IMU side; the camera side is one fixed rotation to re-measure).
