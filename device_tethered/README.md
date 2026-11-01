# ECHO Collector — tethered build (ESP32-S3 head + PC recorder)

> **SUPERSEDED by the Android collector** (`android_collector/`, 2026-10-31): the user's rooted OnePlus phone is now the collection end — zero extra hardware. The record format and PC-side importer are unchanged; this tethered build is kept as a proven reference implementation.

**Status: software written and proven WITHOUT hardware.** The wire
protocol, the PC recorder and the record-format output are locked by
tests (`tests/test_tethered.py`), including a full loopback: the
committed simulated capture of the user's room, re-emitted as
protocol bytes + a synthetic camera feed and recorded by this code,
passes the unchanged format reader and reproduces the unchanged
importer payoff (scale 0.2501 m/unit vs the 0.25 anchor, IMU level
0.13° off the pipeline estimate — the same numbers as the Pi-era
proof, because the loopback is lossless). The firmware sketch has
NOT been compiled or run on a real ESP32-S3; the bring-up checklist
below is part of the deliverable.

This design supersedes the Pi program in `device/` (kept for
reference). The user's architecture call: compute lives on the PC;
the hand holds only sensors. The capture-folder format in
`echo_scene/sensor_record.py` did **not** change — only the
collection end did, which is exactly what fixing the format first
was for.

## Shape of the thing

```
  handheld head (breadboard)                PC
  ┌──────────────────────────────┐   USB   ┌────────────────────┐
  │ ESP32-S3 ── I2C ── BNO085    │ ══════► │ recorder.py        │
  │              └── VL53L5CX    │  serial │  stamps on arrival │
  │ button + LED                 │ ◄────── │  (PC monotonic)    │
  │ UVC camera ─────────────USB──┼────────►│  writes the folder │
  └──────────────────────────────┘         └────────────────────┘
```

Parts (user's budget call: the MCU is a ~¥50 part; sensors are
where the money goes): ESP32-S3 dev board, BNO085 breakout
(on-chip fusion is required — the format stores fused quaternions),
VL53L5CX breakout, UVC USB camera (wide angle preferred), one
pushbutton + LED, Dupont leads, a USB cable long enough to walk
with. No screen, no battery, no storage on the head: the PC does
compute, storage and display.

Wiring: I2C SDA GPIO8 / SCL GPIO9 (change in the sketch if your
board differs), both sensors on the one bus — BNO085 at 0x4B,
VL53L5CX at 0x29, no clash. Button GPIO11 to GND (internal
pull-up), LED GPIO10 via a resistor.

## Protocol — ECHO-T1 v1

Reference implementation: `protocol.py` (the firmware implements
the same bytes in C++; if they ever disagree, protocol.py and this
spec win). All integers little-endian.

| field | size | meaning |
|---|---|---|
| sync | 2 B | `0xA5 0x5A` |
| type | 1 B | 0x01 IMU · 0x02 TOF · 0x03 BUTTON · 0x04 HELLO |
| tick | 4 B | MCU microseconds (`micros()`), modulo 2³² |
| len | 2 B | payload bytes (parser rejects > 4096) |
| payload | len B | per type, below |
| crc16 | 2 B | CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) over type..payload |

Payloads:

- **IMU (0x01), 40 B** — 10 × float32: `qw,qx,qy,qz` fused
  orientation (world-from-device, scalar-first, the record
  format's convention), `ax,ay,az` m/s² incl. gravity, `gx,gy,gz`
  rad/s, all device frame. One frame per BNO085 rotation-vector
  report (100 Hz), carrying the freshest accel/gyro.
- **TOF (0x02), 128 B** — 64 × int16 zone ranges in mm, row-major
  from the imager's top-left, −1 = no target. Byte-for-byte the
  record format's tof.csv row. 15 Hz at 8×8 (the part's max).
- **BUTTON (0x03), 1 B** — 0x01 = pressed (debounced on the MCU).
  The PC toggles recording per press.
- **HELLO (0x04), text** — once at boot, e.g.
  `ECHO-T1 fw=0.1 proto=1 imu=100 tof=15`. Informational.

Tick wrap: the µs tick wraps every 2³² µs ≈ 71.6 min. Frames on
one serial link arrive in tick order, so the parser unwraps by
forward-delta accumulation — exact while consecutive frames are
< half a wrap apart in true time (the slowest stream gaps 67 ms).
The recorder stores each sample's RAW tick in the sidecar (below),
so the wrap stays visible in the data, not just handled.

Downlink (PC → head, single bytes, best-effort): `R` = recording
(LED on), `S` = stopped (LED off), `H` = re-send HELLO (the
recorder sends this on port open, since the head may have booted
before the port existed).

Data rate ≈ 7.2 KB/s total. Any USB CDC link is loafing.

## Recorder

```bash
pip install opencv-python pyserial   # PC side only
python -m device_tethered.recorder --port COM7 --camera 0 \
    --out ~/echo_captures
```

ENTER or the head's button starts/stops a capture. Each capture
lands in `capture_YYYYMMDD_HHMMSS/` in exactly the standard
format: `video.mp4`, `frames.csv`, `imu.csv`, `tof.csv`,
`meta.json`. Stamps: **PC monotonic clock, sensor samples stamped
on arrival** — the head's clock never enters the format's
timestamps. Instead, one minimal extension file rides along:

- **`mcu_ticks.csv`** — `stream,t_s,mcu_tick_us`: every sensor
  sample's raw MCU tick beside its PC arrival stamp. The format
  reader ignores it; meta.json points at it under `tethered`.
  With both clocks' readings per sample, offset, jitter and drift
  between them are measurable from any real capture — which is
  the only honest way to claim the two clocks agree.

Frame files are named `frame_00001.jpg`, … in frames.csv — the
convention the pipeline's frame extractor and the old Pi program
share, so poses from a pipeline run key onto capture times
unchanged. Camera frames are stamped when the recorder ingests
them (grab → queue → stamp; see the honest boundary).

## Firmware

`firmware/echo_collector/echo_collector.ino` — Arduino framework
for hackability. Board: ESP32S3 Dev Module, "USB CDC On Boot:
Enabled". Libraries: SparkFun BNO08x, SparkFun VL53L5CX. It polls
the IMU (rotation vector + accel + gyro at 100 Hz) and the ToF
(15 Hz), emits the frames above, debounces the button, and drives
the LED from the downlink bytes. A dead sensor at boot blinks the
LED forever instead of streaming partial data.

## Bring-up checklist (first hardware session)

1. **I2C scan**: expect 0x4B (BNO085) and 0x29 (VL53L5CX).
2. **Quaternion order/axes**: hold the head still, rotate 90°
   about each axis; the recorded quaternions must say the same,
   in the format's device frame (x right, y up, z back). The
   BNO085 library reports (w,x,y,z) — the sketch writes them in
   that order into the wxyz payload; verify, don't trust.
3. **Achieved rates**: recorder meta.json prints measured stream
   rates; confirm IMU ≈ 100 Hz and ToF ≈ 15 Hz under real I2C
   load. If the BNO08x library interprets the report interval
   differently than the sketch assumes, fix `IMU_REPORT_MS`.
4. **ToF semantics**: tape-measure checks at 0.3/1/2/3 m on a
   matte wall; window glass and black cloth must come out −1 or
   honest, never invented. The sketch accepts target status 5/9
   as valid — re-check against what the part reports.
5. **Sync measurement**: from `mcu_ticks.csv`, plot PC arrival
   minus unwrapped MCU tick: report the offset's jitter (expect
   USB-serial milliseconds, plus the camera's own queueing). This
   number bounds every cross-stream claim the importer makes.
6. **Camera timestamps**: film a stopwatch; frames.csv stamps are
   ingest-time, so measure the grab-to-stamp latency spread.

## Honest boundary — what only the real head can prove

The loopback proves bytes, parsing, recording and payoff maths.
It fabricates none of the physics, so it cannot prove:

- **Serial jitter / throughput under real load** — USB CDC
  buffering, OS scheduling and the I2C polling loop all add
  arrival jitter the simulator (which stamps from the tick
  itself) does not have.
- **USB camera timestamp behaviour** — real UVC drivers buffer;
  frame ingest stamps will lag exposure by a device-specific,
  possibly variable amount.
- **Sensor rates under real I2C load** — whether BNO085 reports
  and VL53L5CX ranging actually sustain 100 Hz / 15 Hz on one
  400 kHz bus through these libraries.
- **Real sensor physics** — ToF on glass/mirrors/dark cloth,
  IMU drift over minutes, and the hand-built board's true
  `cam_from_device` (the recorder ships the reference mounting;
  checklist item 2/4 is where the real one gets measured).
