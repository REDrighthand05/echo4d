# EchoCollector v2

A dedicated, native Android capture app for ECHO — a clean-room rewrite of
EchoCollector, built for one phone: the **OnePlus Ace 2 Pro (PJA110)**.
No code was taken from the old APK. Kotlin, Camera2, ARCore, one clock.

## What it fixes

The old app's ARCore pose stream died after ~6 seconds of recording
(analysis of capture_20261105: "10 frames in 6 s", then cut to Camera2-only;
poses never resumed). This app removes every cause:

- The ARCore session is created **once** at start and resumed **once**. It is
  paused only when the user presses Stop. There is no watchdog, fallback or
  mode switch that can pause or replace it mid-recording.
- `session.update()` runs in a tight loop on its **own dedicated thread**.
  That thread copies the pose (7 floats + timestamp + tracking state) and
  nothing else — no I/O, no UI, no lock it can block on — so it cannot be
  starved.
- The whole session lives in a **foreground service with a partial wake
  lock**, so Android cannot throttle or kill it when the screen goes off.
- If tracking is lost, we keep recording and log the tracking state.
  Losing tracking is data (a `PAUSED` row in `poses.csv`), not a reason
  to stop.

## What it does

Three screens, nothing else:

- **Record** — one big Record/Stop button (pulses while recording), a live
  status line (elapsed, pose count, frame count, tracking state, lens), a
  Main / Ultrawide lens toggle, and a sensor checklist. A warning card
  appears the moment any stream stalls.
- **Captures** — past captures with name, date, size, video duration, pose
  count and **pose coverage %** (colour-coded: green ≥90%, amber ≥50%,
  red below), with delete.
- **Diagnostics** — the capability probe manifest for this exact phone.

### Capability probe (first run + Diagnostics)

Enumerates and **tests** everything, then writes `capability.json`:

- Every camera (`CameraManager.getCameraIdList()`): facing, focal lengths,
  sensor size, output sizes, and whether it **actually opens** (each one is
  opened briefly and closed; failures carry the exact exception message).
- Every sensor (`SensorManager.getSensorList(TYPE_ALL)`): type, name,
  vendor, range, resolution, power, min delay; the ones the recorder uses
  are marked.
- ARCore: `ArCoreApk.checkAvailability()` plus an actual attempt to create
  a `Session`.

### Stream health watchdog

Every stream (video frames, ARCore poses, IMU, magnetometer, light,
barometer) has a last-sample timestamp. If any stream produces nothing for
**>3 seconds**, the app logs the exact state (ARCore tracking state,
camera state, sensor listener state), shows a visible warning naming the
stream, and writes it to `watchdog.log`. Nothing is ever silently lost.
On Stop, a post-record self-check (`selfcheck.json`) reports per-stream
sample counts, first/last timestamps, coverage % against the video
duration, and the largest gap — and flags anything below 90% pose
coverage or with a gap over 1 s.

## Capture format

One folder per capture,
`EchoCaptures/capture_YYYYMMDD_HHMMSS/` under the app's external files
directory (`Android/data/com.echo.collector/files/Movies/EchoCaptures/`).
Same format the desktop EchoScene app already reads:

| File | Contents |
|---|---|
| `video.mp4` | H.264, 1920×1080, 30 fps, main camera (or ultrawide if selected) |
| `frames.csv` | `frame,t_s,file` — one row per video frame |
| `poses.csv` | `t_s,tx,ty,tz,qx,qy,qz,qw,tracking` — ARCore pose, whole walk |
| `imu.csv` | `t_s,ax,ay,az,gx,gy,gz` — accelerometer + gyroscope |
| `mag.csv` | `t_s,mx,my,mz` — magnetometer |
| `light.csv` | `t_s,lux` — ambient light |
| `baro.csv` | `t_s,hPa` — barometer, only if the sensor exists |
| `meta.json` | device, lens + FOV, resolution/fps, sensors recorded, start wall-clock time, `t0` |
| `capability.json` | copy of the probe manifest for this phone |
| `selfcheck.json` | post-record self-check (counts, coverage, gaps) |
| `watchdog.log` | stale-stream events, if any |

**One clock.** Every timestamp in every file is
`SystemClock.elapsedRealtimeNanos()`: `t_s = (now - t0) / 1e9`, where `t0`
is taken once at the start of the recording. `SensorEvent.timestamp`,
ARCore `Frame.timestamp` and Camera2 `SENSOR_TIMESTAMP` are all in this
same monotonic domain, so every stream lines up.

**Sensors are auto-detected.** The app records whatever
`SensorManager` reports — accelerometer, gyroscope, magnetometer, light,
and barometer if present. The barometer is never assumed: the PJA110's
barometer is uncertain in spec sheets, so the app checks at runtime and
simply omits `baro.csv` if there is none.

**Lenses.** Main (24 mm, IMX890) records video + ARCore poses + sensors.
Ultrawide (16 mm, IMX355) records video + sensors only: ARCore's shared
camera only supports the camera ARCore itself chooses (the main), so
poses are not available on the ultrawide. This is shown in the UI and in
`meta.json` (`arcore_poses: false`) instead of silently producing an
empty `poses.csv`.

## Build

Prerequisites: Android Studio (Hedgehog or newer) with Android SDK 34,
or a JDK 17 + the `gradlew` script below.

```bash
./gradlew assembleDebug
# APK: app/build/outputs/apk/debug/app-debug.apk
```

Or open the project in Android Studio and press Run. The first launch
asks for the Camera permission, then runs the capability probe once
(about 10 seconds — it opens each camera briefly).

## Architecture

```
MainActivity          Record screen (UI only; never touches hardware)
CapturesActivity      Past captures list + delete
DiagnosticsActivity  Capability manifest + re-run probe
RecorderService       Foreground service (type: camera) + partial wake lock.
                      Owns the session; broadcasts status ~2x/s.
RecorderSession       Camera2 + ARCore shared camera + sensors + CSV
                      writers + watchdog. One t0, one clock.
CapabilityProbe       First-run probe: cameras (open-tested), sensors, ARCore.
StreamWatchdog        Per-stream last-sample tracking, >3 s stall alerts,
                      post-record self-check.
CsvWriter             Buffered CSV writer; callbacks append to a pre-sized
                      StringBuilder, a writer thread flushes to disk.
LensHelper            Back-camera enumeration by focal length (main vs
                      ultrawide) + horizontal FOV.
CaptureStore          Lists capture folders, computes stats from the files.
```

Threads (each callback does minimal work; all I/O is on the writer thread):

- `recorder` — Camera2 operations and the capture-session callbacks.
- `arcore-update` — the `session.update()` pose loop, nothing else.
- `sensors` — `SensorEventListener` callbacks.
- `writer` — flushes every CSV buffer to disk once a second.
- `watchdog` — checks stream health once a second.

## Deliberately not included

This is an experimental capture tool, not a client app. It has no live
route view / live map, no scan guide or prompts, no auto-transfer to the
desktop, no voice notes, and no battery/thermal management. It also has
no live camera preview: the capture session carries exactly two surfaces
(ARCore's + the MediaRecorder's), which keeps the pipeline as robust as
possible. Frame the shot with the phone, press Record, and check the
result in Captures afterwards.

## Known limitations

- ARCore poses require the main lens (ARCore limitation, see Lenses).
- ARCore depth is not available on the PJA110, and there is no hardware
  depth camera / ToF sensor. Depth still comes from the video (the
  desktop DAV2 model), as before. The sensors' job is the pose, the
  level (gravity), the heading (magnetometer) and the height
  (barometer) — so the depth frames land in the right place.
- `MediaRecorder` output is a single `video.mp4`; per-frame files are
  not produced (the desktop pipeline reads `frames.csv` for timing).
