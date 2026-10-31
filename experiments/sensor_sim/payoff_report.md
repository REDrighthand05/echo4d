# Sensor payoff report — simulated capture of the room

Capture: `experiments/sensor_sim/capture_room2_sim/` (SIMULATED —
fabricated from the jobs/room2 reconstruction; see the honest boundary
below). Importer: `echo_scene/sensor_record.py`, run by
`experiments/sensor_sim/import_proof.py`. The importer read only the
capture folder + the room2 reconstruction — the same inputs a real
capture would produce.

## Payoff 1 — TRUE SCALE from the ToF stream

| | value |
|---|---|
| derived scale | **0.2500 m per scene unit** |
| reference (Pioneer A/D anchor consensus) | 0.25 m/unit (centre 0.249, ±5%) |
| deviation | **+0.00%** — inside the anchor's ±5% band |
| zone pairs used | 45587 (from 764 ToF frames) |
| pairs dropped by the 3-sigma trim | 1001 |
| fit residual | RMS 20.4 mm, median abs 12.9 mm |
| median per-zone ratio (cross-check) | 0.2500 m/unit |

Method: one global least-squares fit through the origin over every
(cloud distance, ToF distance) zone pair — no voting, no per-zone
decisions.

## Payoff 2 — LEVEL from the IMU stream

| | angle |
|---|---|
| IMU-derived up vs the pipeline's up-alignment result for room2 (its pose estimator) | **0.13°** |
| IMU frame registration error vs simulated boot truth | 0.92° |
| (reference check) stored alignment.json vs its own job's estimator, room2_trim_acceptance | 0.00° |

Note on references: jobs/room2 predates alignment.json, so "the
pipeline's up-alignment result" for this job is its estimator output.
That this is the right identification is verified on the sibling job
room2_trim_acceptance, whose stored alignment.json reproduces its own
estimator's up to 0.00°. (The sibling job is a
separate reconstruction in its own COLMAP frame, ~10.8° away from
room2's — cross-frame angle comparisons between jobs are meaningless
and are not used.)

The IMU up is a MEASUREMENT (mean accelerometer vector carried into
the reconstruction frame by the registered orientation stream); the
references are the pipeline's pose-geometry ESTIMATE. Agreement at
this level means the device replaces the guess with a measurement of
comparable-or-better quality from the first second of capture.

## Honest boundary — what only real hardware can prove

This simulation validates the record format, the importer maths and
their noise robustness. It cannot validate physics it fabricated:

* The simulated ToF was raycast from the reconstruction itself
  (scaled by the anchor value), so scale agreement confirms the data
  path + estimator, NOT an independent measurement. A real VL53L5CX
  sees real surfaces: glass, mirrors, dark fabrics, sunlight and
  multi-path returns will produce zones the cloud does not predict.
* Sync: all simulated streams share one exact clock. On the board,
  camera/IMU/ToF timestamps come from different drivers; jitter of
  even 20–50 ms shifts ToF/IMU samples along a moving route.
* IMU drift here is a smooth 0.6°/min model. Real AHRS drift over
  minutes (temperature, vibration, magnetic disturbance indoors) can
  be worse and less smooth.
* ToF noise here is Gaussian in range. Real zone behaviour (crosstalk,
  cover-glass smudge, target-dependent bias) is not.
* Extrinsic calibration (cam_from_device) was exact here; a hand-built
  board needs its own calibration check.
