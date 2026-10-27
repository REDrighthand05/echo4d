# Desk V2: external TUM mocap pose preflight (research-only)

**Pre-registered in Issue #15 comment 6437774757; not yet executed on the original source.** This candidate does not modify the reconstruction, frozen data, 48/12 roles, or 15cm gate. It cannot establish independent surface truth.

## Immutable source and procedure

- Original real desk V2: [run 40404725134](https://github.com/REDrighthand05/echo4d/actions/runs/40404725134), ZIP SHA256 df328812537ea4e4667f36ec7cdd9f901ff5c7fe25a1b37022daf81c20538104.
- Mesh SHA256 320da61fcd421c0e55aa6fc014750aeaa3e6e2d98b607e9feaae268f83d2a939; metric trajectory SHA256 5d60549557d38ef2e7a3f864c4f6dac52ccb7e88574a9fe97c9f1080b80af392.
- Original TUM Freiburg1 desk archive SHA256 e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c, original rgb.txt SHA256 d1bc510ecca08540e03be8df55af8857753b614d8a5c38526bc669ce6c802284. These are project-observed pins, **not vendor signatures**. The exact mocap groundtruth.txt SHA is recorded when the original archive is opened, never invented.
- Select original RGB timestamps from rgb.txt[::8][:75], first 60 geometry frames, holdouts 4,9,14,19,24,29,34,39,44,49,54,59. Never use artificial 4fps MP4 timestamps or choose a new cohort.
- Fixed reference interpolation: original TUM mocap translation + quaternion SLERP; require an enclosing bracket no wider than 100ms. No extrapolation, offset fitting, posthoc tolerance adjustment, or missing-pose substitution.
- Fit one proper rigid SE(3) alignment on **48 training camera centers only**, with scale fixed at 1.0 (the sensor-derived metric scale is preserved). Evaluate the 12 original held-out pose center/orientation residuals without fitting to them. No depth pixels are read.
- TUM motion capture is an independent *pose measurement*, not independent scanned surface truth. The dataset's published camera-frame calibration/time offset is inherited, not independently recalibrated.

## Reproduce and blocked receipt

Run from the repository root after obtaining the two pinned original archives:

    PYTHONPATH=src python tools/desk_external_pose_audit.py --official-tum-archive rgbd_dataset_freiburg1_desk.tgz --frozen-evidence-zip desk-residual-diagnostic-40404725134.zip --output desk-external-pose-audit.json
    python -m pytest -q tests/python/test_desk_external_pose_audit.py
    sha256sum rgbd_dataset_freiburg1_desk.tgz desk-residual-diagnostic-40404725134.zip desk-external-pose-audit.json

Missing/mismatched source or invalid timestamp/role/pose emits a structured BLOCKED_NO_VALIDATED_POSE_REFERENCE JSON and exit 2. No simulated reference fallback. Synthetic tests exercise holdout leakage, wrong cohort, missing pose, degenerate training trajectory, bad quaternion, and timestamp gaps.

## Falsifiable next step and boundaries

This is only a **pose-only first stage** of the pre-registered experiment. After independent code/coordinate-convention review and an actual pose reference run, a separate read-only fixed-Mesh comparison may report the unchanged 12-frame measured-depth pixels, common/lost/new overlap and paired SSE. If external poses do not materially improve paired SSE and coverage without discarding original pixels, the hypothesis that estimated holdout pose alone dominates the residual is weakened. Any correction selected on these holdouts requires a different frozen scene for validation.

Original desk V2 remains **93.7357% coverage, 16.7779cm RMSE, 15cm FAIL**. Synthetic CI is not a real mocap experiment or independent-human review; no GPU, RGB-only C-path, geometry improvement or external surface truth is claimed. Rollback: close Draft, leave main untouched.
