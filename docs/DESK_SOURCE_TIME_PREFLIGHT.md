# Desk V1 original timestamp preflight (#47 / #38)

This is an **index-only read** of the SHA-pinned **real** TUM Freiburg1 Desk archive.
It verifies original `rgb.txt` and `depth.txt` and lists all 68 distinct source-frame
identities used in the already frozen 75-frame/48-train/12-holdout + anchors/diagnostics
experiment. It does **not** run FFmpeg, COLMAP, depth pixel processing, pose scoring,
metric scaling or 3D reconstruction.

The deterministic source selection is **exactly** the existing
`rgb_rows[::8][:75]`. It checks the **unchanged** 30 ms threshold and original
raw source timestamps; it does not use the artificial 4 fps video timestamps.
Each row reports original RGB name/time, closest measured depth name/time,
signed and absolute difference, accepted/blocked, and source reuse.

If the frozen V1 selection has any missing or conflicting measured Z:
- report `BLOCKED_FROZEN_V1_INDEX_PAIRING` and return exit 2;
- preserve V1's original failed run, role identities, original time limits and
  screening gates unchanged;
- only if justified by source availability, propose a **new** pre-registered
  V2 cohort/split in a separate PR *before* running geometry. This tool
  does not select V2 or suggest retrospectively choosing frames based on
  depth values or SfM quality.

```sh
PYTHONPATH="$PWD/tools" python -m pytest -q tests/python/test_tum_desk_sensor_preflight.py
python tools/tum_desk_sensor_preflight.py --work /tmp/desk-index-work --report-dir /tmp/desk-index-report
```

An offline archive may be given with `--archive /absolute/file.tgz`, but its
SHA and byte length are checked against the existing pin.
Source archive download is the only large operation; no GPU, COLMAP,
FFmpeg or source-image decode needed. Original RGB/depth pixel bytes,
TUM GT pose, and independent measured geometry are never used by preflight.

**Neither PASS nor BLOCKED measures the precision of a reconstructed room.**
