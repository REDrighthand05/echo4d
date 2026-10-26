# Frozen desk RGB ↔ depth timestamp audit: actual negative result (2026-10-26)

**Evidence source**: [real GitHub Actions run #40334987283](https://github.com/REDrighthand05/echo4d/actions/runs/40334987283), original [artifact #12656318028](https://github.com/REDrighthand05/echo4d/actions/runs/40334987283/artifacts/12656318028). Original source: TUM `freiburg1_desk` archive exactly **344,011,403 bytes**, SHA256 `e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c`. This is a self-observed pin, not a signed publisher statement. Evaluated checkout: GitHub Actions merge SHA from `report.json`, head `e10921dbfc693b510091114b48c6d9e51d51cfe4`.

## Facts from actual machine outputs

- RGB-only SfM: **75/75** frames registered, 7,013 sparse points. This is not yet a dense/metric reconstruction.
- Original raw `depth.txt`: **595 timestamp rows**.
- Frozen data roles: **68 unique source RGB images**. **65/68** had depth within unchanged ±30ms threshold; **3/68** did not.
- Out-of-window pairings (all at the very start of the source recording, and all `geometry_training` frames):
  - `frame_00000.png` original RGB `1305031452.791720`, nearest measured depth `1305031453.374112`: **+582.392ms**.
  - `frame_00001.png` original RGB `1305031453.059754`, nearest depth `1305031453.374112`: **+314.358ms**.
  - `frame_00002.png` original RGB `1305031453.323682`, nearest depth `1305031453.374112`: **+50.430ms**.
- `frame_00003.png` nearest depth **+13.674ms**; `frame_00004.png` withheld has **+9.913ms**. The nearest measured depth index for the first three RGB frames happens to be the same sensor file; audit reports two *duplicate nearest candidate* warnings, not two valid matches.
- Median absolute gap **12.566ms**; 95th percentile **18.534ms**; max **582.392ms**. All 12 frozen geometric holdout frames, all 3 scale anchors, all 7 scale diagnostics and the 43 training-only roles meet 30ms. Training set itself has 48 frames, but 2 of them have priority role label `scale_anchor`, so the printed role partition is 43+3 invalid training +2 anchor +12 holdout +7 diagnostic +1 anchor outside first 60 = 68.
- `sensor-pairing-preflight.json` SHA256 `b0f2b75ce93f5dd184dd20b90352b9f3c18dae98f8b4766b1c6d1946f4e33aed` (32,060 bytes). `report.json` SHA256 `3979661be35b52830dff4868ca79162633b1f9e83a9a43e589aa91d1b5f90af2`. `frozen-protocol.json` SHA256 `b7aa185fe62050c940537c18ddcb34dd0a189db59479126fe55272b12c938a16`.
- GitHub Actions job overall **FAIL by design**: `DataError: Missing sensor Z within 30ms for fixed frame_00000.png`. No measured metric scale, training Mesh or 12-view mesh accuracy score has been produced. Both original #38 and diagnostic #53 MUST remain labelled failed as to full mesh.

## What the evidence means

The observed problem is source-depth coverage at the **first three fixed RGB frames**, not a measured constant clock offset requiring a +582ms global shift. **Do not correct all timestamps by the first missing frame's distance.** Both the original 30ms gate and the frozen 48 train / 12 holdout identities remain unchanged in this branch. The index audit opened **no withheld depth pixels** and cannot itself prove geometric reconstruction accuracy.

Potential *separate* new trial (not part of #38 or #53; requires its own preregistered version and reviewer acceptance): keep all 75 RGB for SfM and the original 12 frozen holdouts/scale anchors/diagnostics, but mark the first three RGB frames as `RGB_ONLY_NO_SENSOR_DEPTH` and use **45 depth-training frames rather than 48**. Explicitly call this a new source-availability cohort, disclose that it was selected **after examining the failed timestamp audit**, and never describe it as the frozen 48/12 result or change the numerical geometric screening after seeing outcomes. Check all per-frame timestamps against the unchanged ±30ms rule before geometric work.

**Scope**: Timestamp index audit and provenance only, no raw photos/depth images copied into Git and no claim that a historical room's unobserved surfaces have been reconstructed.
