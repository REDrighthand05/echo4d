# V2 source-paired preregistration — 2026-10-26, before its first run

This candidate is based on main `22551bc` and reuses the #38/#51 experiment
and #53 timestamp diagnostic. It changes source availability selection in a
separately named experiment, not the old V1 result or its acceptance thresholds.

The original V1 run40332593504 failed before scale/fusion. Downloaded and
SHA-verified #53 run40334987283 audit reports 68 roles: 65 within30ms, 3 outside,
2 repeated nearest-depth candidates. The first three RGB names have nearest
depth offsets +0.5823919773, +0.3143579960, +0.0504298210 seconds. This is source
availability evidence; no heldout geometric error has been observed or optimized.

## Fixed V2 source selection

1. Verify the same official344011403-byte archive, SHA256
   `e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c`.
2. Read only original `rgb.txt` and `depth.txt`. Reject invalid times, paths
   and reused source identities. Do not open RGB/depth pixels or groundtruth.
3. In original RGB order, retain entries whose nearest original depth timestamp
   is within the unchanged30ms window. No time shift is fitted.
4. Select entries0,8,16,... from this eligible list, up to exactly75. Selected
   depth sources must be unique. Fewer than75 remains FAIL, never a reduced run.
5. Publish the complete source plan, both original index hashes, original RGB
   indices/timestamps/paths and matched depth identities, then freeze its SHA
   into the protocol **before video encoding or COLMAP**.
6. Preserve dense-cohort roles:48 training,12 geometry holdouts, anchors15/45/65,
   diagnostics60/61/62/63/64/66/67. Preserve calibrated PINHOLE inputs, raw/5000m,
   7cm TSDF, training-only bounds,80% coverage and15cm RMSE research screen.
   GT is still evaluation-only; no heldout depth pixels select the cohort.

V1 remains the default CLI mode with its original source selection and fails
early if its pairing is impossible. New workflow explicitly invokes:

```bash
python tools/tum_desk_full_mesh.py --cohort v2-source-paired \
  --work /tmp/desk-v2 --report-dir artifacts/tum-desk-full-mesh
```

The V2 result is pending actual public-data execution. Never combine it with
V1 scores or label it a repair of V1's original failure. Source pairing PASS
alone does not establish scale, Mesh quality or external surface truth.

---


## Historical V1 protocol and failure audit

# PR #34 / D — TUM freiburg1_desk second-scene full Mesh protocol

This is a **new real-scene dense geometry experiment candidate**, not a continuation of `freiburg1_xyz` scores. Existing desk evidence [#21](https://github.com/REDrighthand05/echo4d/pull/21) proves only RGB SfM camera registration (75/75 twice), **not metric geometry**. The input is real TUM RGB images re-encoded at 4fps rather than native phone video. Physical Z depth is supplied by TUM's depth sensor, not inferred from RGB. Withheld scores reflect same-sensor consistency, **not independently scanned geometric truth**.

## Pre-registered roles (frozen in PR #34 D comment *before experiment*)

- Scene: `freiburg1_desk` official HTTPS `rgbd_dataset_freiburg1_desk.tgz`; self-observed reproducibility pin **344011403 bytes**, SHA256 `e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c` (not vendor signature).
- RGB: original images at source index stride 8, exactly 75 `frame_00000..00074.png`, re-encoded as 4fps test MP4; CPU COLMAP camera `PINHOLE 525,525,319.5,239.5`. Demand all 75 register. No GT pose or depth as SfM input.
- Geometry cohort: first 60 RGB images; exactly **12 withheld** indices `4,9,14,19,24,29,34,39,44,49,54,59` and remaining **48 sensor-depth training** images.
- Three scale anchors: indices `15,45,65`; seven **separate scale diagnostics**: `60,61,62,63,64,66,67`. Anchors 15 and 45 intentionally may overlap *training*, but neither anchors nor scale diagnostics overlap the final 12 geometric holdouts.
- Measured Z: raw TUM 16-bit PNG counts / **5000** = metres. Match source timestamps within 30ms; duplicate/missing sensor measurements fail, with **no after-the-fact frame replacement**.
- Mesh settings: RGB/depth downsample by 4, training-depth-only P2/P98 bounds + 0.3m margin, 0.07m voxels, <= 1,500,000 voxels, maximum depth 4m, minimum two observations. Camera translations get one measured SfM-to-metre scale factor; already-metric depth does not.
- Withheld scoring: exactly 12 sensor-Z frames not used to set scale, training, or bounds. Log per-view original depth file hash, timestamps, valid/overlap pixel counts, coverage, MAE, overlap RMSE and P95. Fixed exploratory screening: >=80% coverage and <=0.15m overlap RMSE. **Failing is a reportable scientific result; never tune the frozen parameters to pass.**
- GT availability: prior desk report lists all 75 pose references; independently re-check after Mesh scoring. GT poses do not enter RGB SfM, calibration, mesh, or depth scoring. Do not compare numerical xyz-vs-desk mesh scores as if these are one frozen dataset.

## Repeatable run

Requires repository locked dependencies, CPU COLMAP and FFmpeg; allow space for the 344MB original archive, extracted RGB+sensor data and mesh results. The work directory must not already exist.

```bash
PYTHONPATH="$PWD/tools" python -m pytest -q tests/python/test_tum_desk_full_mesh.py
python tools/tum_desk_full_mesh.py --work /tmp/desk-full-work --report-dir /tmp/desk-full-report
```

An optional `--archive path/to/official.tgz` allows offline repeat using byte-for-byte SHA validation. GitHub Actions workflow `.github/workflows/tum-desk-full-mesh.yml` records actual checkout/commit, preserves successful artifacts **and failure details**, and retains them for 14 days. Outputs when available: `frozen-protocol.json`, `measured-depth-evidence.json`, `metric-trajectory-from-sensor.json`, `desk-48train-12holdout.json`, `mesh/mesh.ply`, `desk-heldout-metrics.json` (original 12 per-view rows), `report.json`, `SUMMARY.md` and file hashes. Archive Mesh plus machine evidence separately before 14-day expiration.

This standalone Draft PR is a proposal against #34 integration branch, **not** a merge into main. Results remain unverified until the real workflow finishes. Failure causes include download/hash, missing depth/GT, insufficient sparse measured matches, oversized depth-derived volume, empty mesh and withheld research-gate failure.


## Failure isolation: fixed RGB vs sensor depth timing (Issue #47)

Actual [desk run #40332593504](https://github.com/REDrighthand05/echo4d/actions/runs/40332593504)
extracted and registered **75/75 RGB** with COLMAP, but the subsequent measured
depth association failed at fixed `frame_00000.png`: `Missing sensor Z within
30ms`. The original run did not save the signed closest-depth offset, so
no numeric offset (and no timing correction) can be asserted from that log.

`sensor_pairing_diagnostics()` now records **all 68 frozen roles** (48
geometry-train + 12 geometry-holdout + 3 anchors, two of which overlap training,
and 7 separate diagnostics), listing the nearest original sensor timestamp,
signed/absolute gap, whether it meets the **unchanged 30ms** gate, and reuse
of the same sensor file. Before the strict `match_sensor_inputs` call, the
runner writes `sensor-pairing-preflight.json`; the nested SfM run and
original TUM archive hash remain unchanged. No held-out pixel values are opened
by this timing-only audit.

**The timing audit is not automatic realignment.** A frame outside 30ms still
fails; it is not reselected, shifted, interpolated or substituted to make
the reconstruction pass. If the desk cohort really cannot satisfy its frozen
input contract, keep the old failed experiment as a negative result. Any
future different timestamp tolerance, frame cohort or calibrated clock offset
requires a **new, independently justified and predeclared experiment**, with
its own identifiers and independent evidence; never retroactively alter this
one. Index-only audit PASS is not a reconstructed room or Hi3D inference.

