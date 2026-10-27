# GPT D permanent source-verification receipt (2026-10-26)

This records **actual GitHub Actions data processing**, not a generated Hi3D furniture experiment.

- Exact tested branch commit: `b4249ca09f451409126de4a294b2d801c9ccb28f`
- [Real TUM archive verification + 17 tests: run 40321455499](https://github.com/REDrighthand05/echo4d/actions/runs/40321455499): completed/success.
- [Full repository verify: run 40321455571](https://github.com/REDrighthand05/echo4d/actions/runs/40321455571): completed/success, Windows/Linux x Python 3.12/3.13 all 4 jobs success, each 204 Python passed + 1 skipped, 13 Node passed.
- [Original exact Actions report archive #12648119432](https://github.com/REDrighthand05/echo4d/actions/runs/40321455499/artifacts/12648119432), ZIP 4,516 bytes, SHA-256 `fe393c061b42daa2685ff07ab1f58a825259d2c975bbaef4431c83151cdd4d0f` (21-day GitHub retention).
- Inside original ZIP:
  - `historical-ledger-check.json` SHA-256 `a798173c61402e919e7a9af4e599d8c61a2f849146e5f192c25ca6b8b1b5fa60`.
  - `source-reverified.json` SHA-256 `5f1ac23131d7945fa9eba8e4dcbbef849900f3be375417fc1af3163203390a75`.
  - `source-download.json` SHA-256 `9c4bc4aa37ea7db4632a2d6965bd0f2873d54e7612ee026b01a2325b50a3566e`.
- Frozen ledger bytes SHA-256 `5245bc0ae4415f528dd2a7b679ead34388c636f3a54bbde32a825e1c2220d2e8`.
- TUM original public archive SHA-256 `a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98`; size 448,204,271 bytes.
- Verified from **new download**, with actual extracted raw file bytes matching historic hashes: **12/12 original holdout RGB PNG**, **12/12 original holdout depth PNG**, original raw `groundtruth.txt`.
- Raw TUM groundtruth SHA-256 `aac0319a6ef4e1cdf61e779d2152b95aa7e9f7b1749d6d18717b43ddabffede2`.
- Source result code: `SOURCE_BYTES_VERIFIED_BUT_FURNITURE_IDENTITY_BLOCKED`.
- Durable pose samples transcribed from actual ZIP at [verified-mocap-nearest-12.csv](verified-mocap-nearest-12.csv), joined to the [per-file photo-depth ledger](xyz-frozen-source-ledger.json) by frozen frame name. The samples are **raw nearest-time motion-capture** values, not an interpolated/calibrated camera-to-RGB pose; holdout image timestamps are not re-encoded video timestamps.
- Precondition failures: **0 verified single furniture**, **0 real object masks**, **0 independently physically measured dimensions for same furniture**, **0 real Hi3D GPU runs**, **0 physical improvement claim**.

The full GitHub Actions report includes per-frame camera sample timing, translations/quaternions and raw image-depth SHA. The file hashes and frozen ledger here preserve a reproducible audit even after the GitHub artifact expires; re-run the dedicated workflow to independently confirm raw bytes. Do not use the whole-room ECHO RMSE as a furniture-specific Hi3D improvement metric.
