Desk V2 scientific residual diagnostics

Pre-registered BEFORE accessing pixel residual arrays in Issue #15 comment 6427704181.

Uses the original saved 12 held-out depth frames, original saved camera+mesh and metric track, and SHA-locked official TUM depth PNGs. Runs no SfM, no reconstruction, no pose refitting, no mesh changes.

Depth bins: 0.15-1, 1-2, 2-3, 3-4 meters. Image bins: center 60% on both axes vs outer. Edge proxy: any valid 4-connected sensor-neighbor changes >=10cm. Error bins: <3cm, 3-5cm, 5-10cm, 10-30cm, >=30cm. All 12 frames and every measured-valid pixel remain in the report. Same-sensor consistency only, never independent surface ground truth.

Before attributing error, the program MUST recreate the original per-frame and aggregate coverage/MAE/RMSE to within 1e-6m; otherwise it outputs DIAGNOSTIC_NOT_COMPLETED and does not publish conclusions. Original V2 RMSE of 16.78cm still FAILS frozen 15cm gate.

The original GitHub run artifact expires after 14 days. Save an independently reviewed result and original SHA outside the ephemeral CI artifact after running.
