# Video-only COLMAP SfM experiment (Issue #11)

This **experimental** command does not produce `echo-dataset-1`, calibrated meters, depth, a mesh, or person reconstruction. It extracts RGB frames from a video and asks installed COLMAP to estimate an arbitrary-scale camera trajectory.

## Prerequisites

- Python environment installed as in the root README (including NumPy, SciPy, Pillow).
- FFmpeg and ffprobe in PATH.
- A COLMAP build with CPU SIFT extraction and matching.
- Video containing a mostly static, textured scene and camera translation; success is not guaranteed. Obtain consent before recording people.

```bash
python -m echo4d.video_sfm input.mp4 --output outputs/video-sfm --max-frames 100
```

The destination must not already exist. Inspect `report.json` even when the command fails. If registration works, `trajectory.json` contains registered filenames, decoded source video PTS, and camera-to-world matrices in **row-major JSON arrays**, with `translation_unit=unknown_colmap_units` and `metric_scale_known=false`. The source video SHA-256, command argv/status, selected frame indices and source timestamps are recorded. Original timestamps describe decoded PTS; PNG images have no media timestamps of their own. Image naming and frame selection are deterministic for a particular decoded video/FFmpeg build.

Note: no camera calibration/undistortion or external metric scale is inferred, and lens distortion can reduce quality. CPU COLMAP can take substantial time; feature extraction/mapper may fail. The result must not be fed to the metric TSDF importer without independent, registered metric depth and externally measured scale. An evaluation alignment scale is **not** a measurement of scale by this tool.

## Verification

```bash
python -m pytest tests/python/test_video_sfm.py -q
```

These tests mock executables to validate failure handling, **not** real SfM. Real RGB video and independent TUM ground-truth alignment are separately required to satisfy the whole of Issue #11. Do not call CI success a reconstruction quality result.

## Real TUM RGB-only benchmark (additional independent stage)

The Actions workflow at `.github/workflows/tum-video-sfm.yml` is designed to:

1. Download the original TUM `freiburg1_xyz` archive and enforce the pinned SHA-256 from Issue #7.
2. Select **source RGB frame indices 120–239** (fixed before running), re-encode them at 30 fps as a derived MP4, and record the source frame names and SHA-256. This is a real-image experiment, **not** a native phone video and not original capture timing.
3. Execute FFprobe → FFmpeg → **actual CPU COLMAP** from those RGB frames, without giving the estimator either the TUM known poses or depth.
4. Only on successful SfM, interpolate the original independent TUM ground-truth trajectory to the selected RGB source timestamps and fit a single post-hoc Sim(3) transformation for evaluation. Emit ATE position RMSE (meters), orientation mean/p95 error (degrees), registration ratio and pair count.
5. Upload the real `report.json`, `sfm-report.json`, `metrics.json` and `trajectory.json` where available. **If COLMAP fails, the workflow must fail and preserve an explicit failure report rather than pretend success.**

To reproduce on an Ubuntu system with FFmpeg and COLMAP installed:

```bash
python -m pytest tests/python/test_video_sfm.py tests/python/test_sfm_evaluation.py -q
python tools/tum_video_sfm.py --work outputs/tum-real-work --report-dir outputs/tum-real-report
```

Both destinations must not exist. The process downloads roughly 448 MB of official TUM data, extracts frames locally and can consume substantial CPU time. The pipeline may fail on a particular COLMAP build; review actual Actions logs instead of assuming success from the presence of code. GT-derived Sim(3) scale must never be copied into the unknown-scale SfM output or passed as a separately measured physical scale to the reconstruction adapter.

Limitations: a single sequence is not an independent evaluation of true dense geometry; scale- and rigid-aligned camera positions are not a validated mesh. ATE after Sim(3) uses the **evaluation GT** and is not an online positioning error guarantee. A genuine phone video with independently measured scale and depth remains a separate validation task.
