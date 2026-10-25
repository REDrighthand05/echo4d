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
