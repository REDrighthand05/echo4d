# PR #34 / B — Source-PTS extraction gate (2026-10-26)

Status: **real local FFmpeg/FFprobe extraction tested; COLMAP registration comparison not yet measured**. This is an extraction behavior report, not a reconstructed-scene or geometry score.

## Frozen inputs and limits

- Baseline #34 head `d030f615aae94b70218badb28c29300da7894285`.
- Related PR #12 head `3315b6d14853d72076706d36a1c06f9be513e7da`: `fps=4` fixed-rate resampling.
- Related PR #13 head `dd1b4be5d8b85b4bdf1fe11661876c8e15d0cdb0`: decoded-frame-index selection with original decoded PTS.
- Local baseline tools: Python 3.13.5, FFmpeg/FFprobe 7.1.5 (Debian); COLMAP not installed locally.
- Images: 12 generated 128×96 PNG frames numbered 000–011, deterministic RGB backgrounds and white frame number. Encoded with libx264 CRF 20 yuv420p.
- CFR encoding: 4fps over all 12 frames, video SHA-256 `380c141984135fb79db67019b2a2741de3ab65a19de9d8871c13dbfa4cb6eab0`.
- VFR encoding: concat demuxer with successive frame intervals in seconds `[.08,.28,.20,.40,.12,.35,.15,.45,.10,.22,.32]`; SHA-256 `495ac47e69f204070938e7b22de8f822844384ec37c48442f7fb8f44dac68e1d`.
- Both test clips were probed as 12 decoded frames; no source camera intrinsics exist for generated flat-color frames. **Do not use these clips for COLMAP camera-position or mesh-quality claims.**

## Exact reproducibility recipe

From a clean folder with Python, Pillow and FFmpeg installed:

```bash
mkdir -p input
python - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw
p=Path('input').resolve()
for i in range(12):
    im=Image.new('RGB',(128,96),(i*20%255,i*41%255,i*67%255))
    ImageDraw.Draw(im).text((14,40),str(i),fill='white')
    im.save(p/f'{i:03d}.png')
durations=[.08,.28,.20,.40,.12,.35,.15,.45,.10,.22,.32]
lines=['ffconcat version 1.0']
for i in range(12):
    lines.append(f"file '{p/f'{i:03d}.png'}'")
    if i<11:
        lines.append(f'duration {durations[i]}')
(p/'vfr.ffconcat').write_text('\n'.join(lines)+'\n')
PY
ffmpeg -y -hide_banner -loglevel error -framerate 4 -i input/%03d.png -c:v libx264 -pix_fmt yuv420p -crf 20 cfr.mp4
ffmpeg -y -hide_banner -loglevel error -safe 0 -f concat -i input/vfr.ffconcat -fps_mode vfr -c:v libx264 -pix_fmt yuv420p -crf 20 vfr.mp4
for kind in cfr vfr; do
  ffprobe -v error -select_streams v:0 -show_entries frame=best_effort_timestamp_time -of json "$kind.mp4"
  mkdir -p "resampled_$kind" "source_pts_$kind"
  ffmpeg -hide_banner -loglevel error -nostdin -i "$kind.mp4" -vf fps=4 -frames:v 12 -start_number 0 -f image2 "resampled_$kind/frame_%05d.png"
  ffmpeg -hide_banner -loglevel error -nostdin -i "$kind.mp4" -map 0:v:0 -an -sn -vf 'select=not(mod(n\,2))' -fps_mode passthrough -frames:v 6 -start_number 0 -f image2 "source_pts_$kind/frame_%05d.png"
done
```

The commands directly exercise the old #12-style FPS filter and the #13-style decoded-frame-stride filter on the **same encoded inputs**. The source-PTS mode's new CLI uses that same select contract, with bounded metadata checking and the old output naming/time format.

## Observed time mapping and decoded source image selection

| Video | Decoded source PTS (s) | Resampled `fps=4` output | Source-PTS `stride=2` output |
| --- | --- | --- | --- |
| CFR | 0,.25,.5,.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75 | 12 frames; source indices 0–11 | 6 frames; indices 0,2,4,6,8,10 |
| VFR | 0,.08,.36,.56,.96,1.08,1.44,1.60,2.04,2.12,2.36,2.68 | 11 frames; source indices 1,2,3,3,5,5,7,7,9,10,10 | 6 frames; indices 0,2,4,6,8,10 |

The frame-index assignments above were determined by comparing pixel colors with all 12 uniquely colored source frames (background point away from overlaid text). In this deterministic sample, fixed-FPS resampling duplicates some source images and skips others. Source-PTS sampling emits each selected index once; its VFR inter-frame time gaps remain unequal. The modes deliberately produce **different cohorts** under the same `max_frames` ceiling; the observed counts do not demonstrate one COLMAP solver or 3D result is better.

## Downstream format & risks

- Current #34 public outputs remain: `trajectory-scale-unknown.json` (`frames[].name/t_s`, `camera_to_sfm_world`, `units=unknown`), `registered_timestamps.csv`, `all_input_timestamps.csv`, `report.json`.
- The new `--timestamp-mode source-pts` is opt-in. Default `resampled` intentionally keeps existing experiments intact. The added report carries `source_pts_sampling.frame_map` with decoded frame index and absolute/relative file PTS.
- #17/#23 consumers still receive relative seconds and predictable names; **names refer to different pictures between sampling modes**, so external measured-depth synchronization must be rechecked against the source timeline before use.
- Invalid/duplicate/nonmonotone decoded PTS, metadata over 8 MB, and mismatched frame output all fail rather than silently fabricating a timeline.
- A calibrated camera or actual sensor capture timestamps are not derived from PTS; reconstructed metric scale and surface accuracy remain unproven by this change.
- **Registration / sparse model / ATE / failures on real scenes**: not measured in this local synthetic extraction test because this environment lacks COLMAP. Follow-up gate: compare #34's real pinned TUM sample with the same file/known intrinsics/frame budget under both modes, report each extracted filename→source frame mapping and COLMAP registration/errors separately, and only compare solver quality when the actual input cohort is identical.

Related: PR #34 task B; original #12/#13. No changes to COLMAP multi-model selection in this candidate.
