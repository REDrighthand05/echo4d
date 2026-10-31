# ECHO promo pipeline — FROZEN (v3, 2026-10-31)

The complete, self-contained pipeline that produced the five approved
promo samples (`echo_promo_main_sample.mp4`, `echo_promo_t1..t4_sample.mp4`).
Frozen by user order after the v3 verdict round. It is **not** integrated
into the app UI; merging into the software comes later. This folder is
also the visual prototype for the future live process preview.

Run everything with the app venv python from this directory:

```
cd echo_scene_app/promo
../.venv/bin/python render_main.py full            # main promo
../.venv/bin/python render_track.py 1,2,3,4        # batch tracks
```

One command per video. Output lands in `promo/output/` (an optional
second argument overrides the output directory). Intermediates go to
`promo/work/`; generated data lives in `promo/assets/`.

## The frozen look (do not retune without a user verdict)

- **Detroit lattice**: a GLOBAL blue grid filling the whole space
  (independent of the scene's surfaces) plus meaningless scattered
  line shards. (`prep_grid2.py` → `assets/grid2.npz`; shards are the
  last 150 lines of `data.npz`'s grid.)
- **Photos are never black cards.** Hero photos and route cards are
  sampled in TRUE colour (no grade, no border blend), feathered with a
  smootherstep black-gradient edge, and materialise on their wall
  already lit and readable. Hero photos: 136×240 samples + quincunx
  fill; route cards: 48×84 + 3×3 fill.
- **Camera rhythm is strictly segmented**: TRAVEL (eased glide along
  the route) → ARRIVE & HOLD (camera settled at the photo's own 3/4
  watch pose *before* the photo appears; only a ±1% breathing motion)
  → photo lands and crushes *while the camera holds* → travel on.
  Every hero is watched from its own pose — poses are never merged;
  tight accent pairs get a short post-hold and a quick glide instead.
  Finale pulls back to the wide view.
- **Particles/crush are approved as-is** — the motion math is
  byte-identical to v2 (`pos = start + delta·p² + burst·sin(π·p^0.7)`,
  alpha ramp `p/0.15`, flight 0.75s). Do not retune.
- **Solidification swell （实体化）**: once a photo's points settle,
  each grows over 1.5s from a dot into a filled square splat at voxel
  pitch 0.11 (scene true colours, depth-tested against the base pass),
  so finished parts of the room read as solid matter while the rest
  still builds; by the finale the whole room is solid. This mirrors
  the product's point-cloud solidification.
- **End segment (B)**: finished scene → shrinks into a file icon
  labelled `voxel_model.glb` → cursor drags it into a Blender-logo
  icon → closing line 「完美流畅衔接 Blender、UE5 等开发流场景」 →
  credit 「本视频由 Muse 助手结合自研 Echo 渲染算子制作」.
- **T2 sentimental opening is verbatim**: warm grade over the whole
  segment plus the glowing lines 「回忆很珍贵」 / 「把这一段生活，
  好好留住」 for t < 4.6s (`overlays.py`). The other tracks are
  tech-style, no opening text.

## Sampling ladder (user rule)

1. 720p full-length SAMPLE video first (bitrate-capped, plays
   smoothly) — stills alone are never an approval gate.
2. Only after the user's OK: the 1080p final render.
   (The internal render is 1920×1080; samples downscale to 1280×720
   in the encode. A final = same command with the scale filter set
   to 1920×1080.)

## Encode rules (all videos)

- libx264, profile high, **crf 23, maxrate 5M, bufsize 2.5M**,
  preset medium, yuv420p, 30 fps, `+faststart`.
- **Audio is stream-copied, never edited or re-encoded**, and the
  mux is verified by comparing the audio packet MD5 of the output
  against the source track (`verify.py`).
- Verify every output: full decode (`-xerror`), duration vs its
  track, audio MD5 match, per-second bitrate peak.

## Files

| File | Role |
|---|---|
| `batch_lib.py` | Shared library: loads `assets/` data, geometry, lattice, text/halo helpers, `pick_pins`, easing; pulls `render_frame` from the app's Echo Render operator (`echo_super.kernels.echo_render`). |
| `batch_lib3.py` | The v3 template: beat-pinned hero schedule, per-hero hold/travel camera, lit photo/card layers, solidification swell layer. |
| `batch_b.py` | Segment B (scene → file icon → Blender drag → closing lines). Reads `assets/blender_logo.png` and the generated `assets/scene_still.png`. |
| `overlays.py` | Beat-map loader; T2 warm grade + sentimental opening overlay (verbatim). |
| `render_main.py` | Main promo driver. `stills` = self-check stills + scene still; `full [outdir]` = the video. Pins 4.32/7.62/10.02/13.62/14.81, A 17.22s, total 26.56s, music `experiments/promo_music.m4a`. |
| `render_track.py` | Batch driver: `render_track.py 1,2,3,4 [outdir]` or `still`. Per-track A durations / hero counts / drop accents are in `build_tracks()`. Music + beat maps from `experiments/batch_music/`. |
| `analyze_beats.py` | Beat analysis for NEW tracks (spectral-flux onset envelope + comb-grid tempo scan). See caveat below. |
| `verify.py` | Post-render verification (decode, duration, audio MD5, peak bitrate). |
| `prep_data.py` | Data chain 1: app job → `data.npz`, `frame_geo.json`, `stems.json` (via `echo_scene.preview`). |
| `fix_colors.py` | Data chain 1b: rewrite data.npz card/photo colours as true colour (already applied to the frozen `data.npz`). |
| `prep_surface.py` | Data chain 2: → `surface.npz` (surface wireframe + photo feather). |
| `prep_grid2.py` | Data chain 3: → `grid2.npz` (global Detroit lattice). |
| `prep_assets.py` | Data chain 4: → `v3_assets.npz` (true-colour hero photos 136×240 + route cards 48×84, feathered, with per-step world vectors). |

## Data & inputs

- `assets/` holds everything the renders read. The small files are
  committed; **`data.npz` (72MB) and `v3_assets.npz` (193MB) are
  git-ignored** (too large for git) but present on disk. Regenerate
  them with the prep chain above, in order, from the app jobs
  `jobs/room2_trim_acceptance` + `jobs/room2_trim2` (jobs are runtime
  data, also not in git). `assets/scene_still.png` is regenerated by
  the drivers on every render.
- Music and beat maps live in the app repo at `experiments/`
  (`promo_music.m4a`; `batch_music/track{1..4}_*.m4a` +
  `track{1..4}_beatmap.json`). The shipped beat maps are the
  **canonical** frozen inputs. `analyze_beats.py` re-implements the
  documented method and reproduces duration/tempo and the same onset
  peaks, but its accent selection is only approximately equivalent —
  never overwrite a shipped map with its output silently.
- Rendering uses the operator package in the app tree
  (`echo_super/kernels/echo_render`) — paths are resolved relative
  to this folder, so the pipeline works from the repo alone.

## Reproduction check (freeze proof, 2026-10-31)

Re-rendered track 4 end-to-end from this folder
(`render_track.py 4 output/`) and compared against the shipped
sample: same duration, clean full decode, identical audio MD5.
See the commit message / task report for the numbers.
