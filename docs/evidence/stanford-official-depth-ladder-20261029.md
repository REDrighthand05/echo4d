# Stanford Lounge: measured-depth fusion ladder (2026-10-29)

**Real execution:** [Actions #40878621910](https://github.com/REDrighthand05/echo4d/actions/runs/40878621910), [GLB/3-photo + 2-free-view artifact](https://github.com/REDrighthand05/echo4d/actions/runs/40878621910/artifacts/12899172510). This is a **visible geometry experiment, not a full-life-replay PASS**.

**Source baseline:** the actual 3000-frame Open3D 0.19.0 complete reconstruction from [#114 run #40865675012](https://github.com/REDrighthand05/echo4d/actions/runs/40865675012/artifacts/12896335330), pinned upstream `1e7b17438687a0b0c1e5a7187321ac7044afe275`. That run generated a colored 5,635,827-triangle GLB and 3000 estimated poses before its *reporting adapter* failed: `data.color_paths` returned 2999 entries although the native RGB/depth directories and trajectory each contain 3000. The first ladder trial [#40877615978](https://github.com/REDrighthand05/echo4d/actions/runs/40877615978) also failed because numeric native filenames were wrongly treated as zero-based trajectory ordinals. Fixed by verified ordered RGB/depth/pose pairing; both failures remain recorded.

**Input boundary:** all 3000 real native 640×480 RGB frames are retained; the candidate uses only **3000, 600, or 150 real measured-depth frames** in fusion. No preexisting Mesh enters sparse integration; no predicted depth or GPU inference. Crucially, **all sparse runs reuse the camera trajectory estimated from the full 3000-frame RGB-D recording**. This is a *fusion-stage* depth ablation with a high-information pose prior, not a valid 600/150-frame independent capture pipeline and not RGB-only reconstruction. All use PrimeSenseDefault, depth_scale=1000, depth_max=3m, TSDF voxel=3/512m and sdf_trunc=0.04m.

**Actual visible result:** original photos and all three Mesh levels show recognizable armchairs, round table, books, plant, wall and floor. With 600 measured-depth frames (80% fewer), the three recorded views remain very similar to the full baseline; 150 frames (95% fewer) still retain recognizable furniture, but the first recorded view loses more upper-wall geometry and both free views show larger holes. None is a continuous, closed room; unseen walls and free-camera coverage remain poor. Identical world-space free-camera rays are used for all models. The following pixel coverage is only a diagnostic of the rendered five views, **not** surface accuracy:

| View | Full 3000 | 600 measured | 150 measured |
| --- | ---: | ---: | ---: |
| Recorded 0 | 90.2% | 88.4% | 82.8% |
| Recorded 1500 | 99.9% | 99.9% | 99.9% |
| Recorded 2999 | 100.0% | 99.9% | 98.7% |
| Free 1 | 30.3% | 28.4% | 24.1% |
| Free 2 | 15.2% | 14.5% | 12.1% |

**Real Mesh files:** `room-full.glb` in upstream artifact; `measured-600.glb` and `measured-150.glb` in the ladder artifact. Final image: `source-full-600-150-3plus2.png`. The 600/150 GLBs were independently opened with trimesh, preserving vertex colors. Echo4D's own browser viewer has **not** yet been verified.

**Next short engineering step:** on the same scene, keep the frozen pose prior explicit and replace a *small known subset* of measured depth with CPU-computed RGB depth (with metric calibration only from the retained measured frames). Then re-fuse and compare exactly these five views; do not quietly borrow withheld measured depth.
