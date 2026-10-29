# Real Stanford Lounge: prior-shell experiment — TWO negative outcomes

**2026-10-29; no visual improvement.** Original images are real 640x480
Stanford Lounge RGB-D with logged poses. The source 426,789-face mesh is
independently measured. The 169,999-face existing scan preview is a
**PRE-EXISTING PRIOR**, never independently recovered from video.

1. Replacement run [40827393230](https://github.com/REDrighthand05/echo4d/actions/runs/40827393230):
   [five actual A/B views and GLBs](https://github.com/REDrighthand05/echo4d/actions/runs/40827393230/artifacts/12877640450).
   Removed 74,641 measured triangles and added 51,441 prior triangles.
   Human inspection: new dark tears/speckles in walls/furniture; free
   views essentially unchanged or worse. Do NOT publish as an improvement.
   The geometry and all five real views succeeded; the separate Git push
   failed because an unstaged working-tree change blocked rebase.
2. Observed-first run [40827876660](https://github.com/REDrighthand05/echo4d/actions/runs/40827876660):
   protected all 426,789 measured triangles and accepted only broad outer
   prior shell faces with all vertices >7.5cm from measured support.
   **Failed closed:** fewer than 100 genuinely unsupported faces were
   available under the conservative guard; no new hybrid Mesh was
   published. This is not a renderer or test pass. Both geometry guard
   tests passed.

**Decision:** Do not iterate these thresholds indefinitely. Next geometry
experiment should use the original ~3.15-million-face registered reference
instead of the already simplified 170k preview, preserving observed faces.
If that also yields no genuinely new shell, retire this prior-assisted route
and try a different scene/data source. #74 and Desk V2 FAIL unchanged.
