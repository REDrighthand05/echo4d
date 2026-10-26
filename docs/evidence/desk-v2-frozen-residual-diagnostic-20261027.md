# ECHO desk V2 — frozen 12-view residual diagnosis (real CPU run, original mesh)

**This is a scientific diagnostic of a FAILED reconstruction screen, not a quality PASS.**
The exact protocol was [preregistered before inspecting per-pixel residual arrays](https://github.com/REDrighthand05/echo4d/issues/15#issuecomment-6427704181). This result was produced from the original TUM RGB-D archive (SHA256 `e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c`), original **unchanged** V2 scene Mesh (SHA256 `320da61fcd421c0e55aa6fc014750aeaa3e6e2d98b607e9feaae268f83d2a939`), and exact 12 reserved sensor depth PNGs, with no COLMAP rerun, GT poses, geometry adjustment, source-frame reselection, re-scaling, or discarded measurement.

- Diagnostic GitHub real run [40405207741](https://github.com/REDrighthand05/echo4d/actions/runs/40405207741): **completed/success**. Historical run reused: [40356324356](https://github.com/REDrighthand05/echo4d/actions/runs/40356324356).
- Diagnostic raw artifact: [12690047819](https://github.com/REDrighthand05/echo4d/actions/runs/40405207741/artifacts/12690047819), ZIP SHA256 `c300dae88861744a920299e0290b415b4d601ec1e070b1976adbb92de975ecbc`; contained `result.json` SHA256 `f6817da95c4eb5cb70001dcac30ddbe1720e0f9fa3126a1d4470f61ca62bb1a3`. Archive may expire; save original ZIP separately if re-auditing exact numbers is needed.
- Machine check: **169,467** original measured-valid pixels; **158,851** original Mesh-overlap pixels; coverage **93.7356535%**, MAE **5.7896181cm**, overlap RMSE **16.77790247cm**. Every one of these values matches frozen original V2 machine report (numerical tolerance 1e-6m).
- Fixed screen: coverage ≥80% **PASS**; RMSE ≤15cm **FAIL**; **overall V2 screening remains FAIL**.

## Recreated error magnitude histogram (all 158,851 original overlap pixels)

| Absolute residual, predicted Z minus sensor Z | Count | % of overlap | Share of total squared error |
|---|---:|---:|---:|
| <3cm | 96,737 | 60.90% | 0.490% |
| 3–5cm | 27,190 | 17.12% | 0.941% |
| 5–10cm | 21,398 | 13.47% | 2.300% |
| 10–30cm | 8,063 | 5.08% | 5.061% |
| **≥30cm** | **5,463** | **3.439%** | **91.208%** |

The large-residual tail dominates **squared error** (RMSE), not all pixels. It does **not** justify removing hard pixels from the original screen.

## Preregistered independent partitions

These are different descriptive cuts of the **same** pixel set, not independent evidence that can be added together.

| Measured distance | Original valid | Overlap | Coverage | RMSE on overlap | Squared-error share |
|---|---:|---:|---:|---:|---:|
| 0.15–1m | 73,330 | 72,699 | 99.14% | 9.53cm | 14.77% |
| 1–2m | 74,249 | 71,071 | 95.72% | 17.79cm | 50.28% |
| 2–3m | 16,402 | 12,541 | 76.46% | 29.16cm | 23.85% |
| 3–4m | 5,486 | 2,540 | 46.30% | 44.21cm | 11.10% |

| Screen / depth-change proxy | Original valid | Overlap | Coverage | Overlap RMSE | Squared-error share |
|---|---:|---:|---:|---:|---:|
| Central 60%×60% | 72,198 | 70,818 | 98.09% | 16.11cm | 41.11% |
| Outer screen region | 97,269 | 88,033 | 90.50% | 17.30cm | 58.89% |
| Sensor 4-neighbor valid depth jump ≥10cm | 7,185 | 6,325 | 88.03% | 36.25cm | 18.59% |
| No such depth-jump proxy | 162,282 | 152,526 | 93.99% | 15.45cm | 81.41% |

## Original individual heldout views (none excluded)

| Frozen holdout index | Measured valid | Overlap | Overlap RMSE | Share of global squared error |
|---|---:|---:|---:|---:|
| 4 | 14,021 | 13,077 | 29.8644cm | 26.083% |
| 9 | 14,088 | 13,245 | 7.5870cm | 1.705% |
| 14 | 15,367 | 15,327 | 5.2082cm | 0.930% |
| 19 | 14,219 | 13,595 | 13.9541cm | 5.920% |
| 24 | 12,751 | 12,678 | 7.7426cm | 1.700% |
| 29 | 12,817 | 12,369 | 13.3738cm | 4.947% |
| 34 | 15,295 | 15,295 | 4.5451cm | 0.707% |
| 39 | 15,162 | 14,806 | 8.1628cm | 2.206% |
| 44 | 13,213 | 10,627 | 22.6630cm | 12.206% |
| 49 | 13,812 | 13,372 | 19.4804cm | 11.348% |
| 54 | 14,050 | 10,700 | 25.3193cm | 15.340% |
| 59 | 14,672 | 13,760 | 23.4411cm | 16.909% |

## How to interpret, and next test

1. The **extreme pixel tail** is a real and quantitatively significant diagnostic finding, but may reflect occlusions, timing/pose, sensor noise, or incorrect geometry. Nothing here yet proves one cause.
2. The 3–4m observed distance bucket has particularly low predicted coverage (46.3%), while discontinuity-proxy pixels have high overlap RMSE (36.25cm). They should be investigated with fixed independent evidence and original camera/depth alignment, not automatically discarded.
3. Future *causal* checks should hold the original V2 Mesh/poses/thresholds and 12 holdouts intact; inspect representative extreme-error pixels in source RGB/depth for temporal inconsistency, moving objects and surface support **without choosing only easy views**. Any remedial algorithm change requires a new preregistered version.
4. Sensor-Z withheld from TSDF but collected by the same TUM device is not an independent survey-grade surface scan. These bins do not validate the RGB-only predicted-depth path, which remains unfinished.

Reproducible source: [Draft PR #66](https://github.com/REDrighthand05/echo4d/pull/66). Its independent dedicated real workflow and cross-platform verification must be reviewed at a specific head SHA before merging. This evidence file must not be treated as permission to change the original historic FAIL. 
