# ECHO4D integration receipt · 2026-10-27

Scope: receive verified existing work, retire duplicate branches, and close completed historical experiments. No new UI/model pipeline or reconstruction quality claim. User explicitly authorized ready/merge/close; no self-approval, main force-push, or required-check bypass. Initial live main was `e1d3c05a9a1c915b9906f494c25dbc2b2b50b90e`, verify40477803845 successful. Live branch API reported no protection/required contexts, rulesets empty; `verify` was still checked before every code merge.

## PR decisions (all 22 initially open PRs)

| PR | Decision / receipt | Evidence or remaining blocker |
|---|---|---|
| #8 | Received through #10; close/archive, retain original head685dd923 | Core renderer was already reused; original heldout runner, report, raw Mesh and PNG were still unique and are now in main. |
| #10 | Merge db214c5d, reviewed head415fe8ac | Main renderer retained byte-for-byte. Local252 Python+13 Node; verify40490368196, real heldout40490368042, tail40490368070 success. Downloaded tail ZIP SHA305fe6d3… and raw Mesh SHA16aa524c… verified; original11.331471cm preserved. |
| #16 | Receive through updated #18 | Historical design has value; labelled 2026-10-25 snapshot, never represented as new execution/current PR states. |
| #18 | Receive workbench/templates, current handoff and original evidence archive | AGENTS/WORKBENCH stale active dependencies replaced, historical text preserved. Documentation only; final receipt/CI is linked in #47. |
| #38 | Receive through #60; archive old stacked PR | Original V1 source failure preserved; implementation not lost. |
| #52 | Close, superseded by #58/#60 | Duplicate post-SfM timestamp audit; documented row-ID/resource false-positive not promoted to formal gate. |
| #53 | Close, superseded by #58/#60 | Duplicate post-SfM audit; no reason to maintain a second official implementation. |
| #54 | Close, pre-SfM design subsumed by #58/#60 | Complete original68-role V1 BLOCKED JSON retained with original SHA85951e65… in Git; no duplicate script. |
| #58 | Receive through #60; archive old stacked PR | One cohort planner, source preflight, V1/V2 modes and strict downstream source matcher. |
| #59 | Receive through #60; archive old stacked PR | Failed literal-mirror substitution history retained; #60 supplies actual working mirrorlist fix. |
| #60 | Merge805a855b, reviewed head276740fc | verify40490944425 success, RGB40490944857 success. Real40490944940 completes Mesh but fails15cm screen honestly; this non-required scientific FAIL does not block receipt of a safe diagnostic implementation. No threshold changed. |
| #43 | Receive through #69; archive | Original one-direction false-positive fixed before reception. |
| #64 | Receive through #69; archive | Reverse support gate retained; face-order sensitivity fixed by #69. |
| #69 | Mergefd175814, reviewed headb1f0eca6 | Two-sided CPU screen, same-index face/winding invariance, pre-allocation NPZ budgets. Local277 Python+13 Node; verify40491031246 / CPU40491031147 success. No GPU inference. |
| #44 | Receive through #56; archive | Explicit reviewed object intake/inventory retained; whole room is not called a furniture object. |
| #45 | Receive through #56; archive | Observed-only original-face extraction and real BLOCKED evidence retained; duplicate sources no longer count as independent views. |
| #46 | Receive through #56; archive | Frozen raw-source ledger and evaluator retained; observed-only pass cannot certify unsupported surfaces. |
| #55 | Close, choose #56 | Placeholder-color/derived-normal PLY adapter not received. Independent source rejection retained via #56. |
| #57 | Close, choose #56 | Same competing PLY approach; no duplicate official exchange interface. |
| #56 | Merge6f6f38a3, reviewed head5b549081 | Geometry-only vertices/faces NPZ, shared bounded decoder, D overall geometry INCONCLUSIVE/exit3 even if frozen forward metrics pass. verify40491457812 and dedicated workflows pass; C→D43 tests+8 mandatory checks, no skips. Latest desk combination local360 Python+13 Node. |
| #67 | Merge86b571ff, reviewed headf77f045a | Rigid-pose guard and analytical ray-plane oracle. verify40424245297 success; real B40424245285 raw ZIP CRC/digest/Mesh/report and48/12 isolation checked. Weighted RMSE agrees within1.1e-11m; no interpolation/threshold change. |
| #72 | Keep Draft, independent research value | raw rgb[::8] is not original V2 eligible-domain source cohort. Must read pinned source-cohort ledger and verify name/index/path/time row-by-row; add first17-unpairable regression. No real external-pose result accepted. Do not create another replacement PR. |

All retired PRs retain original commits/branches/discussions and receive an explicit successor/selection comment. A closed predecessor is not represented as an independent GitHub merge. Some closes occur automatically through ancestry; disposition is still documented.

## Experiments and issue ownership

| Issue | Decision | Scope and next owner |
|---|---|---|
| #7 | Close completed experiment | Two real format roundtrips; core/tools received through #34. Ordinary-input work continues #2. |
| #9 | Close completed descriptive phase after explicit transfer | Frozen tail distribution, baseline preservation, counterexamples and repeats completed through #10. Physical cause remains unproved; causal/time-block/cross-scene follow-up explicitly received by #15, external truth by #3. |
| #11 | Close completed RGB SfM baseline | xyz repeats and desk75/75 independent-scene runs retained in main. Native-phone/VFR and cross-device robustness continue #2/#15. |
| #14 | Close completed three-anchor experiment | Three real repetitions, seven non-anchor checks, no GT scale leakage; cross-device scale/dense-depth/generalization continue #15/#2/#3. |
| #2 / #3 / #15 | Keep OPEN | Ordinary-RGB predicted depth C, independent surface truth and full A/B/C quality gates remain unfinished. |
| #42 | Keep OPEN | GPU inference, real single-object identity, world calibration, independent dimensions and12 object masks unfinished. Actual verified furniture count remains0. |
| #47 / #1 | Keep OPEN | Coordination/charter, not finite implementation tickets. Preserve recording+past reconstruction, first-person+free third-person, evidence/generated separation, complete reconstruction priority. |
| #4 | Keep OPEN | Full technical selection/cost/license/runtime decision report still lacks acceptance evidence. |

Integration Maintainer owns integration/acceptance and next handoff. Research contributors have not been assigned identifiable natural-person ownership; do not fabricate it.

## Historical failures and reproducibility

- V1 first three frozen RGB roles precede available depth;582.392/314.358/50.430ms gaps remain BLOCKED at30ms. Original68-row audit is archived unchanged.
- Original V2:4543 vertices/7578 faces,93.7357% coverage,16.7779cm RMSE, **FAIL at15cm**. #61 evidence reception did not originally receive the implementation; #60 now does.
- Integration rerun:4572 vertices/7669 faces,94.1953% coverage,16.5386cm RMSE, **also FAIL**. Same source-cohort-plan bytes; fresh COLMAP solve causes run variation. It is not proof of an integration-caused accuracy improvement and cannot replace the original result.
- Both original and repeat raw PLY, camera tracks, full raw report,12-row holdout report, source plan/protocol and per-file SHA receipts are preserved at [evidence/integration-20261027](evidence/integration-20261027/README.md). Original archive bytes were checked against #61's existing receipt. No report was regenerated or edited.
- CPU/synthetic/format/CI success is not independent geometry or GPU success. Prediction, generated appearance, placeholder properties, scale claims and raw sensor observations remain separate.

Final main SHA, final four-job verify and the authoritative remaining-PR count are recorded in the final #47 handoff after this documentation PR is merged.
