# ECHO M1: finite typed chessboard for research paths (branch-only)

This independent development branch implements typed, bounded PLANNING, not a new reconstruction pipeline or a claim of improved Mesh quality. Link it to the #75 M0 catalog after separate review.

- ArtifactType makes kind, frame, scale, clock, evidence class, train/holdout split, **scene cohort ID, coordinate-domain ID and clock-domain ID** part of exact port identity. Spatial ports now additionally require explicit **right/left handedness**; pose ports require **camera_to_world/world_to_camera direction**; metric ports declare **m/cm/mm** and timed ports **s/ms/us/ns**. The planner rejects mismatches unless a separately registered typed conversion module is provided. Two streams both called `frame_pts` or `world` are not automatically synchronized/aligned. No implicit unit, reference-frame, timestamp or evidence-grade conversion. These declared IDs do NOT authenticate raw source SHA.
- ModuleSpec declares input/output, budget and a mandatory 40/64-hex revision pin. A route fingerprint covers selected module revisions and port contracts; this is **not** an experiment ID until input bytes, runtime configuration and checkout are independently pinned by M2. Its installed flag is a caller assertion, not verified source/runtime. Build-phase adapters reject heldout and unassigned ports.
- compose enumerates alternate dependency DAGs with stable graph IDs (not duplicate topological execution orders), prunes unrelated side branches and limits cost, steps, unique state count and routes; paths are always not_executed. No actual model runtime, data provenance verifier, GPU execution or independent-scene truth is implemented.

## Local verification

    python -m pytest -q tests/python/test_composable.py
    python tools/verify.py

Synthetic tests cover algorithm alternatives, metric/relative depth, **cm-vs-m and ms-vs-s unit mismatches, left-vs-right handedness and camera/world pose inversion**, incompatible timestamps and clock domains, coordinate-domain and scene-cohort collisions, revision fingerprint changes, pose frames, generated-vs-measured geometry, holdout/unassigned leakage, invalid metadata, loops, cost and search bounds. A declared conversion module is still a plan, not proof the conversion was run or correct.

## Deferred, not completed

1. M0→M1 bridge: #75 catalog and checked artifact sources; only a real registry may mark adapters installed. File SHA, calibration, model license and runtime still need verification.
2. M2: execute at least two distinct real module graphs on one frozen dataset, preserve failed runs, original images/Mesh, source and checkout hashes, GPU/CPU memory and timing.
3. M3: #74 source-image and 3+2 actual Mesh views, geometry holdouts and cost Pareto comparison. Historical desk V2 16.7779cm vs 15cm stays FAIL; no holdout-driven tuning.

Developed as an independent Git branch from main, not as an edit to PR #75.

## Integration context

This branch incorporates the merged #76 finite-chessboard policy from main f6959cbe. The M1 planner remains a separate synthetic-only module and is **not yet wired into** the 42-route catalog or the #75 CLI. The user's original purpose is reconstructing recognizable past lived spaces, not maximizing the number of enumerated rows or passing CI.

## M1 graph-identity regression (2026-10-28)

A previously reviewed counterexample showed independent `left_stage` and `right_stage` appearing as two routes solely due to execution order. The planner now tracks artifact producers and canonical typed dependency edges, deduplicates identical producer graphs, retains genuinely different providers/revisions, and excludes unrelated branches from route identity and estimated cost. `max_states` bounds unique enqueued states; truncated searches remain explicitly flagged. These are synthetic planner tests only; there is still no actual Mesh runner.
