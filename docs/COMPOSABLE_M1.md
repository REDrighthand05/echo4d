# ECHO M1: finite typed chessboard for research paths (branch-only)

This independent development branch implements typed, bounded PLANNING, not a new reconstruction pipeline or a claim of improved Mesh quality. Link it to the #75 M0 catalog after separate review.

- ArtifactType makes kind, frame, scale, clock, evidence class and training/heldout split part of exact port identity. No implicit unit, reference-frame, timestamp or evidence-grade conversion.
- ModuleSpec declares input/output and budget. Its installed flag is a caller assertion, not verified source/runtime. Build-phase adapters cannot consume heldout observations.
- compose enumerates alternate paths with stable graph IDs and limits for cost, steps, state count and routes; paths are always not_executed. No actual model runtime, data provenance verifier, GPU execution or independent-scene truth is implemented.

## Local verification

    python -m pytest -q tests/python/test_composable.py
    python tools/verify.py

Synthetic tests cover algorithm alternatives, metric/relative depth, incompatible timestamps, pose frames, generated-vs-measured geometry, holdout leakage, invalid metadata, loops, cost and search bounds.

## Deferred, not completed

1. M0→M1 bridge: #75 catalog and checked artifact sources; only a real registry may mark adapters installed. File SHA, calibration, model license and runtime still need verification.
2. M2: execute at least two distinct real module graphs on one frozen dataset, preserve failed runs, original images/Mesh, source and checkout hashes, GPU/CPU memory and timing.
3. M3: #74 source-image and 3+2 actual Mesh views, geometry holdouts and cost Pareto comparison. Historical desk V2 16.7779cm vs 15cm stays FAIL; no holdout-driven tuning.

Developed as an independent Git branch from main, not as an edit to PR #75.
