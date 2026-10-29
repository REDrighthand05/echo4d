# Echo4D integrated product delivery contract (2026-10-30)

This branch coordinates implementation, not a claim that all milestones are complete.

## Product definition
A person can import authorized, timestamped sensor recordings, reconstruct a recognizable historical room, revisit the recorded first-person view at a chosen time, switch to free third-person orbit, and reopen the same saved scene later. Unknown surfaces remain visibly unknown; any reference scan is labeled **prior**, never independently reconstructed evidence.

## Executable acceptance order
1. **Geometry P0**: Redwood Bedroom: inspect the actual measured-pose refusion outcome from research/redwood-measured-loop-pose-refusion-20261030. If ICP rejects all corrections, preserve the negative result and switch to a distinct mature registration/fusion approach. Compare 3 registered RGB camera views and 2 free views, furniture outlines, relative positions, walls, holes, and fragmentation. Do not promote a visually worse mesh.
2. **Cross-scene P0**: Re-run the winning measured RGB-D route on Stanford Lounge and Redwood Bedroom. No hard-coded scene geometry or hidden prior. Publish inspectable colored GLB and five-view A/B for each.
3. **Product P1**: One local import/reconstruct/view/save/reopen workflow with honest progress, failure recovery and no dependence on synthetic demos. Viewer supports free orbit, first-person recorded camera, and timestamp navigation on the same persisted scene.
4. **Temporal P1**: Maintain synchronized camera, geometry provenance, timestamps and visibility of unobserved regions. Do not hallucinate dynamic subjects.
5. **Quality P2**: Improve texture/color fidelity, geometry continuity, scalable fusion and scene navigation using mature libraries, with independent visual acceptance. Distinguish measured/estimated/generated/prior.
6. **Long-term P3**: Authorized wearable capture, power-aware sensor interfaces, continuous timeline, historical scene retrieval and on-device display. These remain unimplemented until real hardware and data support them.

## Work boundaries
- Geometry branch changes real fusion and mesh; information-reduction branch owns input dependency removal/adapters; shared UI has one canonical implementation.
- Preserve #74 FAIL and desk 16.7779cm > 15cm FAIL until independent evidence warrants new results. Do not silently change thresholds.
- No merges with unresolved conflicts, no force pushes. Each promoted feature must work from the actual user workflow, not merely pass unit tests.
- When source coverage is insufficient, render unknown rather than fabricate certainty.
- Every real result links to input provenance, original images, colored mesh, reproducible run and fixed-camera visual comparison. No result means report **no result**.

## Completion criteria
This document is a delivery contract, not proof of completed features. Completion requires a real end-to-end session with a recognizable room, stable third-person navigation, correct time selection, save/reopen, and repeatability on a second independently captured room. Wearable historical capture is a later hardware-dependent milestone.
