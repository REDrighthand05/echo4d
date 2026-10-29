# Stanford Lounge native planar patch: visual decision (2026-10-29)

Real source: 640x480 Lounge RGB-D, 360 frames, logged camera poses.
No pre-existing scanned reference geometry in this experiment.

Full native observed TSDF: 426,789 triangles. Bounded planar inference:
2,654 triangles, 6.5023 m² of patch surface, explicitly orange in the
source-mask GLB. In contrast, the *decimated 85k preview* appeared to need
4,020 inferred triangles; that was not evidence of additional real missing
room geometry. Original observed faces remain unchanged.

**Actual five-view result:** 3 original RGB camera poses plus 2 true
room-scale free viewpoints were successfully rendered using the full native
Mesh (Actions #40822251618). Furniture and the lounge layout remain
recognizable. The room-scale views expose large missing outer surfaces and
broken room boundaries. Before/after are almost indistinguishable; planar
tiles do not deliver practical free roaming. The previous "free" camera
views were merely 35cm shifts near the recorded cameras, not a genuine
third-person room overview, and are superseded by the new room-scale run.

Decision: do not keep tuning morphological radius to chase triangle counts.
This approach did NOT deliver a freely navigable room. #74 stays FAIL and
Desk V2 16.7779 cm > 15 cm stays FAIL.

Next direct geometry action: create a provenance-separated hybrid, preserving
measured RGB-D furniture surfaces and using the already registered
PREEXISTING scanned reference only for missing room-shell geometry. The
reference must never be presented as independently recovered from video.
