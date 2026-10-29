# Stanford Lounge native planar patch: visual decision (2026-10-29)

Real source: 640x480 Lounge RGB-D, 360 frames, logged camera poses.
No pre-existing scanned reference geometry in this experiment.

Full native observed TSDF: 426,789 triangles. Bounded planar inference:
2,654 triangles, 6.5023 m² of patch surface, explicitly orange in the
source-mask GLB. In contrast, the *decimated 85k preview* appeared to need
4,020 inferred triangles; that was not evidence of additional real missing
room geometry. Original observed faces remain unchanged.

Human inspection of 5 **room-scale virtual** VTK viewpoints: the furniture
and layout remain recognizable, but measured vs patched images are nearly
indistinguishable; walls and the scene boundary are still broken. The
official 3 recorded-camera + 2 earlier 35cm-offset "free" camera images
likewise show essentially no practical improvement. Those near-camera
closeups were **not** a proper room-scale free-roam test; the separate
room-scale raycast review workflow addresses this limitation.

Decision: do not keep tuning morphological radius to chase triangle counts.
This approach did NOT deliver a freely navigable room. #74 stays FAIL and
Desk V2 16.7779 cm > 15 cm stays FAIL.

Next direct geometry action: create a provenance-separated hybrid, preserving
measured RGB-D furniture surfaces and using the already registered
PREEXISTING scanned reference only for missing room-shell geometry. The
reference must never be presented as independently recovered from video.
