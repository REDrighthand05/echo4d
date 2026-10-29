# Real Redwood Bedroom: supported Poisson surface hypothesis
Source: 720 frames of original 640x480 RGB-D, measured depth and
recorded camera poses. Mature Open3D Poisson on the observed Mesh.
The 561,887 measured TSDF faces are preserved; new faces are
INFERRED, not measured. Orange in provenance GLB means inferred.
Colored GLB uses nearest observed vertex colors for inferred
patches (not actual measured color there). No existing laser
reference Mesh was used. Three recorded and two free-view images
compare observed TSDF with Poisson candidate. Visual recognition
and true surface accuracy have NOT automatically passed.
#74 and Desk V2 16.7779cm > 15cm remain FAIL.
Source experiment: https://github.com/REDrighthand05/echo4d/actions/runs/40798646737


## Actual 3+2 human visual review, 2026-10-29 — NEGATIVE

Inspected all five original PNG panels in the real Actions artifact. Recorded
view 00000: the photographed wooden writing desk is identifiable, but the
reconstructed desk loses crisp shelves and the chair/foreground breaks into
smudged, disconnected fragments. View 10995: the actual dark bedside cabinet
and curtain are not reliably preserved as separate crisp objects; the
reconstruction merges broad soft-looking surfaces. View 21929: bed, pillows,
and lamp are recognizable, but bed edges and lamp details remain coarse.
Both free cameras show the same obstructed or missing regions as the TSDF
baseline; free-view-2 has a large black unknown area. The orange inferred
patch is tiny and does not meaningfully restore missing objects.

Measured 720-frame TSDF: 561,887 faces. Mature Poisson produced 435,915
candidate faces but the supported gap-only rule added only 1,051 inferred
faces (+0.187%). Same fixed 320x240 cameras: no hit-pixel gain at any of
three recorded views or free-view-1; free-view-2 gains 10 pixels.
**No meaningful visual geometry improvement; NOT accepted for free roaming.**
The measured Mesh and the prior-free baseline remain available unchanged.
No "100%" score or independent geometry accuracy is claimed.

Next ONE geometry action: test the supplied *existing* LiDAR/reference room
surface as a clearly labeled spatial prior and project recorded RGB-D
color onto it, rather than continuing Poisson/TSDF parameter tweaks. Keep
a separate observed-only Mesh and visibly label all prior-supplied faces.
This is explicitly a high-information reconstruction path, NOT recovery
from RGB video alone. It should be evaluated at the same three recorded
and two free cameras. #74 and the old Desk V2 FAIL stay unchanged.

Real experiment and raw artifact:
https://github.com/REDrighthand05/echo4d/actions/runs/40804196191
