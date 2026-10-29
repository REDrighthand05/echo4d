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
