# Stanford Lounge: measured objects + PRE-EXISTING room shell

The independently observed full 426,789-face TSDF is retained except for
triangles demonstrably coincident with the prior room shell. The shell is
selected from the already registered 169,999-face preview of the existing
Stanford Lounge scan using large planar support; it is **not** reconstructed
from RGB-D. Original inferred planar tiles are excluded. GLBs distinguish
cyan measured geometry and orange existing prior geometry.

Run on the pinned real published Stanford files:

    PYTHONPATH=tools python tools/stanford_observed_prior_shell.py \
      --observed docs/real-room-review/stanford-lounge-native-planar/OBSERVED-plus-INFERRED.glb \
      --prior docs/real-room-review/stanford-lounge-second-room/PREEXISTING-PRIOR-RGB.glb \
      --observed-report docs/real-room-review/stanford-lounge-native-planar/result.json \
      --prior-report docs/real-room-review/stanford-lounge-second-room/result.json \
      --output /tmp/stanford-hybrid

The dedicated workflow also downloads original Lounge 640x480 RGB-D through
Open3D, renders three recorded photo/observed/hybrid comparisons and two
room-scale free views, and publishes the resulting GLBs and actual images
on this research branch. No claim of visual improvement before inspection.
#74 and the old Desk FAIL are unchanged. No RGB-only inference.
