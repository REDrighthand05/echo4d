# Stanford Lounge — real measured depth gap-guided refusion

Source: Open3D LoungeRGBDImages, original 640x480 real RGB-D and logged poses.
Frozen baseline: 360 evenly spaced recorded frames, 426,789 observed-only faces.
Experiment: inspect 301 unused recorded views against frozen observed Mesh;
choose 120 views with distinct measured missing surface cells; re-fuse 480
measured frames with Open3D TSDF at 0.025m voxel.
Result: 454,009 observed-only faces. No pre-existing scan, generated
surface or inferred plane was fused. The old Mesh was only used for
selecting additional frames and for before/after rendering.

Actual A/B: three real RGB / before / after views and two room-scale
free before / after views. Furniture and relative positions remain
recognizable, but in recorded views the differences are tiny. Free
view hit pixels change 23,223 -> 24,276 (+4.53%) and 12,813 ->
13,326 (+4.00%); these are *render coverage*, not geometry accuracy.
Human review: slight local fill, persistent large missing outer surfaces,
no meaningful free-walk improvement. NOT ACCEPTED as stable free orbit.
No changes to Issue #74 FAIL or old Desk 16.7779cm > 15cm FAIL.

Full native PLY and per-candidate selection stats remain in:
https://github.com/REDrighthand05/echo4d/actions/runs/40839801229/artifacts/12881634221
Source run: https://github.com/REDrighthand05/echo4d/actions/runs/40839801229
