# Stanford Lounge — original 3,150,436-face reference scan (negative geometry result)

Real source: 640×480 Stanford Lounge RGB-D frames, recorded camera poses, native 426,789-face observed TSDF, and the separate **PRE-EXISTING** Open3D LoungeRGBDImages reference scan. The reference is **not** independently reconstructed from the RGB-D video.

Real run: https://github.com/REDrighthand05/echo4d/actions/runs/40833196509

Artifact with full colored 3,150,436-face original PRIOR GLB, five actual A/B images, and 426,789-face observed + additive shell GLB:
https://github.com/REDrighthand05/echo4d/actions/runs/40833196509/artifacts/12879293706

**Observed output:** Original 3,150,436-face reference scan, color from measured RGB-D on 93.15% of its vertices; unknown-color vertices remain grey. Compared with the previous 169,999-face reference preview, the full scan had 1,146,516 broad outer-plane candidate triangles, but 1,146,155 were already spatially supported/overlapping observed geometry. **Only 361 original scan triangles remained eligible to add**. The observed 426,789 faces were preserved in the additive hybrid.

**Human review of 3 recorded and 2 room-scale free views:** The sofa chairs, round table and plants remain identifiable in both observed and existing-scan models. Original full scan is not a room-shell fix: in the two free views, existing scan has substantially more black unobserved area than the measured mesh. The first recorded view also has more black at the top. No visually meaningful improvement from 361 extra faces. Stop the original-prior shell addition line; do not tune it indefinitely.

Real free-view covered pixels (measured vs full PRIOR): 45,991 vs 29,823 and 27,136 vs 13,298. These are *rendered hit counts*, not reconstruction accuracy. All five views and GLBs are available in the artifact above.

Next geometric action: return to **measured** Stanford RGB-D and improve side/back surface coverage by selecting actual 640×480 depth frames from the remaining recorded trajectory for the undercovered free-view regions; no existing scan geometry in that reconstruction.

No change to main, #74 FAIL or old Desk V2 16.7779 cm > 15 cm FAIL.
