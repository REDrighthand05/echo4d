# Stanford Lounge hybrid A/B: replacement is a real negative result

Run: https://github.com/REDrighthand05/echo4d/actions/runs/40827393230
Actual 3+2 images and both colored GLBs:
https://github.com/REDrighthand05/echo4d/actions/runs/40827393230/artifacts/12877640450

The 2026-10-29 replacement experiment used real Stanford Lounge RGB-D
and an independently PRE-EXISTING, already registered scan. It removed
74,641 of 426,789 measured faces as supposed duplicates, and inserted
51,441 prior planar faces (total 403,589). Inspection of all five actual
views found **new dark tears/speckles in the wall and furniture** and
almost unchanged or slightly worse room-scale free views. This is not
a successful room shell completion. It must not be represented as such.

Follow-up observed-first version leaves ALL measured faces intact and only
adds prior triangles unsupported by measured vertices. All added geometry
remains explicitly prior, not independent reconstruction. This is a
conservative geometry change, not a promise of visual improvement.
#74 and Desk V2 remain FAIL.
