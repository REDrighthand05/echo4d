# Mature room reconstruction restart

The user requested preserving the current result, then finding mature GitHub implementations afresh. The previous experiments are frozen at [archive/room-results-20261029](https://github.com/REDrighthand05/echo4d/tree/archive/room-results-20261029) and [archive/tum-pose-results-20261029](https://github.com/REDrighthand05/echo4d/tree/archive/tum-pose-results-20261029). This branch starts independently from main d0295f8d855cbeef9f4f4c74705b3d9e4edffebd.

## What was found in the actual upstream sources

| Project | Suitable role | Decision |
|---|---|---|
| [RTAB-Map](https://github.com/introlab/rtabmap) | Complete RGB-D SLAM, loop closures, map export, standalone application | Strong complete-system candidate. Current maintained code and BSD licensing described in official README. Keep as next independent provider rather than reimplementing its graph. |
| [Open3D Reconstruction System](https://github.com/isl-org/Open3D/tree/v0.19.0/examples/python/reconstruction_system) | Complete official RGB-D pipeline: fragment odometry → fragment registration/loop closures → refined global poses → whole-scene integration | First actual execution in this branch, using the upstream system unmodified at 1e7b17438687a0b0c1e5a7187321ac7044afe275 (v0.19.0). This replaces our hand-built 96-camera graph, not another threshold variant. |
| [MVS-Texturing](https://github.com/nmoehrle/mvs-texturing) | Sharp view selection and seam-adjusted texture atlas on a registered mesh + images | Suitable subsequent texture provider; official README says BSD 3-Clause. Does not reconstruct absent walls. |
| [OpenMVS](https://github.com/cdcseacave/openMVS) | Dense cloud, mesh, refinement and texture from calibrated images/poses and sparse point cloud | Complete photogrammetry alternative, not first choice when real RGB-D is already available. Official repository reports AGPL-3.0; integration scope must respect that. |
| [BundleFusion](https://github.com/niessner/BundleFusion) | RGB-D global consistency and surface re-integration | Proven system, but original README requires VS2013, DirectX June2010, CUDA7 and non-commercial licensing; not a quick drop-in CPU CI provider. |

Sources were read from the above GitHub repositories on 2026-10-29. A mature library does not make an arbitrary partial integration a mature reconstruction system.

## Concrete execution

`tools/run_official_room.py` is an adapter, not another geometry implementation. It calls the pinned upstream `run_system.py --make --register --refine --integrate` on the complete native Stanford Lounge sequence, using the official Lounge defaults. It does not load the provided GT trajectory or previous mesh into candidate reconstruction. It retains input image/depth hashes, official estimated graphs/trajectory, native PLY, full GLB, a separately simplified preview and 3 photo/camera renders plus 2 genuinely free cameras.

Run:
```sh
python tools/run_official_room.py --upstream /path/to/Open3D-v0.19.0 --output /fresh/output
```

The new experiment uses all frames rather than the old selected480 and estimates its own poses. It is an independent complete-system baseline, not a single-variable A/B against #112. Mesh colors initially remain upstream vertex colors; texture atlasing is a separate subsequent stage. It must not claim 100% completeness, independent surface accuracy or GPU/Hi3D results.

The workflow has a180-minute execution budget and preserves failures. A submitted/running job is not a successful room. The next outcome is an actual full model and five images, or an exact recorded execution failure. Main and the archive snapshots are not modified by this experiment.
