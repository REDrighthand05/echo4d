"""Pose-side global bundle adjustment on jobs/room2_ba (scratch copy of room2).

Hypothesis under test: the residual wall warp in the fused room scene is
pose/trajectory error, not depth error (depth-side corrections were
exhausted on room1_big: spread stuck at 0.0834). The principled tool is
global bundle adjustment over the existing COLMAP reconstruction with
focal length + distortion refinement.

Steps: load sfm/0 -> reprojection error -> pycolmap.bundle_adjustment
(in place) -> error again -> write model back -> regenerate poses.json
in stage_poses' exact format. jobs/room2 is never touched.
"""
import json
import sys
from pathlib import Path

import pycolmap

JOB = Path("jobs/room2_ba")
MODEL = JOB / "sfm" / "0"


def write_poses_json(rec: "pycolmap.Reconstruction", job_dir: Path) -> int:
    """Mirror of echo_scene.pipeline.stage_poses' poses.json writer."""
    poses = {}
    for image in rec.images.values():
        has_pose = image.has_pose() if callable(image.has_pose) else image.has_pose
        if not has_pose:
            continue
        cam = rec.cameras[image.camera_id]
        cfw = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
        q = cfw.rotation.quat  # x, y, z, w in pycolmap
        t = cfw.translation
        poses[image.name] = {
            "cam_from_world_quat_xyzw": [float(v) for v in (q[0], q[1], q[2], q[3])],
            "cam_from_world_t": [float(v) for v in t],
            "camera": {
                "model": str(cam.model),
                "width": cam.width,
                "height": cam.height,
                "params": [float(v) for v in cam.params],
            },
        }
    (job_dir / "poses.json").write_text(json.dumps(poses, ensure_ascii=False, indent=2))
    return len(poses)


def main() -> None:
    rec = pycolmap.Reconstruction(str(MODEL))
    print(f"loaded: {rec.num_reg_images()} reg images, {rec.num_points3D()} points3D")
    for cam in rec.cameras.values():
        print(f"camera {cam.camera_id}: model={cam.model} {cam.width}x{cam.height} "
              f"params={[round(float(v), 4) for v in cam.params]}")
    err_before = rec.compute_mean_reprojection_error()
    print(f"mean reprojection error BEFORE BA: {err_before:.6f} px", flush=True)

    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = True
    options.refine_principal_point = False
    options.refine_extra_params = True
    options.print_summary = True
    pycolmap.bundle_adjustment(rec, options)

    err_after = rec.compute_mean_reprojection_error()
    print(f"mean reprojection error AFTER BA:  {err_after:.6f} px", flush=True)
    for cam in rec.cameras.values():
        print(f"camera {cam.camera_id} after: params={[round(float(v), 4) for v in cam.params]}")

    rec.write(str(MODEL))
    n = write_poses_json(rec, JOB)
    print(f"wrote refined model to {MODEL} and poses.json with {n} poses")
    print(json.dumps({"err_before": err_before, "err_after": err_after}))


if __name__ == "__main__":
    sys.exit(main())
