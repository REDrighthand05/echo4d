"""Shared-single-camera rebuild of the room2 job (pose-side test #2).

Identical to production stage_poses (same frames, same imported pair
list -- jobs/room2/sfm/pairs.txt reused verbatim, same incremental
mapping, same poses.json writer) EXCEPT feature extraction runs with
pycolmap.CameraMode.SINGLE, so all 320 frames share ONE camera instead
of 230+ independent per-image cameras. Tests whether the surplus
per-frame intrinsics were absorbing shape error (see
experiments/ba_room2_report.md).

Usage: python experiments/sc_room2.py [job_dir]   (default jobs/room2_sc)
Writes poses.json + sparse_points.ply into the job, prints a JSON
summary at the end. Long-running: extraction + matching + mapping.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pycolmap  # noqa: E402

from echo_scene.pipeline import _write_ply  # noqa: E402


def main() -> int:
    job = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("jobs/room2_sc")
    frames_dir = job / "frames"
    sfm_dir = job / "sfm"
    sfm_dir.mkdir(parents=True, exist_ok=True)
    db_path = sfm_dir / "database.db"

    # Resume support: a previous (crashed) run may already have extracted.
    n_existing = 0
    if db_path.exists():
        try:
            _db = pycolmap.Database.open(str(db_path))
            n_existing = _db.num_images()
            _db.close()
        except Exception:
            n_existing = 0
    if n_existing >= 320:
        print(f"[sc] database already holds {n_existing} extracted images; skipping extraction", flush=True)
    else:
        if db_path.exists():
            db_path.unlink()
        print("[sc] extract_features with CameraMode.SINGLE ...", flush=True)
        pycolmap.extract_features(
            db_path, frames_dir, camera_mode=pycolmap.CameraMode.SINGLE
        )
    db = pycolmap.Database.open(str(db_path))
    cams = {}
    for cam in db.read_all_cameras():
        cams[int(cam.camera_id)] = {
            "model": str(cam.model),
            "width": cam.width,
            "height": cam.height,
            "params": [float(v) for v in cam.params],
        }
    print(f"[sc] cameras in fresh db: {json.dumps(cams)}", flush=True)
    db.close()

    pairs_path = sfm_dir / "pairs.txt"
    n_pairs = sum(1 for _ in open(pairs_path))
    print(f"[sc] match_image_pairs over {n_pairs} imported pairs ...", flush=True)
    pycolmap.match_image_pairs(
        db_path,
        pairing_options=pycolmap.ImportedPairingOptions(match_list_path=pairs_path),
    )

    print("[sc] incremental_mapping ...", flush=True)
    reconstructions = pycolmap.incremental_mapping(db_path, frames_dir, sfm_dir)
    if not reconstructions:
        print(json.dumps({"ok": False, "reason": "no reconstruction"}))
        return 1
    for idx, r in reconstructions.items():
        print(f"[sc] model {idx}: {r.num_reg_images()} registered images", flush=True)
    rec = max(reconstructions.values(), key=lambda r: r.num_reg_images())

    # --- poses.json + sparse_points.ply: byte-for-byte the stage_poses writer ---
    points = [
        (p.xyz[0], p.xyz[1], p.xyz[2], p.color[0], p.color[1], p.color[2])
        for p in rec.points3D.values()
    ]
    _write_ply(points, job / "sparse_points.ply")

    poses = {}
    cam_ids = set()
    for image in rec.images.values():
        has_pose = image.has_pose() if callable(image.has_pose) else image.has_pose
        if not has_pose:
            continue
        cam_ids.add(image.camera_id)
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
    (job / "poses.json").write_text(json.dumps(poses, ensure_ascii=False, indent=2))

    cam_summary = {}
    for cid in sorted(cam_ids):
        cam = rec.cameras[cid]
        cam_summary[int(cid)] = {
            "model": str(cam.model),
            "width": cam.width,
            "height": cam.height,
            "params": [float(v) for v in cam.params],
        }
    print(json.dumps({
        "ok": True,
        "job": str(job),
        "models": {int(k): v.num_reg_images() for k, v in reconstructions.items()},
        "registered_images": rec.num_reg_images(),
        "sparse_points": len(points),
        "distinct_cameras_in_best_model": len(cam_ids),
        "cameras": cam_summary,
        "mean_reprojection_error": float(rec.compute_mean_reprojection_error()),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
