#!/usr/bin/env python3
"""ECHO scene pipeline: video in, scene out.

Stages (each writes into the job dir and updates job.json):
  1. extract  - pull sharp, non-duplicate frames from the video (ffmpeg)
  2. poses    - COLMAP SfM camera trajectory            (external: colmap)
  3. markers  - printed AprilTag sheet -> metric scale + floor level
                (optional; writes markers.json, no-op without markers)
  4. depth    - per-frame metric-aligned depth           (later milestone)
  5. fuse     - TSDF fusion via the echo4d kernel        (later milestone)
  6. present  - scene.ply / scene.glb + viewer page
  mesh        - optional surface mesh via echo_scene.mesh, after fuse;
                best-effort, skipped quietly when the module is absent

Usage:
  python3 -m echo_scene.pipeline extract <video> <job_dir>
  python3 -m echo_scene.pipeline poses <job_dir>
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def _run(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def _tool_path(name: str) -> str:
    """Resolve an external media tool (ffmpeg / ffprobe) to a runnable path.

    Frozen builds ship static Windows binaries under assets/ffmpeg/
    (see the datas in echo-scene.spec), so when running frozen look in
    the PyInstaller bundle dir (sys._MEIPASS) and beside the executable
    first. The source-tree assets/ffmpeg/ copy is checked too, under
    this platform's own file name: a Windows source run finds the .exe,
    while a Linux dev box (where only the .exe exists) skips it. Final
    fallback is PATH via shutil.which, which keeps dev behaviour
    unchanged; if the tool is nowhere, the bare name is returned so the
    subprocess error stays the familiar FileNotFoundError.
    """
    if getattr(sys, "frozen", False) or sys.platform.startswith("win"):
        names = [f"{name}.exe", name]
    else:
        names = [name]
    roots = []
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            roots.append(Path(meipass))
        roots.append(Path(sys.executable).resolve().parent)
    roots.append(Path(__file__).resolve().parent.parent)
    for root in roots:
        for exe_name in names:
            cand = root / "assets" / "ffmpeg" / exe_name
            if cand.is_file():
                return str(cand)
    return shutil.which(name) or name


def _ffprobe_duration(video: Path) -> float:
    out = subprocess.run(
        [_tool_path("ffprobe"), "-v", "error", "-show_entries",
         "format=duration", "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def stage_extract(video: Path, job_dir: Path) -> dict:
    frames_dir = job_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    duration = _ffprobe_duration(video)
    # Target roughly 160-320 frames: one frame every 0.25s by default,
    # thinned for long clips. The budget is ~2 fps (320 frames over a
    # ~160 s walkthrough); the denser sampling is what lets the custom
    # pairing in stage_poses register most frames instead of a third of
    # them. Blur/duplicate rejection comes with the quality pass; v0
    # keeps every sampled frame.
    interval = max(0.25, duration / 320.0)
    _run([
        _tool_path("ffmpeg"), "-y", "-v", "error", "-i", video,
        "-vf", f"fps=1/{interval:.3f}",
        "-q:v", "2", frames_dir / "frame_%05d.jpg",
    ])
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    info = {
        "stage": "extract",
        "video": str(video),
        "duration_seconds": round(duration, 2),
        "frame_interval_seconds": round(interval, 3),
        "frame_count": len(frames),
        "frames_dir": str(frames_dir),
    }
    (job_dir / "job.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return info


def _write_ply(points, path: Path):
    """points: list of (x, y, z, r, g, b). Plain ASCII PLY, viewable anywhere."""
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        for prop in ("x", "y", "z"):
            f.write(f"property float {prop}\n")
        for prop in ("red", "green", "blue"):
            f.write(f"property uchar {prop}\n")
        f.write("end_header\n")
        for x, y, z, r, g, b in points:
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")


def _build_pair_list(names, pairs_path: Path) -> int:
    """Write a COLMAP match list that keeps the reconstruction connected.

    Stock sequential matching pairs only near neighbours in a narrow
    window, and on this footage most frames never registered (30 of the
    120 selected). The custom list combines three deterministic sets:
      - a +-30-frame temporal band over all frames (wide overlap, so the
        chain cannot drift apart),
      - first-40 x last-40 loop-closure pairs (a walkthrough usually
        ends near where it started),
      - sparse global "glue" pairs between every 12th frame, so distant
        stretches of the trajectory share verified matches.
    Verified on the room job: 182 of 240 frames registered in a single
    connected model. Returns the number of pairs written.
    """
    n = len(names)
    pairs = set()
    for i in range(n):
        for j in range(i + 1, min(n, i + 31)):
            pairs.add((i, j))
    for i in range(min(40, n)):
        for j in range(max(0, n - 40), n):
            if i < j:
                pairs.add((i, j))
    glue = list(range(0, n, 12))
    for a in range(len(glue)):
        for b in range(a + 1, len(glue)):
            pairs.add((glue[a], glue[b]))
    with open(pairs_path, "w") as f:
        for i, j in sorted(pairs):
            f.write(f"{names[i]} {names[j]}\n")
    return len(pairs)


def stage_poses(job_dir: Path) -> dict:
    import pycolmap

    # Trajectory always uses the FULL extracted frame set: camera tracking
    # lives on maximum overlap, and the custom pairing needs the dense
    # sequence. (An earlier version preferred the sharpness-selected subset
    # here; that starved the matcher -- 70/320 registered vs 180/240 on the
    # full set. Selection only economizes the depth stage, which reads
    # poses.json and therefore only ever sees registered frames anyway.)
    frames_dir = job_dir / "frames"
    sfm_dir = job_dir / "sfm"
    sfm_dir.mkdir(parents=True, exist_ok=True)
    db_path = sfm_dir / "database.db"
    reconstructions = {}
    match_pairs = None
    existing = [
        d for d in sfm_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and (d / "cameras.bin").exists()
    ] if sfm_dir.exists() else []
    if existing:
        for d in existing:
            reconstructions[int(d.name)] = pycolmap.Reconstruction(str(d))
    else:
        if db_path.exists():
            db_path.unlink()
        # One physical camera shot the whole video, so all frames share
        # ONE camera model. The default per-image cameras let each frame
        # invent its own lens (focal wandered 1590-2270 px on the room
        # job), which hides shape error in the intrinsics. Verified
        # 2026-10-31 (experiments/sc_room2_report.md): same 230/320
        # registrations, single converged camera, small real gain in
        # cross-frame consistency.
        pycolmap.extract_features(db_path, frames_dir,
                                  camera_mode=pycolmap.CameraMode.SINGLE)
        names = sorted(
            p.name for p in frames_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        pairs_path = sfm_dir / "pairs.txt"
        match_pairs = _build_pair_list(names, pairs_path)
        print(f"custom pairing: {match_pairs} pairs over {len(names)} frames", flush=True)
        pycolmap.match_image_pairs(
            db_path,
            pairing_options=pycolmap.ImportedPairingOptions(match_list_path=pairs_path),
        )
        reconstructions = pycolmap.incremental_mapping(db_path, frames_dir, sfm_dir)
    if not reconstructions:
        info = {"stage": "poses", "ok": False, "reason": "no reconstruction"}
        (job_dir / "job.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
        return info
    rec = max(reconstructions.values(), key=lambda r: r.num_reg_images())

    points = [
        (p.xyz[0], p.xyz[1], p.xyz[2], p.color[0], p.color[1], p.color[2])
        for p in rec.points3D.values()
    ]
    _write_ply(points, job_dir / "sparse_points.ply")

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
    info = {
        "stage": "poses",
        "ok": True,
        "registered_images": rec.num_reg_images(),
        "match_pairs": match_pairs,
        "sparse_points": len(points),
        "sparse_points_ply": str(job_dir / "sparse_points.ply"),
        "poses_json": str(job_dir / "poses.json"),
    }
    (job_dir / "job.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return info


def stage_select(job_dir: Path, max_frames: int = 120, min_sharpness: float = 20.0) -> dict:
    from .quality import select_frames

    chosen = select_frames(job_dir / "frames", max_frames, min_sharpness)
    payload = [str(p.resolve()) for p in chosen]
    (job_dir / "selected_frames.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2)
    )
    info = {"stage": "select", "ok": True, "selected": len(payload)}
    (job_dir / "job.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return info


def stage_depth(job_dir: Path) -> dict:
    from .depth import estimate_depth, save_depth_preview

    models_dir = Path(__file__).resolve().parent.parent / "models"
    base_model = models_dir / "depth_anything_v2_base.onnx"
    small_model = models_dir / "depth_anything_v2_small.onnx"
    # The Base model gives visibly smoother, more continuous surfaces
    # than Small (verified on the room job); fall back to Small when the
    # Base weights have not been downloaded.
    model = base_model if base_model.exists() else small_model
    frames_dir = job_dir / "frames"
    poses_file = job_dir / "poses.json"
    if poses_file.exists():
        names = sorted(json.loads(poses_file.read_text()).keys())
    else:
        names = sorted(p.name for p in frames_dir.glob("frame_*.jpg"))
    depth_dir = job_dir / "depth"
    preview_dir = job_dir / "depth_preview"
    depth_dir.mkdir(exist_ok=True)
    preview_dir.mkdir(exist_ok=True)
    done = []
    for name in names:
        src = frames_dir / name
        if not src.exists():
            continue
        depth = estimate_depth(src, model)
        stem = Path(name).stem
        np.save(depth_dir / f"{stem}.npy", depth)
        save_depth_preview(depth, preview_dir / f"{stem}.png")
        done.append(name)
    info = {"stage": "depth", "ok": True, "depth_frames": len(done), "model": model.name}
    (job_dir / "job.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return info


def stage_mesh(job_dir: Path) -> dict:
    """Optional mesh step after fuse (surface over the fused cloud).

    echo_scene.mesh is delivered separately and may not exist yet; any
    failure here is only logged so the pipeline still finishes with the
    point-cloud scene.
    """
    try:
        from echo_scene.mesh import stage_mesh as _stage_mesh
    except ImportError as e:
        info = {"stage": "mesh", "ok": False, "skipped": True,
                "reason": f"mesh module unavailable: {e}"}
        print(json.dumps(info, ensure_ascii=False), flush=True)
        return info
    try:
        return _stage_mesh(job_dir)
    except Exception as e:  # the mesh is a garnish, never fatal
        info = {"stage": "mesh", "ok": False, "skipped": True,
                "reason": f"mesh stage failed: {e}"}
        print(json.dumps(info, ensure_ascii=False), flush=True)
        return info


def stage_voxel(job_dir: Path) -> dict:
    """Optional voxel-model step after mesh (solid pixel-style model).

    Same non-fatal contract as the mesh step: any failure is only logged
    so the pipeline still finishes with the point-cloud scene.
    """
    try:
        from echo_scene.voxels import stage_voxel_model as _stage_voxel
    except ImportError as e:
        info = {"stage": "voxel", "ok": False, "skipped": True,
                "reason": f"voxel module unavailable: {e}"}
        print(json.dumps(info, ensure_ascii=False), flush=True)
        return info
    try:
        return _stage_voxel(job_dir)
    except Exception as e:  # a garnish, never fatal
        info = {"stage": "voxel", "ok": False, "skipped": True,
                "reason": f"voxel stage failed: {e}"}
        print(json.dumps(info, ensure_ascii=False), flush=True)
        return info


def stage_present(job_dir: Path) -> dict:
    import shutil

    template = Path(__file__).resolve().parent.parent / "viewer.html"
    shutil.copyfile(template, job_dir / "viewer.html")
    info = {
        "stage": "present",
        "ok": (job_dir / "points.bin").exists(),
        "viewer": str(job_dir / "viewer.html"),
    }
    (job_dir / "job.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return info


def main(argv):
    if len(argv) >= 2 and argv[1] == "extract" and len(argv) == 4:
        info = stage_extract(Path(argv[2]).resolve(), Path(argv[3]).resolve())
    elif len(argv) >= 2 and argv[1] == "poses" and len(argv) == 3:
        info = stage_poses(Path(argv[2]).resolve())
    elif len(argv) >= 2 and argv[1] == "select" and len(argv) == 3:
        info = stage_select(Path(argv[2]).resolve())
    elif len(argv) >= 2 and argv[1] == "markers" and len(argv) == 3:
        from echo_scene.markers import stage_markers

        info = stage_markers(Path(argv[2]).resolve())
    elif len(argv) >= 2 and argv[1] == "depth" and len(argv) == 3:
        info = stage_depth(Path(argv[2]).resolve())
    elif len(argv) >= 2 and argv[1] == "fuse" and len(argv) == 3:
        from echo_scene.fuse import stage_fuse

        info = stage_fuse(Path(argv[2]).resolve())
    elif len(argv) >= 2 and argv[1] == "mesh" and len(argv) == 3:
        info = stage_mesh(Path(argv[2]).resolve())
    elif len(argv) >= 2 and argv[1] == "voxel" and len(argv) == 3:
        info = stage_voxel(Path(argv[2]).resolve())
    elif len(argv) >= 2 and argv[1] == "present" and len(argv) == 3:
        info = stage_present(Path(argv[2]).resolve())
    else:
        print(__doc__)
        return 2
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
