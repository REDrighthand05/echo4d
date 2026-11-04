#!/usr/bin/env python3
"""ECHO scene pipeline: video in, scene out.

Stages (each writes into the job dir and updates job.json):
  1. extract  - pull sharp, non-duplicate frames from the video (ffmpeg)
  2. poses    - COLMAP SfM camera trajectory            (external: colmap)
  3. markers  - printed AprilTag sheet -> metric scale + floor level
                (optional; writes markers.json, no-op without markers)
  4. depth    - per-frame metric-aligned depth           (later milestone)
  5. fuse     - per-frame depth fusion + corroboration trim
  6. clean    - continuity cleanup: keep the connected main body
  6b. scale   - true-scale anchoring: find a standard-size reference
                object (catalogue in echo_scene/scale.py) and rescale
                the scene to metres; unknown scale never blocks a job
  7. present  - standalone scene.html (data embedded, opens as a file)
  mesh        - optional surface mesh via echo_scene.mesh, after fuse;
                best-effort, skipped quietly when the module is absent
  voxel       - frozen voxel solidification, after mesh; best-effort
  preview     - process-preview page (preview.html) in the job folder
  replay      - route-replay page (replay.html): a stand-in figure
                glides the recovered camera route over the scene

Everything a job produces is a plain local file in the job folder --
no server, no localhost page (v0.2 outputs-as-files rule).

Usage:
  python3 -m echo_scene.pipeline extract <video> <job_dir>
  python3 -m echo_scene.pipeline poses <job_dir>
"""
import base64
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


_HWACCEL_CACHE: list | None = None


def _hwaccel_args() -> list:
    """Best verified hardware-decode args for frame extraction (DS
    WO-D kernel; A/B-verified on the user's RTX 2060: median pixel
    delta 1/255 vs software decode). Probed once per process; empty
    when no accelerator verifies, and callers fall back to software
    decoding if a hardware run fails."""
    global _HWACCEL_CACHE
    if _HWACCEL_CACHE is None:
        try:
            from .hwaccel import extract_args, probe_hwaccel

            found = probe_hwaccel(_tool_path("ffmpeg"))
            _HWACCEL_CACHE = extract_args(found[0] if found else None)
        except Exception:
            _HWACCEL_CACHE = []
    return _HWACCEL_CACHE


_MATCH_W, _MATCH_H = 96, 54
_MATCH_MIN_CORR = 0.9


def _gray_thumbs_from_video(video: Path) -> np.ndarray | None:
    """Every container frame of the video as a 96x54 gray thumbnail
    (one ffmpeg decode, raw bytes). None when the decode fails."""
    try:
        out = subprocess.run(
            [_tool_path("ffmpeg"), "-v", "error", "-i", str(video),
             "-vf", f"scale={_MATCH_W}:{_MATCH_H},format=gray",
             "-f", "rawvideo", "-"],
            capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, OSError):
        return None
    n = len(out) // (_MATCH_W * _MATCH_H)
    if n == 0:
        return None
    return np.frombuffer(out[: n * _MATCH_W * _MATCH_H],
                         dtype=np.uint8).reshape(
                             n, _MATCH_H, _MATCH_W).astype(np.float64)


def _gray_thumb_of_image(path: Path) -> np.ndarray | None:
    try:
        out = subprocess.run(
            [_tool_path("ffmpeg"), "-v", "error", "-i", str(path),
             "-vf", f"scale={_MATCH_W}:{_MATCH_H},format=gray",
             "-f", "rawvideo", "-"],
            capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, OSError):
        return None
    if len(out) < _MATCH_W * _MATCH_H:
        return None
    return np.frombuffer(out[: _MATCH_W * _MATCH_H],
                         dtype=np.uint8).reshape(
                             _MATCH_H, _MATCH_W).astype(np.float64)


def _write_frame_capture_times(video: Path, frames: list,
                               job_dir: Path) -> int:
    """Pair each extracted frame with its capture-clock time and
    write frame_capture_times.json; returns the number paired.

    Only for jobs whose import landed a sensor record (see
    sensor_record.import_capture_record). The extracted frames are
    renumbered pulls from the video -- frame_00004.jpg is the fourth
    EXTRACTION, not the capture's fourth frame -- so the pairing is
    recovered by visual correlation against the video itself (the
    orientation audit's method: on the 2026-11-03 capture all 35
    posed frames matched at corr 1.000). The winning container
    frame's index is the capture frame's index, whose sensor time
    the record's frames.csv carries. Matches below the correlation
    bar are left out, and if fewer than half the frames pair, no
    file is written at all: fuse then orients without IMU rather
    than pairing poses with guessed instants.
    """
    from echo_scene import sensor_record as sr

    cap = sr.read_job_capture(job_dir)
    if cap is None or len(frames) == 0:
        return 0
    vid = _gray_thumbs_from_video(video)
    if vid is None:
        return 0
    flat = vid.reshape(len(vid), -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(flat, axis=1)
    mapping = {}
    for path in frames:
        img = _gray_thumb_of_image(path)
        if img is None:
            continue
        v = img.reshape(-1).astype(np.float64)
        v = v - v.mean()
        nv = float(np.linalg.norm(v))
        if nv < 1e-9:
            continue
        corr = (flat @ v) / (norms * nv + 1e-9)
        idx = int(np.argmax(corr))
        if corr[idx] >= _MATCH_MIN_CORR and idx < len(cap.frame_t):
            mapping[path.name] = float(cap.frame_t[idx])
    if len(mapping) < max(3, len(frames) // 2):
        return 0
    (job_dir / "frame_capture_times.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2))
    return len(mapping)


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
    tail = ["-vf", f"fps=1/{interval:.3f}",
            "-q:v", "2", frames_dir / "frame_%05d.jpg"]
    accel = _hwaccel_args()
    try:
        _run([_tool_path("ffmpeg"), "-y", "-v", "error", *accel,
              "-i", video, *tail])
    except subprocess.CalledProcessError:
        if not accel:
            raise
        globals()["_HWACCEL_CACHE"] = []
        _run([_tool_path("ffmpeg"), "-y", "-v", "error",
              "-i", video, *tail])
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    paired = _write_frame_capture_times(video, frames, job_dir)
    info = {
        "stage": "extract",
        "video": str(video),
        "duration_seconds": round(duration, 2),
        "frame_interval_seconds": round(interval, 3),
        "frame_count": len(frames),
        "frames_dir": str(frames_dir),
    }
    if paired:
        info["frame_capture_times"] = paired
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

    # The Base model gives visibly smoother, more continuous surfaces
    # than Small (verified on the room job); fall back to Small when the
    # Base weights have not been downloaded. Resolution goes through
    # the shared paths module -- the installed layout keeps models at
    # the install root, not beside this file.
    from .paths import require_depth_model

    model = require_depth_model(prefer_base=True)
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
    # Batched inference (DS kernel, merge-gated in
    # tests/test_depth_batch.py); falls back to per-frame on any
    # batch-level failure, per frame on a batch miss.
    existing = [n for n in names if (frames_dir / n).exists()]
    depths = None
    try:
        from .depth import _get_session
        from .depth_batch import estimate_depth_batch

        depths = estimate_depth_batch(
            [frames_dir / n for n in existing], _get_session(model),
            batch_size=4)
        if depths is None or len(depths) != len(existing):
            depths = None
    except Exception:
        depths = None
    done = []
    for i, name in enumerate(existing):
        src = frames_dir / name
        depth = depths[i] if depths is not None else None
        if depth is None:
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


def stage_clean(job_dir: Path) -> dict:
    """Continuity cleanup over the fused cloud, between fuse and mesh.

    Non-fatal by design (same contract as mesh/voxel): the scene is
    already complete before this stage, so a cleanup failure falls back
    to the uncleaned cloud instead of failing the job; the returned
    clean_applied flag records the truth. The clean module itself
    writes job.json when it runs.
    """
    try:
        from echo_scene.clean import stage_clean as _stage_clean

        info = _stage_clean(job_dir)
    except Exception as e:
        info = {"stage": "clean", "ok": False,
                "reason": f"clean stage failed: {e}"}
        print(json.dumps(info, ensure_ascii=False), flush=True)
    info = dict(info)
    applied = bool(info.get("ok"))
    info["ok"] = True
    info["clean_applied"] = applied
    return info


def stage_scale(job_dir: Path) -> dict:
    """True-scale anchoring between clean and mesh (Pioneer A).

    Non-fatal by design (same contract as clean): a job whose scale
    stays unknown is still a complete job, so any failure here --
    including the scale module being unavailable -- only means the
    scene keeps its arbitrary units; scale.json records the outcome
    whenever the stage ran (the wrapper writes a minimal unknown
    record when the module itself could not).
    """
    try:
        from echo_scene.scale import stage_scale as _stage_scale

        info = _stage_scale(job_dir)
    except Exception as e:
        info = {"stage": "scale", "status": "unknown",
                "scale_to_meters": None, "evidence": "未知",
                "reason": f"scale stage failed: {e}"}
        print(json.dumps(info, ensure_ascii=False), flush=True)
        try:
            (job_dir / "scale.json").write_text(
                json.dumps(info, ensure_ascii=False, indent=2))
            (job_dir / "job.json").write_text(
                json.dumps(info, ensure_ascii=False, indent=2))
        except OSError:
            pass
    info = dict(info)
    info["ok"] = True
    info["scale_applied"] = bool(
        info.get("status") == "set" and info.get("applied"))
    return info


def stage_present(job_dir: Path) -> dict:
    """Write the standalone scene page into the job folder.

    scene.html is ONE self-contained file: the point cloud is embedded
    as base64 (the viewer then never fetches points.bin) and the mesh
    buffers ride as data: URLs, so the page opens by double-click with
    no server -- the v0.1 localhost viewer page is gone; the user gets
    files, not addresses. The universal model files (voxel GLB/PLY,
    mesh PLY, points PLY) already sit in the same folder as plain
    files; their paths are reported in the info dict.
    """
    points_bin = job_dir / "points.bin"
    if not points_bin.exists():
        info = {"stage": "present", "ok": False, "reason": "points.bin missing"}
        (job_dir / "job.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
        return info
    template = Path(__file__).resolve().parent.parent / "viewer.html"
    html = template.read_text(encoding="utf-8")
    points_b64 = base64.b64encode(points_bin.read_bytes()).decode()
    html = html.replace(
        "<script>\n",
        f'<script>window.POINTS_B64 = "{points_b64}";</script>\n<script>\n',
        1,
    )
    if "window.POINTS_B64" not in html:
        info = {"stage": "present", "ok": False,
                "reason": "viewer template seam not found"}
        (job_dir / "job.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
        return info
    for name in ("mesh_vertices.bin", "mesh_indices.bin"):
        buf = job_dir / name
        if buf.exists():
            uri = ("data:application/octet-stream;base64,"
                   + base64.b64encode(buf.read_bytes()).decode())
            html = html.replace(f"fetch('{name}')", f"fetch('{uri}')")
    scene_path = job_dir / "scene.html"
    scene_path.write_text(html, encoding="utf-8")
    files = {"scene_html": str(scene_path)}
    for key, name in (("voxel_glb", "voxel_model.glb"),
                      ("voxel_ply", "voxel_model.ply"),
                      ("mesh_ply", "mesh.ply"),
                      ("points_ply", "scene_points.ply")):
        candidate = job_dir / name
        if candidate.exists():
            files[key] = str(candidate)
    info = {"stage": "present", "ok": True, "scene_html": str(scene_path),
            "files": files}
    scale_path = job_dir / "scale.json"
    if scale_path.exists():
        try:
            scale_rec = json.loads(scale_path.read_text())
            info["scale"] = {
                "status": scale_rec.get("status"),
                "scale_to_meters": scale_rec.get("scale_to_meters"),
                "evidence": scale_rec.get("evidence"),
                "error_pct": scale_rec.get("error_pct"),
            }
        except (OSError, ValueError):
            pass
    (job_dir / "job.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return info


def stage_preview(job_dir: Path) -> dict:
    """Process-preview page (preview.html) as a file in the job folder.

    Garnish like the mesh: a failure here never fails a finished
    scene, so the wrapper reports ok with preview_applied=False.
    """
    try:
        from echo_scene.preview import write_preview

        result = write_preview(job_dir, job_dir / "preview.html")
        info = {"stage": "preview", "preview_applied": True, **result}
    except Exception as e:
        info = {"stage": "preview", "preview_applied": False,
                "reason": f"preview failed: {e}"}
        print(json.dumps(info, ensure_ascii=False), flush=True)
    info["ok"] = True
    (job_dir / "job.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return info


def stage_replay(job_dir: Path) -> dict:
    """Route-replay page (replay.html) as a file in the job folder.

    Garnish like the preview: a failure here never fails a finished
    scene, so the wrapper reports ok with replay_applied=False. The
    page needs only points.bin + poses.json (alignment.json when the
    fuse stage wrote one), so it can also be generated on demand for
    jobs finished by older versions.
    """
    try:
        from echo_scene.replay import write_replay

        result = write_replay(job_dir, job_dir / "replay.html")
        info = {"stage": "replay", "replay_applied": True, **result}
    except Exception as e:
        info = {"stage": "replay", "replay_applied": False,
                "reason": f"replay failed: {e}"}
        print(json.dumps(info, ensure_ascii=False), flush=True)
    info["ok"] = True
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
    elif len(argv) >= 2 and argv[1] == "clean" and len(argv) == 3:
        info = stage_clean(Path(argv[2]).resolve())
    elif len(argv) >= 2 and argv[1] == "scale" and len(argv) == 3:
        info = stage_scale(Path(argv[2]).resolve())
    elif len(argv) >= 2 and argv[1] == "mesh" and len(argv) == 3:
        info = stage_mesh(Path(argv[2]).resolve())
    elif len(argv) >= 2 and argv[1] == "voxel" and len(argv) == 3:
        info = stage_voxel(Path(argv[2]).resolve())
    elif len(argv) >= 2 and argv[1] == "present" and len(argv) == 3:
        info = stage_present(Path(argv[2]).resolve())
    elif len(argv) >= 2 and argv[1] == "preview" and len(argv) == 3:
        info = stage_preview(Path(argv[2]).resolve())
    elif len(argv) >= 2 and argv[1] == "replay" and len(argv) == 3:
        info = stage_replay(Path(argv[2]).resolve())
    else:
        print(__doc__)
        return 2
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
