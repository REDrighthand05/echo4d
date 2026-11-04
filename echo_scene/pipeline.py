#!/usr/bin/env python3
"""ECHO scene pipeline: video in, scene out.

Stages (each writes into the job dir and updates job.json):
  1. extract  - dense chained frame sampling for the pose pool
                (stride from the video's own correlation decay,
                dead frames dropped, legacy grid unioned in)
  2. poses    - COLMAP SfM camera trajectory (single camera) plus
                two-hop PnP relocalization of unregistered frames
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
import math
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


# --- Dense chained extraction (poses v0.2) --------------------------------
# The verified-match graph of a sparsely sampled walkthrough splits
# into components with ZERO matches across the junctions: at ~0.25 s
# spacing the viewpoint change between neighbours is simply too large
# (pose pioneer, experiments/pose_pioneer/REPORT.md: the 2026-11-03
# capture registered 35/100 selected frames that way, 51-52 once the
# sampling went dense). The pose pool is therefore built from the
# video's own decorrelation curve: the stride is the largest frame lag
# whose median thumbnail correlation still clears _POSE_CORR_FLOOR --
# neighbouring pool frames are then guaranteed strong mutual overlap,
# and the chain cannot fall apart. Near-black / textureless frames
# (matcher output is zero by construction) never enter the pool.
_POSE_CORR_FLOOR = 0.70
_POSE_MAX_LAG = 24
_MAX_POSE_FRAMES = 900
_DEAD_MEAN = 8.0
_DEAD_STD = 10.0


def _median_corr_at_lag(flat, norms, lag, step):
    idx = np.arange(0, len(flat) - lag, step)
    if len(idx) == 0:
        return 0.0
    denom = norms[idx] * norms[idx + lag]
    ok = denom > 1e-9
    if not ok.any():
        return 0.0
    corr = np.einsum("ij,ij->i", flat[idx][ok], flat[idx + lag][ok]) / denom[ok]
    return float(np.median(corr))


def _plan_pose_frames(thumbs, duration):
    """Choose container-frame indices for the pose pool from video
    thumbnails (n, H, W). Returns (indices, plan info dict).

    Pool = every `stride`-th frame (stride from the correlation decay,
    see above) UNION the legacy extraction grid (so the frames the
    depth stage was economized around are always present), minus dead
    frames. If the pool would exceed _MAX_POSE_FRAMES the stride grows
    until it fits -- connectivity is then knowingly traded for compute,
    and the plan records that it happened.
    """
    n = len(thumbs)
    mean = thumbs.mean(axis=(1, 2))
    std = thumbs.std(axis=(1, 2))
    dead = (std < _DEAD_STD) | (mean < _DEAD_MEAN)
    flat = thumbs.reshape(n, -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(flat, axis=1)
    step = max(1, n // 600)
    corr_by_lag = {}
    stride = _POSE_MAX_LAG
    for lag in range(1, _POSE_MAX_LAG + 1):
        c = _median_corr_at_lag(flat, norms, lag, step)
        corr_by_lag[lag] = round(c, 3)
        if c < _POSE_CORR_FLOOR:
            stride = max(1, lag - 1)
            break
    interval = max(0.25, duration / 320.0)
    fps = n / duration if duration > 0 else 0.0
    grid = set()
    if fps > 0:
        k = 0
        while k * interval <= duration + 1e-6:
            grid.add(min(n - 1, int(round(k * interval * fps))))
            k += 1
    live = [i for i in range(n) if not dead[i]]

    def assemble(s):
        pool = set(range(0, n, s)) | grid
        pool = {i for i in pool if not dead[i]}
        if live:
            pool.add(live[0])
            pool.add(live[-1])
        return pool

    pool = assemble(stride)
    capped = False
    while len(pool) > _MAX_POSE_FRAMES and stride < n:
        stride += 1
        capped = True
        pool = assemble(stride)
    info = {
        "extraction_mode": "dense_chained",
        "pose_frame_stride": stride,
        "pose_frame_corr_floor": _POSE_CORR_FLOOR,
        "pose_frame_corr_at_stride": corr_by_lag.get(stride),
        "dead_frames_dropped": int(dead.sum()),
        "pose_frames_capped": capped,
    }
    return sorted(pool), info


def _ffprobe_video_size(video: Path):
    out = subprocess.run(
        [_tool_path("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(video)],
        capture_output=True, text=True, check=True)
    stream = json.loads(out.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def _extract_frame_indices(video: Path, indices, frames_dir: Path) -> bool:
    """Extract exactly the given container-frame indices, in order, as
    frame_%05d.jpg, by streaming one full decode and saving the wanted
    frames from Python. (An ffmpeg select() expression with hundreds
    of eq() terms overflows the filter graph, and counting decoded
    frames here keeps the indices in the same numbering as the
    thumbnail pass that planned them.) False on any shortfall --
    caller falls back to the legacy fps extraction."""
    from PIL import Image

    try:
        width, height = _ffprobe_video_size(video)
    except Exception:
        return False
    frame_bytes = width * height * 3
    want = set(indices)
    last = indices[-1] if indices else -1
    written = 0
    idx = -1
    proc = subprocess.Popen(
        [_tool_path("ffmpeg"), "-v", "error", "-i", str(video),
         "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE)
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            idx += 1
            if idx in want:
                img = Image.frombytes("RGB", (width, height), buf)
                img.save(frames_dir / f"frame_{written + 1:05d}.jpg",
                         quality=92)
                written += 1
                if idx >= last:
                    break
    except Exception:
        return False
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    return written == len(indices)


def stage_extract(video: Path, job_dir: Path) -> dict:
    frames_dir = job_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    duration = _ffprobe_duration(video)
    thumbs = _gray_thumbs_from_video(video)
    plan_info = None
    if thumbs is not None and len(thumbs) >= 8:
        indices, plan_info = _plan_pose_frames(thumbs, duration)
        for stale in frames_dir.glob("frame_*.jpg"):
            stale.unlink()
        if indices and _extract_frame_indices(video, indices, frames_dir):
            frames = sorted(frames_dir.glob("frame_*.jpg"))
            manifest = {p.name: i for p, i in zip(frames, indices)}
            (job_dir / "frames_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2))
            paired = _write_frame_capture_times(video, frames, job_dir)
            info = {
                "stage": "extract",
                "video": str(video),
                "duration_seconds": round(duration, 2),
                "frame_count": len(frames),
                "frames_dir": str(frames_dir),
                **plan_info,
            }
            if paired:
                info["frame_capture_times"] = paired
            (job_dir / "job.json").write_text(
                json.dumps(info, ensure_ascii=False, indent=2))
            return info
        plan_info = None  # dense path failed; legacy extraction below
    # Legacy fallback (thumbnail decode or indexed extraction failed):
    # the pre-v0.2 uniform grid, roughly 160-320 frames -- one frame
    # every 0.25s by default, thinned for long clips. Kept verbatim so
    # a dense-path failure can never make extraction worse than it was.
    for stale in frames_dir.glob("frame_*.jpg"):
        stale.unlink()
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


# --- Two-hop PnP relocalization (poses v0.2) -------------------------------
# After incremental mapping, some frames of the dense pool still miss
# registration: their 2D-3D correspondence count starves because the
# one registered neighbour they match has too few model points of its
# own. The pioneer fix (experiments/pose_pioneer/run_exp_h2.py,
# verified on the 2026-11-03 capture: +1 clean selected frame, the one
# that took the count 51 -> 52) manufactures the missing correspondences
# by triangulation: a neighbour keypoint without a model point is
# triangulated from that neighbour's OTHER registered partners (the
# second hop), then the frame is localized by PnP. Acceptance is the
# pioneer's account: >= 8 correspondences AND >= 8 PnP inliers.
_RELOC_MIN_CORRESPONDENCES = 8
_RELOC_MIN_INLIERS = 8
_RELOC_MIN_TRI_ANGLE = math.radians(1.5)
_RELOC_MAX_REPROJ_PX = 6.0
_PAIR_ID_K = 2 ** 31 - 1


def _camera_fx_fy_cx_cy(cam):
    """(fx, fy, cx, cy) for the camera models the pipeline produces.
    Radial terms are ignored: bearings are only used for triangulation
    candidates, which a 6 px reprojection check then vets."""
    params = [float(v) for v in cam.params]
    name = str(cam.model)
    if len(params) >= 4 and ("PINHOLE" in name and "SIMPLE" not in name
                             or "OPENCV" in name):
        return params[0], params[1], params[2], params[3]
    return params[0], params[0], params[1], params[2]


def _bearing_vec(intr, xy):
    fx, fy, cx, cy = intr
    v = np.array([(xy[0] - cx) / fx, (xy[1] - cy) / fy, 1.0])
    return v / np.linalg.norm(v)


def _project_px(intr, rot, t, xyz):
    fx, fy, cx, cy = intr
    xc = rot @ np.asarray(xyz, dtype=np.float64) + t
    if xc[2] <= 0:
        return None
    return np.array([fx * xc[0] / xc[2] + cx, fy * xc[1] / xc[2] + cy])


def _triangulate_midpoint(c1, r1, c2, r2):
    """Midpoint triangulation of two rays; None when the rays are too
    parallel or the point lands behind either camera."""
    w = np.cross(r1, r2)
    if np.linalg.norm(w) < math.sin(_RELOC_MIN_TRI_ANGLE):
        return None
    a = np.column_stack([r1, -r2, w])
    try:
        s = np.linalg.solve(a, c2 - c1)
    except np.linalg.LinAlgError:
        return None
    if s[0] <= 0 or s[1] <= 0:
        return None
    return (c1 + s[0] * r1 + c2 + s[1] * r2) / 2


def _load_match_graph(db_path: Path):
    """Images, keypoints and verified pair matches from a COLMAP
    database (stdlib sqlite only)."""
    import sqlite3

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    images = {iid: (name, cam) for iid, name, cam in con.execute(
        "select image_id, name, camera_id from images")}
    keypoints = {}
    for iid, rows, cols, data in con.execute(
            "select image_id, rows, cols, data from keypoints"):
        keypoints[iid] = (
            np.frombuffer(data, dtype=np.float32).reshape(rows, cols)[:, :2]
            .astype(np.float64) if data else np.zeros((0, 2)))
    verified = {pid for (pid,) in con.execute(
        "select pair_id from two_view_geometries where rows >= 8")}
    pairs = {}
    for pid, rows, blob in con.execute(
            "select pair_id, rows, data from matches where data is not null"):
        if pid not in verified or not rows:
            continue
        a, b = pid // _PAIR_ID_K, pid % _PAIR_ID_K
        m = np.frombuffer(blob, dtype=np.uint32).reshape(rows, 2).copy()
        pairs.setdefault(a, []).append((b, m))
        pairs.setdefault(b, []).append((a, m[:, ::-1].copy()))
    con.close()
    hits = {}
    for a, lst in pairs.items():
        d = hits.setdefault(a, {})
        for b, m in lst:
            for k_a, k_b in m:
                d.setdefault(int(k_a), []).append((b, int(k_b)))
    return images, keypoints, pairs, hits


def _relocalize_unregistered(rec, db_path: Path, max_rounds: int = 4):
    """Grow the reconstruction by localizing pool frames the mapper
    left unregistered, via direct + two-hop-triangulated
    correspondences. Single-camera discipline: only frames of the
    reconstruction's one camera are eligible, and localized frames
    join that same camera -- no per-image intrinsics are ever created.

    All-or-nothing: the work happens on a disk-roundtripped copy; any
    failure discards it and the mapper's model stands. Returns
    (added image names, reconstruction to use going forward)."""
    import tempfile

    import pycolmap

    def has_pose(im):
        hp = im.has_pose
        return bool(hp() if callable(hp) else hp)

    images, keypoints, pairs, hits = _load_match_graph(db_path)
    if not images or not pairs:
        return [], rec
    cam_ids = {im.camera_id for im in rec.images.values() if has_pose(im)}
    if len(cam_ids) != 1:
        print(f"relocalization skipped: model has {len(cam_ids)} cameras, "
              "single-camera models only", flush=True)
        return [], rec
    cam_id = cam_ids.pop()
    try:
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "model"
            model_dir.mkdir()  # COLMAP's write() requires the dir to exist
            rec.write(str(model_dir))
            work = pycolmap.Reconstruction(str(model_dir))
            added_all = _relocalize_rounds(
                work, cam_id, images, keypoints, pairs, hits, max_rounds)
        return added_all, work
    except Exception as e:
        print(f"relocalization discarded ({e}); mapper model stands",
              flush=True)
        return [], rec


def _relocalize_rounds(work, cam_id, images, keypoints, pairs, hits,
                       max_rounds):
    """The rounds of _relocalize_unregistered, on the working copy.
    Raises on any inconsistency -- the caller discards the copy."""
    import pycolmap

    def has_pose(im):
        hp = im.has_pose
        return bool(hp() if callable(hp) else hp)

    added_all = []
    for _round in range(max_rounds):
        cam = work.cameras[cam_id]
        intr = _camera_fx_fy_cx_cy(cam)
        pnp_cam = pycolmap.Camera()
        pnp_cam.camera_id = cam_id
        pnp_cam.model = cam.model
        pnp_cam.width = cam.width
        pnp_cam.height = cam.height
        pnp_cam.params = [float(v) for v in cam.params]
        poses = {}
        rots = {}
        centers = {}
        for im in work.images.values():
            if not has_pose(im):
                continue
            cfw = im.cam_from_world() if callable(im.cam_from_world) \
                else im.cam_from_world
            poses[im.image_id] = cfw
            rots[im.image_id] = cfw.rotation.matrix()
            centers[im.image_id] = rots[im.image_id].T @ (-cfw.translation)

        def rebuild_pt_of():
            pt_of = {}
            for pid_, p in work.points3D.items():
                for el in p.track.elements:
                    pt_of.setdefault(el.image_id, {})[el.point2D_idx] = (
                        pid_, p.xyz)
            return pt_of

        pt_of = rebuild_pt_of()
        added = []
        for x_id, (name, x_cam_id) in sorted(images.items()):
            if x_cam_id != cam_id or x_id in poses:
                continue
            corr = {}  # k_x -> (xyz, [(img, kp), ...], point_id | None)
            for a_id, m in pairs.get(x_id, []):
                if a_id not in poses:
                    continue
                for k_x, k_a in m:
                    k_x, k_a = int(k_x), int(k_a)
                    if k_x in corr:
                        continue
                    if k_a in pt_of.get(a_id, {}):
                        pid_, xyz = pt_of[a_id][k_a]
                        corr[k_x] = (xyz, [(a_id, k_a)], pid_)
                        continue
                    for a2, k_a2 in hits.get(a_id, {}).get(k_a, []):
                        if a2 not in poses or a2 == x_id:
                            continue
                        if k_a2 in pt_of.get(a2, {}):
                            # The second hop already owns a point: that
                            # is a direct correspondence, better than a
                            # fresh triangulation. (Reusing this keypoint
                            # in a new point's track instead would trip
                            # COLMAP's point2D ownership check.)
                            pid_, xyz = pt_of[a2][k_a2]
                            corr[k_x] = (xyz, [(a2, k_a2)], pid_)
                            break
                        r1 = rots[a_id].T @ _bearing_vec(
                            intr, keypoints[a_id][k_a])
                        r2 = rots[a2].T @ _bearing_vec(
                            intr, keypoints[a2][k_a2])
                        p = _triangulate_midpoint(centers[a_id], r1,
                                                  centers[a2], r2)
                        if p is None:
                            continue
                        ok = True
                        for y, ky in ((a_id, k_a), (a2, k_a2)):
                            pr = _project_px(intr, rots[y],
                                             poses[y].translation,
                                             p)
                            if pr is None or np.linalg.norm(
                                    pr - keypoints[y][ky]) \
                                    > _RELOC_MAX_REPROJ_PX:
                                ok = False
                                break
                        if ok:
                            corr[k_x] = (p, [(a_id, k_a), (a2, k_a2)], None)
                            break
            if len(corr) < _RELOC_MIN_CORRESPONDENCES:
                continue
            ks = sorted(corr.keys())
            xy = np.array([keypoints[x_id][k] for k in ks])
            xyz = np.array([corr[k][0] for k in ks])
            est_opts = pycolmap.AbsolutePoseEstimationOptions()
            est_opts.ransac.max_error = 8.0
            est_opts.ransac.confidence = 0.999
            est_opts.ransac.min_inlier_ratio = 0.05
            res = pycolmap.estimate_and_refine_absolute_pose(
                xy, xyz, pnp_cam, est_opts)
            if res is None or res["num_inliers"] < _RELOC_MIN_INLIERS:
                continue
            img = pycolmap.Image(name=name, keypoints=keypoints[x_id],
                                 camera_id=cam_id, image_id=x_id)
            work.add_image_with_trivial_frame(img)
            stored = work.images[x_id]
            frame = work.frames[stored.frame_id]
            frame.set_cam_from_world(cam_id, res["cam_from_world"])
            work.register_frame(stored.frame_id)
            mask = res["inlier_mask"].astype(bool)
            # A source keypoint can be claimed by only one point: two
            # of this frame's keypoints may match the same keypoint of
            # a registered neighbour, and the second claim would trip
            # COLMAP's track consistency check. Such an observation
            # joins the point that already owns the source keypoint.
            claimed_by = {}
            for k, is_inl in zip(ks, mask):
                if not is_inl:
                    continue
                if corr[k][2] is not None:
                    work.add_observation(
                        corr[k][2], pycolmap.TrackElement(x_id, k))
                    continue
                elems = [(x_id, k)] + list(corr[k][1])
                conflicts = {claimed_by[e] for e in elems
                             if e in claimed_by}
                if conflicts:
                    if len(conflicts) == 1:
                        owner = conflicts.pop()
                        work.add_observation(
                            owner, pycolmap.TrackElement(x_id, k))
                        claimed_by[(x_id, k)] = owner
                    continue
                track = pycolmap.Track()
                for e_img, e_kp in elems:
                    track.add_element(e_img, e_kp)
                new_id = work.add_point3D(
                    corr[k][0], track,
                    np.array([128, 128, 128], dtype=np.uint8))
                for e in elems:
                    claimed_by[e] = new_id
            added.append(name)
            cfw = res["cam_from_world"]
            poses[x_id] = cfw
            rots[x_id] = cfw.rotation.matrix()
            centers[x_id] = rots[x_id].T @ (-cfw.translation)
            pt_of = rebuild_pt_of()
        if not added:
            break
        added_all.extend(added)
        print(f"relocalization round {_round}: +{len(added)} "
              f"{added[:8]}", flush=True)
        pycolmap.bundle_adjustment(work, pycolmap.BundleAdjustmentOptions())
    return added_all


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
    best_key, rec = max(reconstructions.items(),
                        key=lambda kv: kv[1].num_reg_images())
    # Standard mop-up (v0.2): two-hop PnP relocalization of pool frames
    # the mapper left behind. Best-effort by contract -- a failure here
    # keeps the mapper's model rather than losing a finished trajectory.
    relocalized = []
    if db_path.exists():
        try:
            relocalized, rec = _relocalize_unregistered(rec, db_path)
        except Exception as e:
            print(f"relocalization failed, keeping mapper result: {e}",
                  flush=True)
        if relocalized:
            print(f"relocalization added {len(relocalized)} frames: "
                  f"{relocalized}", flush=True)
            try:
                rec.write(str(sfm_dir / str(best_key)))
            except Exception as e:
                print(f"relocalized model not persisted: {e}", flush=True)

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
        "relocalized_images": len(relocalized),
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
