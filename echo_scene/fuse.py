"""Fuse posed relative-depth maps into one colored point cloud.

Alignment: COLMAP sparse points provide (relative_depth, 1/Z_colmap) pairs;
a robust affine fit rel ~= a * (1/Z) + b converts relative inverse depth to
COLMAP-scale depth Z = a / (rel - b). With too few anchors, falls back to a
unitless inverse-depth cloud so the stage still produces a viewable result.
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def _quat_xyzw_to_matrix(q):
    x, y, z, w = [float(v) for v in q]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _robust_mean_direction(vectors: np.ndarray) -> np.ndarray | None:
    """Robust central direction of unit vectors (deterministic).

    Component-wise median first (immune to the minority of frames where
    the camera was tilted hard up/down), then a 35-degree cone filter,
    then a plain mean of the survivors. This estimates ONE direction --
    no geometry is averaged or merged anywhere.
    """
    v = np.asarray(vectors, dtype=np.float64)
    if len(v) == 0:
        return None
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / np.where(norms > 0, norms, 1.0)
    ref = np.median(v, axis=0)
    ref_norm = float(np.linalg.norm(ref))
    if ref_norm < 1e-9:
        ref = v.mean(axis=0)
        ref_norm = float(np.linalg.norm(ref))
        if ref_norm < 1e-9:
            return None
    ref = ref / ref_norm
    keep = (v @ ref) > math.cos(math.radians(35.0))
    if keep.sum() >= max(3, len(v) // 4):
        v = v[keep]
    out = v.mean(axis=0)
    out_norm = float(np.linalg.norm(out))
    if out_norm < 1e-9:
        return None
    return out / out_norm


def _rotation_mapping(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rigid rotation (column convention) taking unit vector a to b.

    Rodrigues' formula, the same construction the marker leveling uses;
    the antiparallel case is a 180-degree turn about X.
    """
    axis = np.cross(a, b)
    axis_len = float(np.linalg.norm(axis))
    cosang = float(np.clip(a @ b, -1.0, 1.0))
    if axis_len <= 1e-9:
        if cosang > 0:
            return np.eye(3)
        return np.diag([1.0, -1.0, -1.0])
    axis = axis / axis_len
    angle = math.acos(cosang)
    kx = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + math.sin(angle) * kx + (1.0 - math.cos(angle)) * (kx @ kx)


def _estimate_world_up(poses: dict) -> np.ndarray | None:
    """World up from the job's camera poses.

    COLMAP cameras use the OpenCV convention (x right, y DOWN, z
    forward), so a camera's image-down axis in world coordinates is the
    second row of its cam_from_world rotation. A walkthrough is filmed
    roughly level, so the robust average of those axes is world down;
    up is its negation. Returns a unit vector or None.
    """
    downs = []
    for pose in poses.values():
        q = pose.get("cam_from_world_quat_xyzw")
        if q is None:
            continue
        rot = _quat_xyzw_to_matrix(q)
        downs.append(rot[1])
    down = _robust_mean_direction(np.array(downs)) if downs else None
    if down is None:
        return None
    return -down


def _estimate_up_from_imu(job_dir: Path, poses: dict):
    """World up from the job's landed IMU record, or None.

    The production LEVEL path for capture jobs: import landed the
    collector's sensor record in job_dir/sensor_record (see
    sensor_record.py), and its measured gravity -- the registered
    accelerometer mean -- IS up, outranking every camera-only
    estimate in stage_fuse. Returns (up, info) or None on any gap
    (no record, no calibration, too few pairable poses): IMU
    orientation is an upgrade, never a requirement, and declining
    is always the honest move.

    Pairing uses frame_capture_times.json (written by stage_extract)
    because extracted frames are renumbered pulls from the video.
    For stored-frame captures (cam_from_device_stored applies a
    roll) the capture's own frame table would silently pair each
    pose with the wrong instant, so without the map this declines;
    native-frame captures (Pi / simulator) pair by name as designed.
    """
    from echo_scene import sensor_record as sr

    cap = sr.read_job_capture(job_dir)
    if cap is None:
        return None
    cfd = sr.cam_from_device_stored(cap)
    if cfd is None:
        return None
    frame_times = None
    times_path = job_dir / "frame_capture_times.json"
    if times_path.exists():
        try:
            raw = json.loads(times_path.read_text())
            frame_times = {str(k): float(v) for k, v in raw.items()}
        except (OSError, ValueError, TypeError):
            frame_times = None
    if frame_times is None and sr.stored_frame_roll_deg(cap) != 0.0:
        return None
    pairs = sum(
        1 for name, pose in poses.items()
        if pose.get("cam_from_world_quat_xyzw") is not None
        and (frame_times.get(name) if frame_times is not None
             else cap.frame_time.get(name)) is not None)
    if pairs < 3:
        return None
    try:
        R_align = sr.register_imu_frame(cap, poses, cfd,
                                        frame_times=frame_times)
        up = sr.estimate_up_from_imu(cap, R_align)
    except (ValueError, AssertionError, IndexError):
        return None
    up = np.asarray(up, dtype=np.float64)
    n = float(np.linalg.norm(up))
    if not np.isfinite(n) or n < 1e-9:
        return None
    info = {"pairs": int(pairs),
            "stored_frame_roll_deg": sr.stored_frame_roll_deg(cap)}
    return up / n, info


def _view_directions(poses: dict) -> np.ndarray:
    """Camera view (forward) axes in world coordinates, one per pose.

    COLMAP cameras use the OpenCV convention (x right, y down, z
    forward), so a camera's forward axis in world coordinates is the
    third row of its cam_from_world rotation. Pose data only -- the
    point cloud is never consulted for orientation.
    """
    views = []
    for pose in poses.values():
        q = pose.get("cam_from_world_quat_xyzw")
        if q is None:
            continue
        views.append(_quat_xyzw_to_matrix(q)[2])
    if not views:
        return np.empty((0, 3))
    v = np.array(views, dtype=np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _estimate_up_great_circle(poses: dict):
    """World up from the view directions' great circle (user's method,
    2026-11-04).

    A walking shot can only turn one way: around the vertical. The
    view directions therefore sweep a great circle on the direction
    sphere -- they lie (near) a plane through the origin whose poles
    are the vertical axis. Steps:

      1. Robust great-circle fit: least-squares plane through the
         origin; the normal is the smallest singular vector of the
         view set (explicit objective: minimise sum of squared
         signed distances to the plane), refit after trimming the
         views far off the plane. The two poles are the up/down
         candidates.
      2. A binary choice, not a continuous estimate: a handheld
         walking shot looks slightly DOWN on average, so the correct
         pole is the one against which the per-view pitch (each
         view's signed elevation against the pole, averaged) comes
         out negative; the wrong pole would make the same footage
         an up-tilted shot. A near-exact tie is broken by the
         image-up cluster (and says so).
      3. Degenerate input is flagged, never dressed up as a fit:
         a near-straight walk (views cluster instead of circling),
         or a room scan whose pitch swings so widely that no single
         turning axis explains the views (per-view pitch scatter
         against the fitted pole is the judge). The pole then comes
         from the image-up cluster, still pitch-checked.

    A note on the geometry: at a steady non-zero pitch the views
    sweep a small circle, not a great one -- but its axis is the
    same vertical, and with headings covering the circle the
    least-squares plane's normal still lands on that axis. What
    actually convicts a bad fit is the pitch scatter, which is why
    it (not the raw plane residual, which just re-measures the
    mean pitch) gates the degenerate flag.

    Returns (up, info) or None when there are too few posed views.
    info carries every number a reviewer needs: singular values,
    plane residual, the signed mean pitch and down-frame fraction
    the binary choice rested on, and the angle to the image-up
    cluster estimate.
    """
    views = _view_directions(poses)
    if len(views) < 8:
        return None
    keep = np.ones(len(views), dtype=bool)
    normal = None
    sing = None
    for _ in range(4):
        _, sing, vt = np.linalg.svd(views[keep], full_matrices=False)
        normal = vt[2]
        resid = np.abs(views @ normal)
        thr = max(float(np.percentile(resid[keep], 75)) * 3.0,
                  math.sin(math.radians(12.0)))
        new_keep = resid <= thr
        if int(new_keep.sum()) < 8 or bool((new_keep == keep).all()):
            keep = new_keep if int(new_keep.sum()) >= 8 else keep
            break
        keep = new_keep
    _, sing, vt = np.linalg.svd(views[keep], full_matrices=False)
    normal = vt[2]
    # In-plane heading coverage of the kept views: a tight cluster
    # (resultant near 1) means the walk barely turned and the circle
    # is not determined by the data.
    proj = views[keep] @ vt[:2].T
    ang = np.arctan2(proj[:, 1], proj[:, 0])
    resultant = float(np.abs(np.mean(np.exp(1j * ang))))
    resid_kept = np.abs(views[keep] @ normal)
    residual_rms = float(np.degrees(np.sqrt(np.mean(
        np.arcsin(np.clip(resid_kept, -1, 1)) ** 2))))
    residual_p90 = float(np.degrees(np.arcsin(np.clip(
        np.percentile(resid_kept, 90), -1, 1))))
    # Per-view pitch against the fitted pole: steady pitch sweeps
    # a clean circle; wild pitch scatter means no single turning
    # axis explains the footage (a room scan looking everywhere).
    pitch_i = np.degrees(np.arcsin(np.clip(
        views[keep] @ normal, -1, 1)))
    pitch_scatter = float(np.std(pitch_i))
    degenerate = bool(sing[1] / sing[0] < 0.30 or resultant > 0.85
                      or pitch_scatter > 12.0)

    cluster_up = _estimate_world_up(poses)

    def pitch_stats(up_vec):
        p = np.degrees(np.arcsin(np.clip(views @ up_vec, -1, 1)))
        return float(np.mean(p)), float(np.mean(views @ up_vec < 0.0))

    info = {
        "n_views": int(len(views)),
        "n_kept": int(keep.sum()),
        "singular_values": [round(float(s), 3) for s in sing],
        "heading_resultant": round(resultant, 3),
        "residual_rms_deg": round(residual_rms, 2),
        "residual_p90_deg": round(residual_p90, 2),
        "pitch_scatter_deg": round(pitch_scatter, 2),
        "degenerate": degenerate,
        "tie_break": None,
    }
    if cluster_up is not None:
        info["cluster_agreement_deg"] = round(float(np.degrees(
            np.arccos(np.clip(abs(float(normal @ cluster_up)),
                              -1, 1)))), 2)
    if degenerate:
        # No circle to speak of: fall back to the image-up cluster
        # for the axis, but keep the binary pitch check -- footage
        # pitched clearly UP against the cluster's up means the
        # cluster picked the wrong pole.
        if cluster_up is None:
            return None
        up = cluster_up
        info["method"] = "up_cluster_fallback_degenerate"
        mean_pitch, down_frac = pitch_stats(up)
        if mean_pitch > 3.0:
            up = -up
            info["tie_break"] = "pitch_flip"
            mean_pitch, down_frac = pitch_stats(up)
        info["mean_pitch_deg"] = round(mean_pitch, 2)
        info["down_frac"] = round(down_frac, 3)
    else:
        pole = normal
        mean_pitch, down_frac = pitch_stats(pole)
        if mean_pitch > 0.0:
            pole = -pole
            mean_pitch, down_frac = pitch_stats(pole)
        if abs(mean_pitch) < 0.5 and cluster_up is not None:
            # Dead-level footage: the pitch sign carries no
            # signal, so the image-up cluster breaks the tie.
            if float(pole @ cluster_up) < 0.0:
                pole = -pole
                mean_pitch, down_frac = pitch_stats(pole)
            info["tie_break"] = "image_up"
        info["mean_pitch_deg"] = round(mean_pitch, 2)
        info["down_frac"] = round(down_frac, 3)
        up = pole
        info["method"] = "great_circle"
    up = up / np.linalg.norm(up)
    return up, info


def _collect_anchor_pairs(job_dir: Path, depth_dir: Path):
    """Return (rel_values, inv_z_values, per_frame) sampled at COLMAP points.

    per_frame maps an image stem to its own (rel, inv_z) arrays so each
    frame's depth can be aligned with its own fit when it has enough
    anchors; monocular depth scale drifts from frame to frame, and one
    global fit leaves the walls bent.
    """
    import pycolmap

    sfm_dir = job_dir / "sfm"
    models = [
        d for d in sfm_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and (d / "cameras.bin").exists()
    ]
    if not models:
        return np.empty(0), np.empty(0), {}
    rec = max(
        (pycolmap.Reconstruction(str(d)) for d in models),
        key=lambda r: r.num_reg_images(),
    )
    depth_cache = {}
    rel_vals, inv_z = [], []
    per_frame = {}
    for image in rec.images.values():
        has_pose = image.has_pose() if callable(image.has_pose) else image.has_pose
        if not has_pose:
            continue
        stem = Path(image.name).stem
        npy = depth_dir / f"{stem}.npy"
        if not npy.exists():
            continue
        if stem not in depth_cache:
            depth_cache[stem] = np.load(npy)
        depth = depth_cache[stem]
        h, w = depth.shape
        cfw = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
        q = cfw.rotation.quat
        rot = _quat_xyzw_to_matrix(q)
        t = np.array([float(v) for v in cfw.translation], dtype=np.float64)
        f_rel, f_inv = [], []
        for p2d in image.points2D:
            if p2d.point3D_id < 0 or p2d.point3D_id not in rec.points3D:
                continue
            xyz = np.array(rec.points3D[p2d.point3D_id].xyz, dtype=np.float64)
            xc = rot @ xyz + t
            z = xc[2]
            if z <= 0.05:
                continue
            px, py = int(round(p2d.xy[0])), int(round(p2d.xy[1]))
            if 0 <= px < w and 0 <= py < h:
                f_rel.append(float(depth[py, px]))
                f_inv.append(1.0 / z)
        if f_rel:
            per_frame[stem] = (np.array(f_rel), np.array(f_inv))
            rel_vals.extend(f_rel)
            inv_z.extend(f_inv)
    return np.array(rel_vals), np.array(inv_z), per_frame


def _robust_affine_fit(x, y):
    """Fit y ~= a*x + b with one round of median-residual trimming."""
    a, b = np.polyfit(x, y, 1)
    for _ in range(2):
        resid = np.abs(y - (a * x + b))
        keep = resid <= max(3.0 * np.median(resid), 1e-9)
        if keep.sum() < max(10, len(x) // 4):
            break
        a, b = np.polyfit(x[keep], y[keep], 1)
    return float(a), float(b)


def _write_binary_ply(points, colors, path: Path):
    n = len(points)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode()
    data = np.empty((n, 15), dtype=np.uint8)
    data[:, 0:12] = points.astype(np.float32).view(np.uint8).reshape(n, 12)
    data[:, 12:15] = colors.astype(np.uint8)
    with open(path, "wb") as f:
        f.write(header)
        data.tofile(f)


def _write_points_bin(points, colors, path: Path):
    n = len(points)
    data = np.empty((n, 15), dtype=np.uint8)
    data[:, 0:12] = points.astype(np.float32).view(np.uint8).reshape(n, 12)
    data[:, 12:15] = colors.astype(np.uint8)
    data.tofile(path)


def _fit_mad(x, y, a, b):
    return float(np.median(np.abs(y - (a * x + b))))


# Intersection-only trim (adopted into production 2026-10-31 after the
# room experiment in experiments/trim_report.md): a fused point survives
# only when its yardstick voxel contains points from at least two distinct
# frames. This is an evidence threshold, not an average -- no point value
# is blended or moved; uncorroborated single-view fringe is deleted.
_TRIM_MIN_FRAMES = 2
_TRIM_YARDSTICK_STRIDE = 8
_TRIM_FRAME_CODE = 1024


def _yardstick_voxel(sample_chunks) -> float | None:
    """Voxel size of the cross-frame yardstick: extent/110.

    Mirrors experiments/measure_spread.py exactly: percentile-trim the
    stride-8 sample cloud once, then take the norm of the recomputed
    1--99 percentile range over 110.
    """
    if not sample_chunks:
        return None
    pts = np.concatenate(sample_chunks)
    lo_t = np.percentile(pts, 1, axis=0)
    hi_t = np.percentile(pts, 99, axis=0)
    keep = np.all((pts >= lo_t) & (pts <= hi_t), axis=1)
    pts = pts[keep]
    if not len(pts):
        return None
    lo = np.percentile(pts, 1, axis=0)
    hi = np.percentile(pts, 99, axis=0)
    return float(np.linalg.norm(hi - lo) / 110.0)


def _corroboration_trim(points, colors, frame_ids, lo, voxel):
    """Keep points whose yardstick voxel holds >= _TRIM_MIN_FRAMES frames.

    Mirrors experiments/trim_fuse.py's corroboration_filter exactly
    (same hashed voxel keys, same distinct-frame counting).
    """
    keys = np.floor((points - lo) / voxel).astype(np.int64)
    lin = (
        (keys[:, 0] * 73856093)
        ^ (keys[:, 1] * 19349663)
        ^ (keys[:, 2] * 83492791)
    )
    vox_id, inverse = np.unique(lin, return_inverse=True)
    pair_code = inverse.astype(np.int64) * _TRIM_FRAME_CODE + frame_ids.astype(np.int64)
    uniq_pairs = np.unique(pair_code)
    distinct = np.bincount(
        uniq_pairs // _TRIM_FRAME_CODE, minlength=len(vox_id)
    )
    keep = distinct[inverse] >= _TRIM_MIN_FRAMES
    stats = {
        "trim_yardstick_voxels": int(len(vox_id)),
        "trim_voxels_with_min_frames": int(
            (distinct >= _TRIM_MIN_FRAMES).sum()
        ),
        "trim_points_before": int(len(points)),
        "trim_points_after": int(keep.sum()),
    }
    return points[keep], colors[keep], keep, stats


def _pick_voxel_size(lo, hi) -> tuple[float, float]:
    """Voxel pitch for the voxel_pick reduction -> (pitch, robust extent).

    The pitch is the deliverable budget, not a sampling estimate: the
    trimmed cloud oversamples every candidate pitch by an order of
    magnitude, and occupied cells grow ~ (extent / pitch)^2, so the
    divisor is what caps the final cloud (and with it the frozen
    solidification deliverable, one cube per kept point). Two measured
    corrections over the original raw-extent / 220, a constant
    calibrated in the 52-pose era when it produced the reference
    cloud:

      * the extent is the trimmed cloud's robust 1-99% box diagonal
        (percentiles taken on the cloud being picked, at the pick
        site), not the raw max/min diagonal. On the 195-pose product
        job (pose productization, 2026-11-04) the raw diagonal
        (162.0) ran 22% above the robust one (132.3), inflated by a
        few far corroborated sheets -- while the pose chain delivered
        4.1x the corroborated points of the reference job, the pitch
        grew with the inflated extent and the final cloud stalled at
        baseline size (287,761 points, median NN 0.52, 2.1x coarser
        than the reference product's granularity).
      * divisor 250 on that robust extent lands the product job at
        pitch ~0.53: final count ~2x, median NN 0.38 (parity with
        the reference product, scale-corrected), old/new coverage
        granularity in parity, stacking fraction and fixed-radius
        surface thickness flat, same-view renders ghost-free -- at
        ~2x the solidification deliverable. /400-class pitches were
        measured in the same study (experiments/pose_productize
        voxel comparison) and rejected on deliverable size alone.
    """
    robust_extent = float(
        np.linalg.norm(
            np.asarray(hi, dtype=np.float64)
            - np.asarray(lo, dtype=np.float64)
        )
    )
    return max(robust_extent / 250.0, 1e-6), robust_extent


def stage_fuse(job_dir: Path, stride: int = 4) -> dict:
    poses = json.loads((job_dir / "poses.json").read_text())
    depth_dir = job_dir / "depth"
    frames_dir = job_dir / "frames"

    rel_vals, inv_z, per_frame = _collect_anchor_pairs(job_dir, depth_dir)
    aligned = len(rel_vals) >= 30
    if aligned:
        a_glob, b_glob = _robust_affine_fit(inv_z, rel_vals)
    else:
        a_glob, b_glob = 1.0, 0.0
    # Per-frame fits: accepted only with enough anchors and a fit at least
    # as good as the global one on that frame's own anchors.
    frame_fits = {}
    if aligned:
        for stem, (f_rel, f_inv) in per_frame.items():
            if len(f_rel) < 60:
                continue
            a_i, b_i = _robust_affine_fit(f_inv, f_rel)
            if a_i <= 0:
                continue
            if _fit_mad(f_inv, f_rel, a_i, b_i) <= _fit_mad(f_inv, f_rel, a_glob, b_glob):
                frame_fits[stem] = (a_i, b_i)

    all_points, all_colors, all_frame_ids = [], [], []
    yardstick_chunks = []
    frame_index = {}
    used = 0
    for name, pose in sorted(poses.items()):
        stem = Path(name).stem
        npy = depth_dir / f"{stem}.npy"
        img_path = frames_dir / name
        if not npy.exists() or not img_path.exists():
            continue
        a, b = frame_fits.get(stem, (a_glob, b_glob))
        rel = np.load(npy).astype(np.float64)
        if aligned:
            denom = rel - b
            valid = denom > 1e-4
            z_map = np.where(valid, a / np.where(valid, denom, 1.0), np.nan)
            z_hi = np.nanpercentile(z_map[valid], 99) if valid.any() else np.nan
            valid &= z_map < z_hi * 1.5
        else:
            z_map = 1.0 / np.maximum(rel, 1e-3)
            med = np.median(z_map)
            valid = (z_map > med * 0.2) & (z_map < med * 5.0)
        cam = pose["camera"]["params"]
        f, cx, cy = float(cam[0]), float(cam[1]), float(cam[2])
        rot = _quat_xyzw_to_matrix(pose["cam_from_world_quat_xyzw"])
        t = np.array(pose["cam_from_world_t"], dtype=np.float64)

        # Cross-frame yardstick sample (stride 8, ray form, exactly as in
        # experiments/measure_spread.py): collected alongside the dense
        # pass so the trim voxel below is the measured yardstick size,
        # not a value derived from the cloud being trimmed.
        ystride = _TRIM_YARDSTICK_STRIDE
        ys_y, xs_y = np.mgrid[0:rel.shape[0]:ystride, 0:rel.shape[1]:ystride]
        zs_y = z_map[::ystride, ::ystride]
        ok_y = valid[::ystride, ::ystride] & np.isfinite(zs_y)
        if zs_y.shape[0] > 1 and zs_y.shape[1] > 1:
            jump_y = np.zeros_like(zs_y)
            jump_y[:, :-1] = np.maximum(
                jump_y[:, :-1], np.abs(zs_y[:, 1:] - zs_y[:, :-1])
            )
            jump_y[:-1, :] = np.maximum(
                jump_y[:-1, :], np.abs(zs_y[1:, :] - zs_y[:-1, :])
            )
            ok_y &= jump_y <= 0.18 * np.maximum(zs_y, 1e-6)
        if ok_y.any():
            xs_yf = xs_y[ok_y].astype(np.float64)
            ys_yf = ys_y[ok_y].astype(np.float64)
            ray_y = np.stack(
                [(xs_yf - cx) / f, (ys_yf - cy) / f, np.ones_like(xs_yf)],
                axis=1,
            )
            yardstick_chunks.append(
                zs_y[ok_y][:, None] * (ray_y @ rot) + (-t @ rot)
            )

        ys, xs = np.mgrid[0:rel.shape[0]:stride, 0:rel.shape[1]:stride]
        zs = z_map[::stride, ::stride]
        ok = valid[::stride, ::stride] & np.isfinite(zs)
        # Depth-edge rejection: pixels whose depth jumps sharply against a
        # neighbour sit on a silhouette edge; monocular depth smears there
        # and back-projects into stretched curtains between surfaces.
        if zs.shape[0] > 1 and zs.shape[1] > 1:
            jump = np.zeros_like(zs)
            jump[:, :-1] = np.maximum(jump[:, :-1], np.abs(zs[:, 1:] - zs[:, :-1]))
            jump[:-1, :] = np.maximum(jump[:-1, :], np.abs(zs[1:, :] - zs[:-1, :]))
            ok &= jump <= 0.18 * np.maximum(zs, 1e-6)
        if not ok.any():
            continue
        xs_f, ys_f, zs_f = xs[ok].astype(np.float64), ys[ok].astype(np.float64), zs[ok]
        xc = (xs_f - cx) / f * zs_f
        yc = (ys_f - cy) / f * zs_f
        cam_pts = np.stack([xc, yc, zs_f], axis=1)
        world = (cam_pts - t) @ rot  # rows: R^T (Xc - t)
        img = np.asarray(Image.open(img_path).convert("RGB"))
        colors = img[::stride, ::stride][ok]
        frame_id = frame_index.setdefault(stem, len(frame_index))
        all_points.append(world)
        all_colors.append(colors)
        all_frame_ids.append(np.full(len(world), frame_id, dtype=np.int32))
        used += 1

    if not all_points:
        info = {"stage": "fuse", "ok": False, "reason": "no points"}
        (job_dir / "job.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
        return info

    points = np.concatenate(all_points)
    colors = np.concatenate(all_colors)
    frame_ids = np.concatenate(all_frame_ids)
    # Robust trim before voxel sizing: a small share of back-projected
    # points lands hundreds of units away (bad depth on shiny/blank areas).
    # Left in, they inflate the extent, the voxel size explodes, and the
    # real room collapses into a handful of giant voxels.
    lo = np.percentile(points, 1, axis=0)
    hi = np.percentile(points, 99, axis=0)
    keep = np.all((points >= lo) & (points <= hi), axis=1)
    if keep.sum() >= 1000:
        points, colors, frame_ids = points[keep], colors[keep], frame_ids[keep]
    # Intersection-only trim: delete points no other frame corroborates
    # (see _corroboration_trim). Runs before voxel_pick, exactly as in
    # the verified experiment; skipped only when no yardstick sample
    # exists at all (degenerate input), leaving prior behavior intact.
    trim_info = {
        "trim_k": _TRIM_MIN_FRAMES,
        "trim_voxel_size": _yardstick_voxel(yardstick_chunks),
        "trim_points_before": int(len(points)),
        "trim_points_after": int(len(points)),
        "trim_yardstick_voxels": 0,
        "trim_voxels_with_min_frames": 0,
    }
    trim_dropped = 0
    if trim_info["trim_voxel_size"] is not None:
        pre_points, pre_colors, pre_ids = points, colors, frame_ids
        points, colors, trim_keep, trim_stats = _corroboration_trim(
            points, colors, frame_ids, lo, trim_info["trim_voxel_size"]
        )
        frame_ids = frame_ids[trim_keep]
        trim_info.update(trim_stats)
        # Keep the trim's dropped set on disk: it is the visible fringe
        # the corroboration rule cut, kept as evidence next to the kept
        # cloud (one small sidecar pair).
        dropped = ~trim_keep
        trim_dropped = int(dropped.sum())
        if trim_dropped:
            _write_points_bin(
                pre_points[dropped],
                pre_colors[dropped],
                job_dir / "points_trim_dropped.bin",
            )
            np.save(
                job_dir / "points_trim_dropped_frame_ids.npy",
                pre_ids[dropped].astype(np.int32),
            )
    del yardstick_chunks
    # Voxel reduction by deterministic selection (voxel_pick, DS-written
    # and reviewed 2026-10-31): each voxel keeps the single real measured
    # point closest to the voxel centroid. The previous implementation
    # averaged all points in a voxel, which synthesizes phantom mid-surface
    # points no camera ever measured -- replaced on the user's
    # no-averaging principle.
    voxel, pick_extent_robust = _pick_voxel_size(
        np.percentile(points, 1, axis=0), np.percentile(points, 99, axis=0)
    )
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    centroid = np.empty((len(counts), 3), dtype=np.float64)
    for k in range(3):
        centroid[:, k] = np.bincount(inverse, weights=points[:, k]) / counts
    d2 = np.sum((points - centroid[inverse]) ** 2, axis=1)
    # Sort primarily by voxel, then by distance, then by original index:
    # only then are each voxel's points contiguous and "first of group"
    # is the closest-to-centroid point (ties -> lowest index). (A first
    # version sorted by distance alone and selected nearly every point;
    # the group-contiguity is the whole trick.)
    order = np.lexsort((np.arange(len(points)), d2, inverse))
    inv_sorted = inverse[order]
    first = np.ones(len(points), dtype=bool)
    first[1:] = inv_sorted[1:] != inv_sorted[:-1]
    pick = np.sort(order[first])
    assert len(pick) == len(counts), (len(pick), len(counts))
    points, colors = points[pick], colors[pick]
    frame_ids = frame_ids[pick]

    # Printed-marker calibration (optional): when the markers stage found
    # the printed sheet, level the cloud so the floor normal becomes +Y
    # (the viewer's up axis, see markers.py) and scale it into meters.
    # Without markers.json this block never runs and output is unchanged.
    marker_info = {}
    # Total rigid rotation applied to the points on export (marker
    # leveling below, or the camera-pose up alignment after it). Written
    # to alignment.json so downstream stages that also read poses.json
    # (the mesh stage's camera centres) can express them in the same
    # exported frame as the points.
    export_rot = None
    markers_file = job_dir / "markers.json"
    if markers_file.exists():
        try:
            markers = json.loads(markers_file.read_text())
        except (OSError, ValueError):
            markers = {}
        floor_normal = markers.get("floor_normal_world")
        if floor_normal:
            n = np.asarray(floor_normal, dtype=np.float64)
            n = n / np.linalg.norm(n)
            up = np.array([0.0, 1.0, 0.0])
            axis = np.cross(n, up)
            axis_len = float(np.linalg.norm(axis))
            if axis_len > 1e-9:
                axis = axis / axis_len
                angle = math.acos(float(np.clip(n @ up, -1.0, 1.0)))
                kx = np.array(
                    [
                        [0.0, -axis[2], axis[1]],
                        [axis[2], 0.0, -axis[0]],
                        [-axis[1], axis[0], 0.0],
                    ]
                )
                rot_align = (
                    np.eye(3)
                    + math.sin(angle) * kx
                    + (1.0 - math.cos(angle)) * (kx @ kx)
                )
                points = points @ rot_align.T
                export_rot = rot_align
            elif float(n @ up) < 0:
                flip = np.diag([1.0, -1.0, -1.0])
                points = points @ flip.T
                export_rot = flip
            marker_info["marker_leveled"] = True
        marker_scale = markers.get("scale_to_meters")
        if marker_scale:
            points = points * float(marker_scale)
            marker_info["marker_scale_to_meters"] = float(marker_scale)

    # Up alignment: the COLMAP world frame is arbitrary, so without
    # this step the exported scene can sit upside down in the viewer
    # (whose up is +Y) and in every other tool. When the marker stage
    # did not already level the cloud from a measured floor, world up
    # comes first from the job's IMU record when import landed one
    # (measured gravity -- see _estimate_up_from_imu); only capture
    # jobs without a usable record fall back to the poses alone
    # (never the cloud): the great-circle fit over the view
    # directions (user's method, 2026-11-04 -- see
    # _estimate_up_great_circle), with the image-up cluster as its
    # degenerate-input fallback. The whole cloud is then rotated
    # rigidly so that up lands on +Y. One rotation applied to every
    # point -- nothing is moved relative to anything else, and the
    # trim / voxel_pick results above are untouched. Runs after
    # voxel_pick, so the mesh and voxel stages (which consume
    # points.bin) inherit the upright frame.
    align_info = {}
    if export_rot is None:
        up_est = None
        imu_est = _estimate_up_from_imu(job_dir, poses)
        if imu_est is not None:
            up_est, imu_info = imu_est
            align_info["up_method"] = "imu"
            align_info["imu"] = imu_info
        if up_est is None:
            gc = _estimate_up_great_circle(poses)
            if gc is not None:
                up_est, gc_info = gc
                align_info["up_method"] = gc_info.pop("method")
                align_info["great_circle"] = gc_info
        if up_est is None:
            up_est = _estimate_world_up(poses)
            if up_est is not None:
                align_info["up_method"] = "up_cluster"
        if up_est is not None:
            export_rot = _rotation_mapping(up_est, np.array([0.0, 1.0, 0.0]))
            points = points @ export_rot.T
            align_info["up_alignment"] = (
                "imu_gravity" if imu_est is not None else "camera_poses")
            align_info["estimated_up_world"] = [float(v) for v in up_est]
    alignment_file = job_dir / "alignment.json"
    if export_rot is not None:
        alignment_file.write_text(
            json.dumps(
                {
                    "rotation": [
                        [float(v) for v in row] for row in export_rot
                    ],
                    "convention": (
                        "column-convention rotation mapping the COLMAP "
                        "world frame to the exported frame; row-vector "
                        "points transform as p @ rotation.T"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        align_info["alignment_json"] = str(alignment_file)
    elif alignment_file.exists():
        # No rotation this run -- a stale sidecar would silently tilt
        # the mesh stage's camera centres against the new points.
        alignment_file.unlink()

    # Per-point source-frame provenance for the process preview: the id
    # of the frame each exported point came from, in points.bin order,
    # plus the id -> frame-stem table. Ids are assigned in fused order
    # and only to frames that contributed points, so the table cannot
    # be re-derived reliably afterwards -- it is written here, once.
    np.save(job_dir / "points_frame_ids.npy", frame_ids.astype(np.int32))
    frame_index_path = job_dir / "frame_index.json"
    frame_index_path.write_text(
        json.dumps(
            {str(fid): stem for stem, fid in frame_index.items()},
            ensure_ascii=False,
            indent=2,
        )
    )

    _write_binary_ply(points, colors, job_dir / "scene_points.ply")
    _write_points_bin(points, colors, job_dir / "points.bin")
    info = {
        "stage": "fuse",
        "ok": True,
        "aligned_to_colmap_scale": aligned,
        "anchor_pairs": int(len(rel_vals)),
        "per_frame_fits": int(len(frame_fits)),
        "frames_used": used,
        "points": int(len(points)),
        "pick_voxel_size": voxel,
        "pick_extent_robust": pick_extent_robust,
        "trim_dropped_points": trim_dropped,
        "points_frame_ids": str(job_dir / "points_frame_ids.npy"),
        "frame_index": str(frame_index_path),
        **trim_info,
        "scene_points_ply": str(job_dir / "scene_points.ply"),
        "points_bin": str(job_dir / "points.bin"),
        **marker_info,
        **align_info,
    }
    (job_dir / "job.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return info
