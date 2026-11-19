"""Fuse posed relative-depth maps into one colored point cloud.

Alignment: COLMAP sparse points provide (relative_depth, 1/Z_colmap) pairs;
a robust affine fit rel ~= a * (1/Z) + b converts relative inverse depth to
COLMAP-scale depth Z = a / (rel - b). With too few anchors, falls back to a
unitless inverse-depth cloud so the stage still produces a viewable result.
"""
import json
import math
from concurrent.futures import ThreadPoolExecutor
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

    Depth maps are streamed -- loaded, sampled, dropped. Image stems
    are unique, so the cache this replaces never hit twice; it only
    held every map in RAM (~6.4GB at 773 full-res frames, OOM-killed
    on a 7.7GB machine). Outputs are identical (verified value for
    value against the cached version on the 204141 product job by
    experiments/goodlight_recheck/replica_fuse.py).
    """
    import pycolmap

    sfm_dir = job_dir / "sfm"
    if not sfm_dir.exists():
        # No COLMAP output at all: no anchors, not a crash. (An
        # empty sfm/ already returned empty below; only a missing
        # dir raised FileNotFoundError from iterdir().)
        return np.empty(0), np.empty(0), {}
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
        depth = np.load(npy)
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
        del depth
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
# Collect workers for the per-frame loop (v0.6.64): each frame's
# math is independent; workers compute, the main thread appends
# in sorted-pose order. 4 matches the depth stage's worker count;
# measured on the 120-frame task before shipping (see REPORT).
_COLLECT_WORKERS = 4
_TRIM_FRAME_CODE = 1024


def _yardstick_voxel(sample_chunks) -> float | None:
    """Voxel size of the cross-frame yardstick: extent/110.

    Reference implementation (kept, untouched): stage_fuse runs the
    same math chunk-wise over the yardstick spill since v0.6.53 --
    same percentiles, same keep mask, same extent/110.

    Mirrors experiments/measure_spread.py exactly: percentile-trim the
    stride-8 sample cloud once, then take the norm of the recomputed
    1--99 percentile range over 110.
    """
    if not sample_chunks:
        return None
    pts = np.concatenate(sample_chunks)
    lo_t, hi_t = np.percentile(pts, [1, 99], axis=0)
    keep = np.all((pts >= lo_t) & (pts <= hi_t), axis=1)
    pts = pts[keep]
    if not len(pts):
        return None
    lo, hi = np.percentile(pts, [1, 99], axis=0)
    return float(np.linalg.norm(hi - lo) / 110.0)


def _corroboration_trim(points, colors, frame_ids, lo, voxel):
    """Keep points whose yardstick voxel holds >= _TRIM_MIN_FRAMES frames.

    Reference implementation of the K=2 rule (kept, untouched):
    stage_fuse runs the same rule chunk-wise over the spill since
    v0.6.52 -- same hashed keys, same distinct-frame counting,
    same keep condition.

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


def _packed_unique_inverse(keys):
    """Inverse labels of np.unique(keys, axis=0), via a packed 1-D key.

    Voxel keys span only a few hundred cells per axis on real jobs
    (measured spans 236/122/217 on the 120-frame room job), so the
    triple packs bijectively into one int64: 21 bits per axis, x in
    the high bits. Packed order IS lexicographic order on (x, y, z),
    so the grouping -- and the inverse labels -- are the same as the
    axis=0 call (verified identical on the real job: group sizes,
    200k random pairs, unpacked rows). 1-D unique runs ~16x faster
    than the axis=0 call (23.4 s -> 1.5 s on that job). Falls back
    to the axis=0 call whenever a span does not fit 21 bits.
    """
    lo = keys.min(axis=0)
    spans = keys.max(axis=0) - lo
    if np.all(spans < (1 << 21)):
        off = keys - lo
        packed = (off[:, 0] << 42) | (off[:, 1] << 21) | off[:, 2]
        _, inverse = np.unique(packed, return_inverse=True)
        return inverse
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    return inverse


_CHUNK = 1_000_000
_SPILL_TAG = ".fuse_spill_"


class _Spill:
    """One cloud spilled to disk: x/y/z float64 columns, colors
    (N,3) uint8, frame ids (N,) int32. Written per frame with
    tofile; read back chunk-wise with seek+fromfile (no mmap, so
    faulted pages do not inflate RSS)."""

    def __init__(self, job_dir: Path, tag: str):
        self.dir = Path(job_dir)
        self.paths = {
            "x": self.dir / f"{_SPILL_TAG}{tag}_x.bin",
            "y": self.dir / f"{_SPILL_TAG}{tag}_y.bin",
            "z": self.dir / f"{_SPILL_TAG}{tag}_z.bin",
            "colors": self.dir / f"{_SPILL_TAG}{tag}_colors.bin",
            "fids": self.dir / f"{_SPILL_TAG}{tag}_fids.bin",
        }
        self._wh = {k: open(p, "wb") for k, p in self.paths.items()}
        self.n = 0

    def append(self, points, colors, frame_ids):
        points = np.asarray(points, dtype=np.float64)
        self._wh["x"].write(points[:, 0].tobytes())
        self._wh["y"].write(points[:, 1].tobytes())
        self._wh["z"].write(points[:, 2].tobytes())
        self._wh["colors"].write(np.asarray(colors, dtype=np.uint8).tobytes())
        self._wh["fids"].write(np.asarray(frame_ids, dtype=np.int32).tobytes())
        self.n += len(points)

    def close_write(self):
        for h in self._wh.values():
            h.close()
        self._wh = {}

    def remove(self):
        self.close_write() if self._wh else None
        for p in self.paths.values():
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    # -- reads (each opens, seeks, reads, closes: no mapping) --
    def read_col(self, k: int) -> np.ndarray:
        name = ("x", "y", "z")[k]
        return np.fromfile(self.paths[name], dtype=np.float64)

    def read_col_slice(self, k: int, start: int, stop: int) -> np.ndarray:
        name = ("x", "y", "z")[k]
        with open(self.paths[name], "rb") as f:
            f.seek(start * 8)
            return np.fromfile(f, dtype=np.float64, count=stop - start)

    def read_rows(self, start: int, stop: int) -> np.ndarray:
        m = stop - start
        cols = []
        for k, name in enumerate(("x", "y", "z")):
            with open(self.paths[name], "rb") as f:
                f.seek(start * 8)
                cols.append(np.fromfile(f, dtype=np.float64, count=m))
        return np.stack(cols, axis=1)

    def read_colors(self, start: int, stop: int) -> np.ndarray:
        m = stop - start
        with open(self.paths["colors"], "rb") as f:
            f.seek(start * 3)
            return np.fromfile(f, dtype=np.uint8, count=m * 3).reshape(m, 3)

    def read_fids(self, start: int, stop: int) -> np.ndarray:
        m = stop - start
        with open(self.paths["fids"], "rb") as f:
            f.seek(start * 4)
            return np.fromfile(f, dtype=np.int32, count=m)

    def gather_rows(self, idx: np.ndarray) -> np.ndarray:
        return np.stack([self.read_col(k)[idx] for k in range(3)], axis=1)

    def gather_colors(self, idx: np.ndarray) -> np.ndarray:
        allc = np.fromfile(self.paths["colors"], dtype=np.uint8).reshape(-1, 3)
        return allc[idx]

    def gather_fids(self, idx: np.ndarray) -> np.ndarray:
        allf = np.fromfile(self.paths["fids"], dtype=np.int32)
        return allf[idx]


class _ColSpill:
    """One xyz-only cloud spilled to disk: x/y/z float64 columns.

    The cross-frame yardstick sample (stride 8) needs no colors
    and no frame ids -- only positions -- so it gets its own spill
    instead of riding the full _Spill (which would also write 3
    color bytes + 4 frame-id bytes per point). Same conventions as
    _Spill: written per frame with tofile, read back with
    seek+fromfile (no mmap, so faulted pages do not inflate RSS),
    hidden .fuse_spill_* files, removed by stage_fuse (the finally
    block's glob is the backstop).
    """

    def __init__(self, job_dir: Path, tag: str):
        self.dir = Path(job_dir)
        self.paths = {
            "x": self.dir / f"{_SPILL_TAG}{tag}_x.bin",
            "y": self.dir / f"{_SPILL_TAG}{tag}_y.bin",
            "z": self.dir / f"{_SPILL_TAG}{tag}_z.bin",
        }
        self._wh = {k: open(p, "wb") for k, p in self.paths.items()}
        self.n = 0

    def append(self, points):
        points = np.asarray(points, dtype=np.float64)
        self._wh["x"].write(points[:, 0].tobytes())
        self._wh["y"].write(points[:, 1].tobytes())
        self._wh["z"].write(points[:, 2].tobytes())
        self.n += len(points)

    def close_write(self):
        for h in self._wh.values():
            h.close()
        self._wh = {}

    def remove(self):
        self.close_write() if self._wh else None
        for p in self.paths.values():
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    # -- reads (each opens, seeks, reads, closes: no mapping) --
    def read_col(self, k: int) -> np.ndarray:
        name = ("x", "y", "z")[k]
        return np.fromfile(self.paths[name], dtype=np.float64)

    def read_col_slice(self, k: int, start: int, stop: int) -> np.ndarray:
        name = ("x", "y", "z")[k]
        with open(self.paths[name], "rb") as f:
            f.seek(start * 8)
            return np.fromfile(f, dtype=np.float64, count=stop - start)

    def read_rows(self, start: int, stop: int) -> np.ndarray:
        m = stop - start
        cols = []
        for k, name in enumerate(("x", "y", "z")):
            with open(self.paths[name], "rb") as f:
                f.seek(start * 8)
                cols.append(np.fromfile(f, dtype=np.float64, count=m))
        return np.stack(cols, axis=1)


def _chunks(n: int):
    for start in range(0, n, _CHUNK):
        yield start, min(n, start + _CHUNK)


def _percentile_1_99_cols(spill: _Spill):
    """(lo, hi) per axis, bit-identical to
    np.percentile(pts, [1, 99], axis=0) on the full cloud.

    The 1-D work is _percentile_1_99_1d: in-place partition, no
    copy (np.percentile copies a 76.5M column: +612 MB at 638
    frames, twice per run).
    """
    lo = np.empty(3, dtype=np.float64)
    hi = np.empty(3, dtype=np.float64)
    for k in range(3):
        col = spill.read_col(k)
        lo[k], hi[k] = _percentile_1_99_1d(col)
        del col
    return lo, hi


def _read_int64_col(path, start, stop):
    """One int64 chunk from a flat file (seek + fromfile, no mmap)."""
    with open(path, "rb") as f:
        f.seek(start * 8)
        return np.fromfile(f, dtype=np.int64, count=stop - start)

def _read_int32_col(path, start, stop):
    """One int32 chunk from a flat file (seek + fromfile, no mmap)."""
    with open(path, "rb") as f:
        f.seek(start * 4)
        return np.fromfile(f, dtype=np.int32, count=stop - start)


def _chunked_unique(get_values, n):
    """Sorted unique values of a virtual int64 array, chunk-wise.

    Exact: np.unique over the concatenation of the per-chunk
    uniques is the same set, in the same sorted order, as
    np.unique over the whole array. The concatenation is short on
    real clouds (measured 120-frame job: lin 14.67M rows -> 410K
    chunk uniques for 206K global uniques, pair codes 554K for
    553K, packed keys 1.85M for 1.26M pick voxels) -- a few MB,
    where the full array is
    612 MB at 638 frames and its sort copy another 612 MB.
    get_values(start, stop) -> int64 array of that slice.
    """
    parts = []
    for start, stop in _chunks(n):
        parts.append(np.unique(get_values(start, stop)))
    if not parts:
        return np.empty(0, dtype=np.int64)
    if len(parts) == 1:
        return parts[0]
    return np.unique(np.concatenate(parts))


def _spill_inverse(get_values, unique_sorted, n, path, accumulate=None):
    """Write per-row inverse labels (index into unique_sorted) to path.

    Exact: per-chunk np.unique gives chunk-local labels; mapping
    the chunk's sorted uniques onto the global sorted uniques with
    searchsorted gives the same labels np.unique(whole) would.
    The file is one flat int64 column, read back chunk-wise with
    _read_int64_col; stage_fuse's finally block removes it.
    accumulate, when given, is called per chunk as
    accumulate(inv_chunk, start, stop) with the global labels in
    hand -- callers fold their counting into this pass instead of
    re-reading the file (trim pair codes; see stage_fuse).
    """
    with open(path, "wb") as f:
        for start, stop in _chunks(n):
            vals = get_values(start, stop)
            u_c, inv_c = np.unique(vals, return_inverse=True)
            g = np.searchsorted(unique_sorted, u_c)
            inv_g = g[inv_c].astype(np.int64, copy=False)
            f.write(inv_g.tobytes())
            if accumulate is not None:
                accumulate(inv_g, start, stop)
            del vals, u_c, inv_c, g, inv_g


def _percentile_1_99_1d(values: np.ndarray):
    """(1st, 99th) percentiles of a 1-D float64 array, bit-identical
    to np.percentile(values, [1, 99]) with the default linear method.

    Partitions the array in place instead of copying it:
    np.percentile copies a 76.5M column (+612 MB at 638 frames,
    twice per run). The value is rebuilt following numpy's own
    arithmetic exactly -- the virtual index is (n-1)*q (numpy's
    linear method deliberately uses that form, not the
    mathematically equivalent _compute_virtual_index, to avoid
    rounding issues), and the interpolation is a + (b-a)*t for
    t < 0.5 but b - (b-a)*(1-t) for t >= 0.5 (numpy's _lerp).
    The plain a*(1-t) + b*t form differs in the last ulp on some
    inputs (caught by the unit test); this form is bit-identical
    on all of them, and ~40% faster than np.percentile.
    Callers must pass a non-empty array they own (a fresh read, or
    a gathered buffer) -- it is mutated.
    """
    n = len(values)
    m = n - 1
    kth = set()
    idx = []
    for q in (0.01, 0.99):
        vi = m * q
        i = int(math.floor(vi))
        f = vi - i
        j = min(i + 1, m)
        kth.add(i)
        kth.add(j)
        idx.append((i, j, f))
    values.partition(sorted(kth))
    out = []
    for i, j, f in idx:
        d = values[j] - values[i]
        if f >= 0.5:
            out.append(float(values[j] - d * (1.0 - f)))
        else:
            out.append(float(values[i] + d * f))
    return out[0], out[1]


def _yardstick_voxel_spill(spill) -> float | None:
    """Voxel size of the cross-frame yardstick: extent/110, streamed.

    Same math as _yardstick_voxel (percentile-trim the stride-8
    sample cloud once, then the norm of the recomputed 1--99
    percentile range over 110), but the sample lives in the
    column spill, not in a concatenated RAM table: the first
    percentile runs per 1-D column (bit-identical to the (N,3)
    call -- probed in v0.6.52), the keep mask is built chunk-wise
    (same boolean per row), and the second percentile runs per
    column over the kept values gathered chunk-wise into one array
    (same multiset in the same order; percentile sorts internally,
    so the value is identical). Returns None on an empty sample
    or an empty kept set, exactly like the reference.
    """
    if spill.n == 0:
        return None
    lo_t = np.empty(3, dtype=np.float64)
    hi_t = np.empty(3, dtype=np.float64)
    for k in range(3):
        col = spill.read_col(k)
        lo_t[k], hi_t[k] = _percentile_1_99_1d(col)
        del col
    n = spill.n
    keep = np.empty(n, dtype=bool)
    for start, stop in _chunks(n):
        rows = spill.read_rows(start, stop)
        keep[start:stop] = np.all((rows >= lo_t) & (rows <= hi_t), axis=1)
    m = int(keep.sum())
    if m == 0:
        return None
    lo = np.empty(3, dtype=np.float64)
    hi = np.empty(3, dtype=np.float64)
    for k in range(3):
        kept = np.empty(m, dtype=np.float64)
        filled = 0
        for start, stop in _chunks(n):
            seg = keep[start:stop]
            if not seg.any():
                continue
            col = spill.read_col_slice(k, start, stop)
            cnt = int(seg.sum())
            kept[filled:filled + cnt] = col[seg]
            filled += cnt
        assert filled == m, (filled, m)
        lo[k], hi[k] = _percentile_1_99_1d(kept)
        del kept
    return float(np.linalg.norm(hi - lo) / 110.0)


def _collect_frame(job):
    """One frame of stage_fuse's collect loop: pure compute, no shared state.

    `job` is (name, pose, a, b, aligned, depth_dir, frames_dir, stride).
    Returns (yard_pts, world, colors): yard_pts is this frame's
    yardstick sample (or None when empty); world/colors are the dense
    back-projected points (or None when the frame yields none). Returns
    None when the frame's files are missing. The caller appends in
    submission order, so spill order, yardstick order and frame ids
    are exactly the serial loop's. All math is verbatim from the
    serial loop -- only the loop scaffolding moved.
    """
    name, pose, a, b, aligned, depth_dir, frames_dir, stride = job
    stem = Path(name).stem
    npy = depth_dir / f"{stem}.npy"
    img_path = frames_dir / name
    if not npy.exists() or not img_path.exists():
        return None
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
    yard_pts = None
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
        yard_pts = zs_y[ok_y][:, None] * (ray_y @ rot) + (-t @ rot)

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
        return (yard_pts, None, None)
    xs_f, ys_f, zs_f = xs[ok].astype(np.float64), ys[ok].astype(np.float64), zs[ok]
    xc = (xs_f - cx) / f * zs_f
    yc = (ys_f - cy) / f * zs_f
    cam_pts = np.stack([xc, yc, zs_f], axis=1)
    world = (cam_pts - t) @ rot
    img = np.asarray(Image.open(img_path).convert("RGB"))
    colors = img[::stride, ::stride][ok]
    return (yard_pts, world, colors)


def stage_fuse(job_dir: Path, stride: int = 4) -> dict:
    """Fuse posed relative-depth maps into one colored point cloud.

Streaming form (v0.6.52; yardstick sample spilled in v0.6.53;
chunked uniques + in-place percentiles in v0.6.54): the per-frame
loop is the same math as before, but each frame's points spill to
column files in the job dir (x/y/z float64 columns + colors +
frame ids) instead of being concatenated into one giant RAM table,
and every global step runs chunked over the spill. Same inputs,
same outputs -- the 120-frame job produces byte-identical files,
and a 638-frame job peaked at 4.5 GB RSS in v0.6.53 where the old
form extrapolates to ~15 GB (it cannot run at all on this 7.7 GB
VM, and an 8 fps job would pass the desktop's 24 GB). Spill files
are hidden .fuse_spill_* files, removed in a finally block. The
yardstick sample (stride 8) has its own xyz-only column spill --
it was the largest remaining RAM structure (~480 MB at 638 frames,
~960 MB at its peak inside _yardstick_voxel). Since v0.6.64 the
per-frame collect loop runs on a small thread pool (workers
compute, the main thread appends in sorted-pose order, so
outputs are unchanged); the pool pre-flight falls back to the
serial loop when threads cannot start. Since v0.6.54 the
trim and voxel_pick uniques also run chunked: the lin / packed keys
are uniqued per 1M-row chunk and their inverse labels spill to a
flat int64 file (same labels as the whole-array call), so no full
612 MB array or sort copy is ever held; the 1--99 percentiles
partition each column in place instead of copying it. The 21-bit
packed-unique fallback (spans > 2^21 per axis) still takes the
original RAM path.
"""
    job_dir = Path(job_dir)
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

    spill = None
    yard_spill = None
    frame_index = {}
    used = 0
    try:
        spill = _Spill(job_dir, "pts")
        yard_spill = _ColSpill(job_dir, "yard")
        # ---- per-frame loop: threaded collect, in-order append ----
        # Each frame's math is independent (depth + pose + one JPEG);
        # workers compute via _collect_frame, the main thread appends
        # in sorted-pose order, so spill order, yardstick order and
        # frame ids are exactly the serial loop's. Batches bound the
        # in-flight results (one frame's points are ~3 MB). A worker
        # exception propagates the same way the serial loop's would,
        # and the finally block still removes the spills. If the pool
        # cannot start (pre-flight no-op fails), the serial loop runs
        # instead -- a job never dies for want of threads.
        jobs = []
        for name, pose in sorted(poses.items()):
            stem = Path(name).stem
            if not (depth_dir / f"{stem}.npy").exists() \
                    or not (frames_dir / name).exists():
                continue
            a, b = frame_fits.get(stem, (a_glob, b_glob))
            jobs.append((name, pose, a, b, aligned, depth_dir,
                         frames_dir, stride))

        def _append(job, res):
            nonlocal used
            if res is None:
                return
            yard_pts, world, colors = res
            if yard_pts is not None:
                yard_spill.append(yard_pts)
            if world is None:
                return
            stem = Path(job[0]).stem
            frame_id = frame_index.setdefault(stem, len(frame_index))
            spill.append(world, colors,
                         np.full(len(world), frame_id, dtype=np.int32))
            used += 1

        pool = None
        try:
            pool = ThreadPoolExecutor(max_workers=_COLLECT_WORKERS)
            pool.submit(lambda: None).result(timeout=30)
        except Exception:
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
            pool = None
        if pool is None:
            for job in jobs:
                _append(job, _collect_frame(job))
        else:
            with pool:
                batch = max(1, 2 * _COLLECT_WORKERS)
                for s in range(0, len(jobs), batch):
                    chunk = jobs[s:s + batch]
                    for job, res in zip(chunk, pool.map(_collect_frame, chunk)):
                        _append(job, res)

        spill.close_write()
        yard_spill.close_write()

        if spill.n == 0:
            info = {"stage": "fuse", "ok": False, "reason": "no points"}
            (job_dir / "job.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
            return info

        n = spill.n

        # Robust trim before voxel sizing: a small share of back-projected
        # points lands hundreds of units away (bad depth on shiny/blank areas).
        # Left in, they inflate the extent, the voxel size explodes, and the
        # real room collapses into a handful of giant voxels.
        # ---- robust trim: chunked keep mask (consumed by the trim below) ----
        lo, hi = _percentile_1_99_cols(spill)
        keep = np.empty(n, dtype=bool)
        for start, stop in _chunks(n):
            rows = spill.read_rows(start, stop)
            keep[start:stop] = np.all((rows >= lo) & (rows <= hi), axis=1)
        # (v0.6.62) No trimmed spill is written here any more: the
        # keep mask is carried into the intersection trim below,
        # which applies it on the fly -- the masked stream holds
        # exactly the rows this spill would have held, in the same
        # order. `keep` stays alive until that section consumes it.

        # Intersection-only trim: delete points no other frame corroborates
        # (see _corroboration_trim). Runs before voxel_pick, exactly as in
        # the verified experiment; skipped only when no yardstick sample
        # exists at all (degenerate input), leaving prior behavior intact.
        # ---- intersection-only trim (K=2), chunked ----
        # Fused robust trim (v0.6.62): the robust keep mask from above is
        # applied on the fly here instead of materialising a trimmed
        # spill first -- the masked stream below holds exactly the rows
        # the trimmed spill would have held, in the same order, so the
        # lin keys, uniques, inverse labels, pair codes, keep mask and
        # kept/dropped sets are all identical, and one whole spill
        # rewrite (read + write of every kept row) disappears. When the
        # robust mask keeps too few points (<1000) the old code skipped
        # the trimmed spill and ran on the full cloud; the fused form
        # does the same with rob_keep = None (identity).
        n_rob = int(keep.sum())
        rob_keep = keep if n_rob >= 1000 else None
        if rob_keep is None:
            n_rob = int(n)
        trim_info = {
            "trim_k": _TRIM_MIN_FRAMES,
            "trim_voxel_size": _yardstick_voxel_spill(yard_spill),
            "trim_points_before": int(n_rob),
            "trim_points_after": int(n_rob),
            "trim_yardstick_voxels": 0,
            "trim_voxels_with_min_frames": 0,
        }
        # The yardstick sample has served its one purpose (the trim
        # voxel size); drop its spill now to free the disk early --
        # the finally block's glob is only the backstop.
        yard_spill.remove()
        yard_spill = None
        trim_dropped = 0
        if trim_info["trim_voxel_size"] is not None:
            voxel_t = trim_info["trim_voxel_size"]
            lin_path = job_dir / f"{_SPILL_TAG}lin.bin"
            trim_fids_path = job_dir / f"{_SPILL_TAG}trim_fids.bin"
            # Pass 1: lin keys + masked frame ids, over the masked stream.
            # The frame ids ride along in masked order so the pair-code
            # pass below can read them by masked chunk without touching
            # the original spill again.
            n_trim = 0
            with open(lin_path, "wb") as lf, open(trim_fids_path, "wb") as ff:
                for start, stop in _chunks(n):
                    if rob_keep is not None:
                        m_r = rob_keep[start:stop]
                        if not m_r.any():
                            continue
                    else:
                        m_r = None
                    rows = spill.read_rows(start, stop)
                    fids = spill.read_fids(start, stop)
                    # (v0.6.66) Keys first, mask after: the voxel keys are
                    # elementwise, so computing them on the whole chunk
                    # and then indexing the 8-byte lin keys costs less
                    # than first copying the 24-byte rows with rows[m_r]
                    # and then keying the copy. Same arithmetic on the
                    # same values, so lin/fids are bit-identical; the
                    # robust mask drops ~5% of rows, whose extra key
                    # math is far cheaper than the saved copy.
                    keys = np.floor((rows - lo) / voxel_t).astype(np.int64)
                    lin = (
                        (keys[:, 0] * 73856093)
                        ^ (keys[:, 1] * 19349663)
                        ^ (keys[:, 2] * 83492791)
                    )
                    if m_r is not None:
                        lin = lin[m_r]
                        fids = fids[m_r]
                    lf.write(lin.tobytes())
                    ff.write(fids.tobytes())
                    n_trim += len(lin)
                    del rows, keys, lin, fids
            vox_id = _chunked_unique(
                lambda s, e: _read_int64_col(lin_path, s, e), n_trim
            )
            inv_trim_path = job_dir / f"{_SPILL_TAG}inv_trim.bin"
            pair_parts = []

            def _collect_pairs(inv_c, start, stop):
                pc = (
                    inv_c * _TRIM_FRAME_CODE
                    + _read_int32_col(trim_fids_path, start, stop).astype(np.int64)
                )
                pair_parts.append(np.unique(pc))

            _spill_inverse(
                lambda s, e: _read_int64_col(lin_path, s, e),
                vox_id, n_trim, inv_trim_path, accumulate=_collect_pairs)
            uniq_pairs = (
                np.unique(np.concatenate(pair_parts))
                if len(pair_parts) > 1 else pair_parts[0]
            )
            del pair_parts
            distinct = np.bincount(
                uniq_pairs // _TRIM_FRAME_CODE, minlength=len(vox_id)
            )
            del uniq_pairs
            trim_keep = np.empty(n_trim, dtype=bool)
            for start, stop in _chunks(n_trim):
                inv_c = _read_int64_col(inv_trim_path, start, stop)
                trim_keep[start:stop] = distinct[inv_c] >= _TRIM_MIN_FRAMES
                del inv_c
            trim_info.update({
                "trim_yardstick_voxels": int(len(vox_id)),
                "trim_voxels_with_min_frames": int(
                    (distinct >= _TRIM_MIN_FRAMES).sum()
                ),
                "trim_points_before": int(n_trim),
                "trim_points_after": int(trim_keep.sum()),
            })
            del distinct, vox_id
            # Keep the trim's dropped set on disk: it is the visible fringe
            # the corroboration rule cut, kept as evidence next to the kept
            # cloud (one small sidecar pair).
            dropped = ~trim_keep
            trim_dropped = int(dropped.sum())
            # One merged pass over the original chunks writes both the
            # dropped sidecar and the kept spill: each chunk's rows,
            # colors and frame ids are read once, then split once by
            # the combined robust+trim mask (v0.6.65: the trim mask is
            # expanded back to original chunk indices, so the chunk is
            # indexed once per output set instead of first copying the
            # robust-masked chunk and splitting that copy).
            # (Previously this was two separate passes -- one for
            # dropped, one for kept.)
            dropped_fids = []
            dropped_file = open(job_dir / "points_trim_dropped.bin", "wb") \
                if trim_dropped else None
            kept = _Spill(job_dir, "trim2")
            off = 0
            try:
                for start, stop in _chunks(n):
                    if rob_keep is not None:
                        m_r = rob_keep[start:stop]
                        if not m_r.any():
                            continue
                        cnt = int(m_r.sum())
                    else:
                        m_r = None
                        cnt = stop - start
                    rows = spill.read_rows(start, stop)
                    cols = spill.read_colors(start, stop)
                    fids = spill.read_fids(start, stop)
                    m_t = trim_keep[off:off + cnt]
                    off += cnt
                    if m_r is not None:
                        # (v0.6.65) Single-index split: expand the
                        # trim mask back to original chunk indices
                        # and index the chunk once per output set,
                        # instead of first copying the robust-masked
                        # chunk and then splitting that copy. Same
                        # kept/dropped rows, same order: m_k is True
                        # exactly where the robust mask and the trim
                        # mask are both True at that masked position;
                        # m_d where the robust mask is True and the
                        # trim mask False. Measured first
                        # (exp/micro_merge65.py): the merged pass
                        # copies ~half the bytes and runs ~30% faster;
                        # it does not lower peak memory -- the masked
                        # copy replaced the chunk, it did not sit
                        # beside it (see REPORT.md).
                        m_k = np.zeros(stop - start, dtype=bool)
                        m_k[m_r] = m_t
                        m_d = m_r & ~m_k
                        m_t = m_k
                    else:
                        m_d = ~m_t
                    if dropped_file is not None:
                        if m_d.any():
                            r_d = rows[m_d]
                            c_d = cols[m_d]
                            f_d = fids[m_d]
                            data = np.empty((len(r_d), 15), dtype=np.uint8)
                            data[:, 0:12] = r_d.astype(np.float32).view(np.uint8).reshape(len(r_d), 12)
                            data[:, 12:15] = c_d.astype(np.uint8)
                            dropped_file.write(data.tobytes())
                            dropped_fids.append(f_d.astype(np.int32))
                            del r_d, c_d, f_d, data
                    if m_t.any():
                        kept.append(rows[m_t], cols[m_t], fids[m_t])
                    del rows, cols, fids, m_t, m_d
                if dropped_file is not None:
                    np.save(
                        job_dir / "points_trim_dropped_frame_ids.npy",
                        np.concatenate(dropped_fids).astype(np.int32),
                    )
            finally:
                if dropped_file is not None:
                    dropped_file.close()
            del dropped_fids
            kept.close_write()
            spill.remove()
            spill = kept
            n = spill.n
            for _p in (lin_path, inv_trim_path, trim_fids_path):
                try:
                    _p.unlink()
                except FileNotFoundError:
                    pass

        # (v0.6.62) The corroboration trim can empty the cloud (a
        # job whose frames never corroborate each other); without
        # this the voxel_pick percentile below crashes on the empty
        # spill with an IndexError. Fail cleanly, like the
        # no-points early return above.
        if n == 0:
            info = {"stage": "fuse", "ok": False, "reason": "no points after trim"}
            (job_dir / "job.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
            return info

        # Voxel reduction by deterministic selection (voxel_pick, DS-written
        # and reviewed 2026-10-31): each voxel keeps the single real measured
        # point closest to the voxel centroid. The previous implementation
        # averaged all points in a voxel, which synthesizes phantom mid-surface
        # points no camera ever measured -- replaced on the user's
        # no-averaging principle.
        # ---- voxel_pick, chunked ----
        _lo, _hi = _percentile_1_99_cols(spill)
        voxel, pick_extent_robust = _pick_voxel_size(_lo, _hi)
        key_lo = np.full(3, np.iinfo(np.int64).max, dtype=np.int64)
        key_hi = np.full(3, np.iinfo(np.int64).min, dtype=np.int64)
        for start, stop in _chunks(n):
            keys = np.floor(spill.read_rows(start, stop) / voxel).astype(np.int64)
            key_lo = np.minimum(key_lo, keys.min(axis=0))
            key_hi = np.maximum(key_hi, keys.max(axis=0))
        spans = key_hi - key_lo
        inv_pick_path = None
        if np.all(spans < (1 << 21)):
            def _packed_from_rows(rows):
                keys = np.floor(rows / voxel).astype(np.int64)
                off = keys - key_lo
                return (off[:, 0] << 42) | (off[:, 1] << 21) | off[:, 2]

            def _packed_values(start, stop):
                return _packed_from_rows(spill.read_rows(start, stop))

            pick_uniq = _chunked_unique(_packed_values, n)
            inv_pick_path = job_dir / f"{_SPILL_TAG}inv_pick.bin"
            n_vox = len(pick_uniq)
            # Pass 2 is written out inline (not via _spill_inverse):
            # the rows are in hand here, so the voxel counts and the
            # centroid sums accumulate from those same rows -- the
            # same per-chunk partial sums, in the same chunk order,
            # as the separate bincount passes would produce, with no
            # extra reads of the inverse file or the columns.
            counts = np.zeros(n_vox, dtype=np.int64)
            centroid_sums = np.zeros((n_vox, 3), dtype=np.float64)
            with open(inv_pick_path, "wb") as inv_f:
                for start, stop in _chunks(n):
                    rows = spill.read_rows(start, stop)
                    vals = _packed_from_rows(rows)
                    u_c, inv_c = np.unique(vals, return_inverse=True)
                    g = np.searchsorted(pick_uniq, u_c)
                    inv_g = g[inv_c].astype(np.int64, copy=False)
                    inv_f.write(inv_g.tobytes())
                    counts += np.bincount(inv_g, minlength=n_vox)
                    for k in range(3):
                        centroid_sums[:, k] += np.bincount(
                            inv_g, weights=rows[:, k], minlength=n_vox)
                    del rows, vals, u_c, inv_c, g, inv_g
            del pick_uniq
            counts = counts.astype(np.float64)
            centroid = centroid_sums / counts[:, None]
            del centroid_sums
        else:
            # Same fallback as v0.6.51: the axis=0 call on the RAM cloud.
            pts_ram = spill.gather_rows(np.arange(n))
            keys = np.floor(pts_ram / voxel).astype(np.int64)
            _, inverse = np.unique(keys, axis=0, return_inverse=True)
            del pts_ram, keys
            counts = np.bincount(inverse).astype(np.float64)
            centroid = np.empty((len(counts), 3), dtype=np.float64)
            for k in range(3):
                col = spill.read_col(k)
                centroid[:, k] = np.bincount(inverse, weights=col) / counts
                del col
        best_d2 = np.full(len(counts), np.inf, dtype=np.float64)
        best_idx = np.full(len(counts), -1, dtype=np.int64)
        for start, stop in _chunks(n):
            rows = spill.read_rows(start, stop)
            if inv_pick_path is not None:
                inv_c = _read_int64_col(inv_pick_path, start, stop)
            else:
                inv_c = inverse[start:stop]
            d2_c = np.sum((rows - centroid[inv_c]) ** 2, axis=1)
            order = np.argsort(inv_c, kind="stable")
            inv_s = inv_c[order]
            d2_s = d2_c[order]
            starts = np.flatnonzero(
                np.concatenate(([True], inv_s[1:] != inv_s[:-1])))
            seg_min = np.minimum.reduceat(d2_s, starts)
            sizes = np.diff(np.concatenate((starts, [len(inv_c)])))
            seg = np.repeat(np.arange(len(starts)), sizes)
            is_min = d2_s == seg_min[seg]
            cand = np.where(is_min, np.arange(len(inv_c)), len(inv_c))
            first_pos = np.minimum.reduceat(cand, starts)
            local_vox = inv_s[starts]
            local_pos = order[first_pos]
            local_d2 = d2_s[first_pos]
            upd = local_d2 < best_d2[local_vox]
            best_d2[local_vox[upd]] = local_d2[upd]
            best_idx[local_vox[upd]] = start + local_pos[upd]
        assert (best_idx >= 0).all(), "unfilled voxel in streaming pick"
        assert len(best_idx) == len(counts), (len(best_idx), len(counts))
        pick = np.sort(best_idx)
        points = spill.gather_rows(pick)
        colors = spill.gather_colors(pick)
        frame_ids = spill.gather_fids(pick)
        if inv_pick_path is not None:
            try:
                inv_pick_path.unlink()
            except FileNotFoundError:
                pass
        else:
            del inverse
        del best_d2, best_idx

        # Total rigid rotation applied to the points on export by the
        # camera-pose up alignment below. Written to alignment.json so
        # downstream stages that also read poses.json (the mesh stage's
        # camera centres) can express them in the same exported frame as
        # the points. (The printed-marker leveling/scale branch that used
        # to sit here was retired with the marker-sheet feature,
        # 2026-11-04; it never ran for jobs without markers.json.)
        export_rot = None

        # Up alignment: the COLMAP world frame is arbitrary, so without
        # this step the exported scene can sit upside down in the viewer
        # (whose up is +Y) and in every other tool. World up
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
            **align_info,
        }
        (job_dir / "job.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
        return info
    finally:
        if spill is not None:
            spill.remove()
        if yard_spill is not None:
            yard_spill.remove()
        for leftover in job_dir.glob(f"{_SPILL_TAG}*"):
            try:
                leftover.unlink()
            except OSError:
                pass
