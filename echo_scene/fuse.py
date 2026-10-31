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
    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    voxel = max(extent / 220.0, 1e-6)
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

    # Up alignment (camera-estimated): the COLMAP world frame is
    # arbitrary, so without this step the exported scene can sit upside
    # down in the viewer (whose up is +Y) and in every other tool. When
    # the marker stage did not already level the cloud from a measured
    # floor, estimate world up from the camera poses (a roughly level
    # walk makes the cameras' image-down axes cluster around gravity)
    # and rotate the whole cloud rigidly so that up lands on +Y. One
    # rotation applied to every point -- nothing is moved relative to
    # anything else, and the trim / voxel_pick results above are
    # untouched. Runs after voxel_pick, so the mesh and voxel stages
    # (which consume points.bin) inherit the upright frame.
    align_info = {}
    if export_rot is None:
        up_est = _estimate_world_up(poses)
        if up_est is not None:
            export_rot = _rotation_mapping(up_est, np.array([0.0, 1.0, 0.0]))
            points = points @ export_rot.T
            align_info = {
                "up_alignment": "camera_poses",
                "estimated_up_world": [float(v) for v in up_est],
            }
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
