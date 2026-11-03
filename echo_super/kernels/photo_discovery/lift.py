"""Stage 3/4 — lift masks into the job's 3D frame, solidify against
the job cloud, and isolate instances. Pure numpy/scipy port of
experiments/discovery_photo/{lift,lift2,lift3_cache,finalize3}.py —
the math is the experiments' math, value for value:

  * depth alignment reuses echo_scene.fuse's robust affine fits
    (rel-depth -> COLMAP Z: global fit + per-frame fits kept when
    they beat the global MAD), exactly the fusion that produced the
    job's scene_points.ply;
  * a mask's pixels are first cut to the mask's own main depth
    cluster (lift2's depth crop);
  * lifted points pass three gates (lift.py): inside the cloud's
    1-99% room envelope (+5% pad), within 0.15 units of a measured
    cloud surface, and 2-frame corroboration — a 0.1 voxel must be
    hit from >= 2 distinct frames, the project's own solidification
    rule (same as the fusion trim);
  * the delivered set is deduplicated on a 0.02 voxel grid;
  * instance isolation (finalize3): the corroborated set is split
    into 0.3-voxel connected components and only the largest is
    delivered. Round 3 proved mask-level cleanup vacuous on this
    data (SAM eats object+surroundings as one blob, in 2D and per
    frame in 3D); instances only separate in the multi-frame
    corroborated set, where the object is consistent and the eaten
    surroundings rotate frame to frame.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .config import DiscoveryConfig

_FUSE = None


def _fuse():
    """echo_scene.fuse, imported lazily (read-only reuse of its fit
    helpers — the experiments did the same)."""
    global _FUSE
    if _FUSE is None:
        from echo_scene import fuse
        _FUSE = fuse
    return _FUSE


def read_ply_points(path) -> np.ndarray:
    """Read the (x,y,z) float32 triples of a fused-scene PLY (the
    experiments' reader: 15-byte records, xyz first)."""
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += f.read(1)
        n = int([ln for ln in header.decode().splitlines()
                 if ln.startswith("element vertex")][0].split()[-1])
        data = np.fromfile(f, dtype=np.uint8, count=n * 15).reshape(n, 15)
    return data[:, 0:12].view(np.float32).reshape(n, 3).astype(np.float64)


def fit_depth_alignment(job_dir, depth_dir):
    """Returns (fits, a_glob, b_glob): per-frame (a, b) affine fits
    where they beat the global one, else the global pair."""
    F = _fuse()
    rel_vals, inv_z, per_frame = F._collect_anchor_pairs(job_dir, depth_dir)
    aligned = len(rel_vals) >= 30
    a_glob, b_glob = (F._robust_affine_fit(inv_z, rel_vals)
                      if aligned else (1.0, 0.0))
    fits = {}
    if aligned:
        for stem, (f_rel, f_inv) in per_frame.items():
            if len(f_rel) < 60:
                continue
            a_i, b_i = F._robust_affine_fit(f_inv, f_rel)
            if a_i <= 0:
                continue
            if F._fit_mad(f_inv, f_rel, a_i, b_i) <= \
                    F._fit_mad(f_inv, f_rel, a_glob, b_glob):
                fits[stem] = (a_i, b_i)
    return fits, a_glob, b_glob


def z_map_from_rel(rel, a, b):
    """Aligned metric Z map + validity mask from one frame's relative
    depth (lift.py's z_map_for, incl. the 0.18 depth-edge rule the
    fusion's dense pass uses)."""
    rel = rel.astype(np.float64)
    denom = rel - b
    valid = denom > 1e-4
    z = np.where(valid, a / np.where(valid, denom, 1.0), np.nan)
    if valid.any():
        z_hi = np.nanpercentile(z[valid], 99)
        valid &= z < z_hi * 1.5
    z = np.where(valid, z, np.nan)
    jump = np.zeros_like(z)
    dz = np.abs(np.diff(z, axis=1))
    jump[:, :-1] = np.maximum(jump[:, :-1], np.where(np.isfinite(dz), dz, 0))
    dz = np.abs(np.diff(z, axis=0))
    jump[:-1, :] = np.maximum(jump[:-1, :], np.where(np.isfinite(dz), dz, 0))
    ok = np.isfinite(z) & (jump <= 0.18 * np.maximum(z, 1e-6))
    return z, ok


def depth_crop(zs, cfg: DiscoveryConfig):
    """Keep only the pixels in the mask's main depth cluster.
    Returns (keep_bool, stats). (lift2.depth_crop, verbatim rule.)"""
    n = len(zs)
    if n < cfg.crop_min_px:
        return np.ones(n, bool), {"crop_applied": False,
                                  "reason": "too_few_px"}
    lo, hi = np.percentile(zs, cfg.crop_pct_lo), np.percentile(
        zs, cfg.crop_pct_hi)
    if hi <= lo:
        return np.ones(n, bool), {"crop_applied": False,
                                  "reason": "flat_depth"}
    counts, edges = np.histogram(zs, bins=cfg.crop_bins, range=(lo, hi))
    p = int(np.argmax(counts))
    c = 0.5 * (edges[p] + edges[p + 1])
    binw = (hi - lo) / float(cfg.crop_bins)
    w = max(cfg.crop_window_bins * binw, cfg.crop_window_frac * c)
    keep = np.abs(zs - c) <= w
    return keep, {"crop_applied": True, "peak_z": round(float(c), 3),
                  "window": round(float(w), 3),
                  "peak_share": round(
                      float(counts[p] / max(counts.sum(), 1)), 3),
                  "kept_frac": round(float(keep.mean()), 3)}


def lift_pixels(ys, xs, z, pose, r_align):
    """Unproject pixels through the frame's pose into the exported
    (scene_points.ply) frame. (lift3_cache.lift_pixels, verbatim.)"""
    F = _fuse()
    zs = z[ys, xs]
    f, cx, cy = [float(v) for v in pose["camera"]["params"][:3]]
    rot = F._quat_xyzw_to_matrix(pose["cam_from_world_quat_xyzw"])
    t = np.array(pose["cam_from_world_t"], dtype=np.float64)
    cam_pts = np.stack([(xs - cx) / f * zs, (ys - cy) / f * zs, zs], axis=1)
    world = (cam_pts - t) @ rot
    return world @ r_align.T


def dedup(pts, voxel):
    if len(pts) == 0:
        return pts
    keys = np.floor(pts / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(first)]


def gates_and_corroborate(chunks, cloud, tree, room_lo, room_hi,
                          cfg: DiscoveryConfig):
    """chunks: [(frame_stem, points), ...] for one class.
    Returns (stats, strong_points): room gate -> NN surface gate ->
    corroboration (>= cfg.min_frames distinct frames per corr_voxel)
    -> dedup. (lift3_cache.gates_and_corroborate, verbatim math.)"""
    pts = np.concatenate([c[1] for c in chunks])
    frame_of = np.concatenate([np.full(len(c[1]), i)
                               for i, c in enumerate(chunks)])
    n_raw = len(pts)
    room_pad = cfg.room_pad_frac * (room_hi - room_lo)
    in_room = np.all((pts >= room_lo - room_pad)
                     & (pts <= room_hi + room_pad), axis=1)
    pts, frame_of = pts[in_room], frame_of[in_room]
    n_in_room = len(pts)
    if len(pts):
        dist, _ = tree.query(pts, k=1)
        on_surface = dist <= cfg.surface_nn_max
    else:
        on_surface = np.zeros(0, bool)
    pts, frame_of = pts[on_surface], frame_of[on_surface]
    stats = {"points_raw_lifted": int(n_raw),
             "points_in_room": int(n_in_room),
             "points_on_surface": int(on_surface.sum()),
             "points_corroborated_2frames": 0,
             "points_dedup": 0}
    if len(pts) == 0:
        return stats, pts
    keys = np.floor(pts / cfg.corr_voxel).astype(np.int64)
    lin = (keys[:, 0] * 73856093) ^ (keys[:, 1] * 19349663) \
        ^ (keys[:, 2] * 83492791)
    vox_id, inverse = np.unique(lin, return_inverse=True)
    pair = inverse.astype(np.int64) * 1024 + frame_of.astype(np.int64)
    uniq_pairs = np.unique(pair)
    distinct = np.bincount(uniq_pairs // 1024, minlength=len(vox_id))
    corr = distinct[inverse] >= cfg.min_frames
    strong = dedup(pts[corr], cfg.dedup_voxel)
    stats["points_corroborated_2frames"] = int(corr.sum())
    stats["points_dedup"] = int(len(strong))
    return stats, strong


def largest_component(pts, voxel):
    """Largest connected component of the point set on a voxel grid
    (26-connectivity). (finalize3.largest_comp, verbatim.)"""
    if len(pts) == 0:
        return pts
    keys = np.floor(pts / voxel).astype(np.int64)
    origin = keys.min(axis=0)
    grid = np.zeros(keys.max(axis=0) - origin + 1, dtype=bool)
    grid[tuple((keys - origin).T)] = True
    lab, _ = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=int))
    pid = lab[tuple((keys - origin).T)]
    sizes = np.bincount(pid)
    sizes[0] = 0
    return pts[pid == int(np.argmax(sizes))]


def robust_bbox(pts):
    """0.5-99.5 percentile core's min/max. (lift.robust_bbox.)"""
    lo = np.percentile(pts, 0.5, axis=0)
    hi = np.percentile(pts, 99.5, axis=0)
    keep = np.all((pts >= lo) & (pts <= hi), axis=1)
    core = pts[keep] if keep.sum() >= 20 else pts
    return core.min(axis=0), core.max(axis=0), int(keep.sum())


class CloudGate:
    """The job cloud in its solidification role: room envelope +
    nearest-surface queries, built once per run."""

    def __init__(self, cloud, cfg: DiscoveryConfig):
        self.cloud = cloud
        self.room_lo = np.percentile(cloud, cfg.room_pct_lo, axis=0)
        self.room_hi = np.percentile(cloud, cfg.room_pct_hi, axis=0)
        self.tree = cKDTree(cloud)

    def count_inside_bbox(self, lo, hi, pad_frac=0.15):
        pad = pad_frac * (hi - lo)
        return int(np.all((self.cloud >= lo - pad)
                          & (self.cloud <= hi + pad), axis=1).sum())
