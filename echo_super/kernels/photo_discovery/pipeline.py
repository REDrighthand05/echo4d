"""photo_discovery pipeline — the round-3 final form as one function.

    run_photo_discovery(job_dir, catalog, cfg,
                        detections=None, masks=None) -> result

Stages (experiments/discovery_photo, REPORT.md/2/3):
  1. OWLv2 open-vocabulary detection over the job's frames with the
     catalogue's queries, boxes converged per class per frame
     (detect.py; skipped when `detections` are injected);
  2. SAM best mask per converged box (segment.py; skipped when
     `masks` are injected);
  3. per-mask lift: depth main-cluster crop, unprojection through the
     job's own depth + poses into the exported frame (lift.py);
  4. per class: room/surface gates + 2-frame corroboration against
     the job cloud, then the round-3 treatment plan — monitor keeps
     rank-1 masks only, monitor+chair deliver only the largest
     connected component of the corroborated set (instance
     isolation), everything else is delivered as corroborated.

`masks` injection forms (the cache the tests use):
  (index_records, {mask_file: bool ndarray})   — sam_segment's return
  {"index": [...], "arrays": {...}}
  {"index": [...], "dir": path}                — PNG cache on disk

Returns {"objects": [...], "point_sets": {label: (N,3) float64},
"room": {...}}. Each object record is JSON-shaped:
  label, zh, typical_m, pitfalls,
  detect_frames, n_detect_frames, top_score, masks_used, treatment,
  points_raw_lifted, points_in_room, points_on_surface,
  points_corroborated_2frames, points_pre_isolation,
  points (delivered, solidified), bbox_min, bbox_max, extent,
  cloud_points_inside_bbox
Points/bboxes are in the job's exported scene units (the job's scale
state applies; on the pioneer job the scale is undefined and sizes
are only meaningful relative to the room, exactly as in REPORT3).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .config import DiscoveryConfig
from .detect import owlv2_detect
from .lift import (CloudGate, depth_crop, fit_depth_alignment,
                   gates_and_corroborate, largest_component, lift_pixels,
                   read_ply_points, robust_bbox, z_map_from_rel)
from .segment import sam_segment


def _resolve_masks(masks):
    """Normalize an injected mask cache to (index, load_fn) where
    load_fn(mask_file) -> bool ndarray."""
    if isinstance(masks, tuple) and len(masks) == 2:
        index, arrays = masks
        return index, lambda fn: np.asarray(arrays[fn]) > 0
    if isinstance(masks, dict) and "index" in masks:
        index = masks["index"]
        if "arrays" in masks:
            arrays = masks["arrays"]
            return index, lambda fn: np.asarray(arrays[fn]) > 0
        if "dir" in masks:
            d = Path(masks["dir"])
            return index, lambda fn: np.asarray(Image.open(d / fn)) > 0
    raise ValueError(
        "masks must be (index, arrays) or {'index': [...], 'arrays': "
        "{...}} or {'index': [...], 'dir': path}")


def _frame_stems(job_dir: Path, poses: dict):
    depth_dir = job_dir / "depth"
    stems = []
    for p in sorted(depth_dir.glob("*.npy")):
        if f"{p.stem}.jpg" in poses:
            stems.append(p.stem)
    return stems


def run_photo_discovery(job_dir, catalog, cfg: DiscoveryConfig = None,
                        detections=None, masks=None, log=print):
    cfg = cfg or DiscoveryConfig()
    job_dir = Path(job_dir)
    poses = json.loads((job_dir / "poses.json").read_text())
    r_align = np.array(
        json.loads((job_dir / "alignment.json").read_text())["rotation"])
    cloud = read_ply_points(job_dir / "scene_points.ply")
    gate = CloudGate(cloud, cfg)
    fits, a_glob, b_glob = fit_depth_alignment(job_dir, job_dir / "depth")
    stems = _frame_stems(job_dir, poses)
    log(f"photo_discovery: {len(stems)} frames with depth+pose, "
        f"cloud={len(cloud)} pts")

    if detections is None:
        detections = owlv2_detect(job_dir / "frames", stems, catalog,
                                  cfg, log=log)
    if masks is None:
        masks = sam_segment(job_dir / "frames", detections, cfg, log=log)
    index, load_mask = _resolve_masks(masks)

    # cluster rank inside (frame, class): descending score — the same
    # recovery lift3_cache used (converge emits clusters in that order
    # and segment picks per class in that order).
    groups = {}
    for m in index:
        groups.setdefault((m["frame"], m["class"]), []).append(m)
    for ms in groups.values():
        for rank, m in enumerate(
                sorted(ms, key=lambda d: (-d["score"], d["mask_file"])), 1):
            m["cluster_rank"] = rank

    frames_of, top_score = {}, {}
    for m in index:
        if m.get("mask_file"):
            frames_of.setdefault(m["class"], set()).add(m["frame"])
            top_score[m["class"]] = max(
                top_score.get(m["class"], 0.0), m["score"])

    # ---- per-mask lift (index order preserved per class) ----
    zcache = {}
    chunks = {}   # label -> [(stem, points, rank)]
    for m in index:
        stem, cls = m["frame"], m["class"]
        if not m.get("mask_file"):
            continue
        mask = load_mask(m["mask_file"])
        if m["mask_px"] > cfg.degenerate_frac * mask.size:
            log(f"{stem} {cls}: degenerate mask skipped")
            continue
        if stem not in zcache:
            rel = np.load(job_dir / "depth" / f"{stem}.npy")
            a, b = fits.get(stem, (a_glob, b_glob))
            zcache[stem] = z_map_from_rel(rel, a, b)
        z, ok = zcache[stem]
        sel = mask & ok
        ys, xs = np.nonzero(sel)
        if len(ys) == 0:
            continue
        keep, _stats = depth_crop(z[ys, xs], cfg)
        ys_c, xs_c = ys[keep], xs[keep]
        take = np.arange(0, len(ys_c), cfg.stride)
        world = lift_pixels(ys_c[take], xs_c[take], z,
                            poses[f"{stem}.jpg"], r_align)
        chunks.setdefault(cls, []).append(
            (stem, world, m.get("cluster_rank", 1)))

    # ---- per-class solidify + treatment + assembly ----
    objects = []
    point_sets = {}
    for cls in sorted(catalog):
        cat = catalog[cls]
        treatment = cfg.treatment_for(cls)
        used = [(s, p) for (s, p, r) in chunks.get(cls, [])
                if treatment["keep_rank"] is None
                or r == treatment["keep_rank"]]
        rec = {
            "label": cls, "zh": cat["zh"], "typical_m": cat["typical_m"],
            "pitfalls": cat["pitfalls"],
            "detect_frames": sorted(frames_of.get(cls, set())),
            "top_score": round(top_score.get(cls, 0.0), 3),
            "masks_used": len(used),
            "treatment": dict(treatment),
        }
        rec["n_detect_frames"] = len(rec["detect_frames"])
        if used:
            stats, strong = gates_and_corroborate(
                used, cloud, gate.tree, gate.room_lo, gate.room_hi, cfg)
        else:
            stats = {"points_raw_lifted": 0, "points_in_room": 0,
                     "points_on_surface": 0,
                     "points_corroborated_2frames": 0, "points_dedup": 0}
            strong = np.empty((0, 3))
        rec.update(stats)
        rec["points_pre_isolation"] = stats["points_dedup"]
        delivered = (largest_component(strong, cfg.isolation_voxel)
                     if treatment["isolate"] else strong)
        point_sets[cls] = delivered
        rec["points"] = int(len(delivered))
        if len(delivered):
            lo, hi, _ = robust_bbox(delivered)
            rec.update({
                "bbox_min": [round(float(v), 3) for v in lo],
                "bbox_max": [round(float(v), 3) for v in hi],
                "extent": [round(float(v), 3) for v in (hi - lo)],
                "cloud_points_inside_bbox":
                    gate.count_inside_bbox(lo, hi),
            })
        else:
            rec.update({"bbox_min": None, "bbox_max": None,
                        "extent": None, "cloud_points_inside_bbox": 0})
        objects.append(rec)
        log(f"{cls}: detect_frames={rec['n_detect_frames']} "
            f"points={rec['points']} extent={rec['extent']}")

    room = {
        "room_extent": [round(float(v), 3)
                        for v in (gate.room_hi - gate.room_lo)],
        "room_min": [round(float(v), 3) for v in gate.room_lo],
        "room_max": [round(float(v), 3) for v in gate.room_hi],
    }
    return {"objects": objects, "point_sets": point_sets, "room": room}
