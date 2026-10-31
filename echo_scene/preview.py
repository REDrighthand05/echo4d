#!/usr/bin/env python3
"""ECHO process live preview: replay how a scene was reconstructed.

Reads a finished job directory and writes one standalone HTML page that
animates the reconstruction process itself, Detroit-style (blue grid
space -- floor AND walls filled with grid squares, not an empty box):

  1. the extracted frames appear as small photo cards floating at their
     camera positions;
  2. the camera route draws itself stroke by stroke as a glowing blue
     line through the camera centres;
  3. frame by frame, the frame's photo lands on the surface it faces and
     is CRUSHED in place: it shatters into constant-size particles that
     burst outward, then get pulled as if by gravity onto exactly their
     true 3D positions and settle into the growing scene;
  4. all points accumulate into the final scene and the mesh fades in.

The camera choreographs itself (v2): a slow ambient drift, with push-in
close-ups on a rotating subset of dissolving frames -- the viewer never
has to orbit by hand (a manual grab still overrides for a few seconds).

Everything shown is the job's real data (poses, photos, fused points
with per-frame provenance) -- a presentation layer, not a fake animation.

Data contract (v1): the per-frame point provenance comes from a
points_frame_ids.npy sitting next to a PRE-alignment points.bin (same
point order as the job's aligned points.bin). Pass --provenance DIR when
that pair lives in a sibling job dir (e.g. an experiment re-run of
stage_fuse that kept the ids). When no provenance exists, points are
assigned approximately to the nearest in-frustum camera -- the grouping
is then indicative, not exact.

Usage:
  python3 -m echo_scene.preview <job_dir> <out.html> [--provenance DIR]
  python3 -m echo_scene.preview <job_dir> <out.html> [--provenance DIR] --promo

--promo re-times the same real data for a 17.22 s promo cut, authored
natively for that length and pinned to the promo track's accents: a
curated set of hero dissolves keeps the exact photo/crush/particle
speeds of the full replay (nothing is sped up), the remaining frames'
points rain in as a quick montage.

The page honours URL hashes for stills/debugging: #t=0.45&still=1 seeks
to a timeline fraction, pauses and hides the control bar.
"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# timeline (seconds on the master clock; the page is driven by these numbers)
# ---------------------------------------------------------------------------
DUR = 72.0
CARD_T0, CARD_T1 = 1.0, 9.0        # photo cards appear, staggered
ROUTE_T0, ROUTE_T1 = 7.0, 17.0     # route draws itself
DISS_T0, DISS_T1 = 15.0, 60.0      # per-frame photo -> points dissolve
PHOTO_IN, PHOTO_HOLD, PHOTO_OUT = 0.10, 0.10, 0.28   # per-frame slot phases
POINT_FLY = 0.75                   # per-frame crush -> settle duration
MESH_T0, MESH_T1 = 61.0, 68.0      # mesh fades in
DIM_T0, DIM_T1 = 60.0, 68.0        # cards dim to ghost at the end

# -- promo mode: a 17.22 s cut timed to the promo track (100 BPM; the
# music starts at 0.76 s, so the first beat of picture is near-black and
# the cards start as the music kicks in). Only the SCHEDULE changes;
# PHOTO_* and POINT_FLY (the readable speeds of the effects) stay
# exactly as above -- extra time is filled with more hero moments and
# longer holds, never with faster particles.
PROMO_DUR = 17.22
PROMO_CARD = (0.88, 2.7)
PROMO_ROUTE = (2.5, 5.4)
PROMO_MESH = (15.7, 16.8)
PROMO_DIM = (15.7, 16.8)
# hero dissolve slots, pinned to the track's strongest accents (file
# time): ranks (as fractions of the hero list) -> slot seconds. The two
# strongest accents are 4.32 and 10.02; 7.62 / 13.62 / 14.81 are the
# next anchors. A hero's crush begins 0.10 s after its slot (PHOTO_HOLD),
# i.e. essentially on the beat. Slots between anchors interpolate.
PROMO_HERO_ANCHORS = [(0.0, 4.22), (4 / 13, 7.52), (7 / 13, 9.92),
                      (11 / 13, 13.52), (1.0, 14.71)]
PROMO_MONT_T0, PROMO_MONT_T1 = 4.0, 15.4   # montage rain window
PROMO_MONT_FLY = 0.45
PROMO_N_HERO = 14
# close-ups sit on the accent heroes (rank fractions of the hero list)
PROMO_CLOSEUP_FRACS = (0.0, 4 / 13, 7 / 13, 11 / 13)

ATLAS_TILE_W, ATLAS_TILE_H = 126, 224   # portrait 1080x1920 ratio
ATLAS_COLS = 16


def _read_points_bin(path: Path):
    """points.bin layout (see echo_scene.fuse): N x (float32 xyz, uint8 rgb)."""
    raw = np.fromfile(path, dtype=np.uint8)
    raw = raw.reshape(-1, 15)
    xyz = raw[:, 0:12].copy().view("<f4").reshape(-1, 3)
    rgb = raw[:, 12:15].copy()
    return xyz.astype(np.float64), rgb


def _quat_rot(q_xyzw):
    x, y, z, w = q_xyzw
    n = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return n


def _floor_level(y: np.ndarray) -> float:
    """Robust floor height: densest y-bin in the lower half of the cloud."""
    lo, hi = float(np.percentile(y, 0.2)), float(np.percentile(y, 55))
    hist, edges = np.histogram(y, bins=160, range=(lo, hi))
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))


def _decimate_mesh(verts, vcols, tris, max_tris=90000):
    """Vertex-cluster decimation for the web payload (display only)."""
    lo = verts.min(axis=0)
    extent = float((verts.max(axis=0) - lo).max())
    cell = extent / 110.0
    for _ in range(24):
        keys = np.floor((verts - lo) / cell).astype(np.int64)
        key1d = (keys[:, 0] * 73856093) ^ (keys[:, 1] * 19349663) ^ (keys[:, 2] * 83492791)
        _, inverse, counts = np.unique(key1d, return_inverse=True, return_counts=True)
        nv = len(counts)
        new_pos = np.zeros((nv, 3))
        for c in range(3):
            new_pos[:, c] = np.bincount(inverse, weights=verts[:, c], minlength=nv) / counts
        new_col = np.zeros((nv, 3))
        for c in range(3):
            new_col[:, c] = np.bincount(inverse, weights=vcols[:, c].astype(np.float64),
                                        minlength=nv) / counts
        new_tris = inverse[tris]
        ok = (new_tris[:, 0] != new_tris[:, 1]) & (new_tris[:, 1] != new_tris[:, 2]) \
            & (new_tris[:, 0] != new_tris[:, 2])
        new_tris = new_tris[ok]
        if len(new_tris) <= max_tris or cell > extent / 18.0:
            return (new_pos.astype(np.float32), new_col.astype(np.uint8),
                    new_tris.astype(np.uint32), len(tris))
        cell *= 1.3
    return (new_pos.astype(np.float32), new_col.astype(np.uint8),
            new_tris.astype(np.uint32), len(tris))


def _hero_slot(rank_frac: float) -> float:
    """Hero slot time for a rank fraction, interpolated between the
    accent anchors in PROMO_HERO_ANCHORS."""
    a = PROMO_HERO_ANCHORS
    if rank_frac <= a[0][0]:
        return a[0][1]
    for (f0, t0), (f1, t1) in zip(a, a[1:]):
        if rank_frac <= f1:
            return t0 + (t1 - t0) * (rank_frac - f0) / max(f1 - f0, 1e-9)
    return a[-1][1]


def _promo_timing(frame_geo, frame_ids, used_ids, burst):
    """Promo schedule: ~14 hero frames get the full photo->crush
    treatment at the normal effect speeds, their slots pinned to the
    track's accent anchors; every other frame's points rain in as a
    quick montage (shorter flight, damped burst), in walking order.
    Returns (frame_meta, pt_t0, pt_t1); burst is damped in place for
    the montage points."""
    n = len(frame_geo)
    heroes = []
    for s in range(PROMO_N_HERO):
        lo = int(n * s / PROMO_N_HERO)
        hi = max(lo + 1, int(n * (s + 1) / PROMO_N_HERO))
        heroes.append(max(range(lo, hi), key=lambda k: frame_geo[k]["n_points"]))
    heroes = sorted(set(heroes))
    hero_set = set(heroes)
    pt_t0 = np.zeros(len(frame_ids), dtype=np.float32)
    pt_t1 = np.zeros(len(frame_ids), dtype=np.float32)
    frame_meta = []
    for rank, j in enumerate(heroes):
        slot = _hero_slot(rank / max(len(heroes) - 1, 1))
        m = dict(frame_geo[j])
        m.update({
            "slot": slot,
            "photo_in_end": slot + PHOTO_IN,
            "dissolve_begin": slot + PHOTO_HOLD,
            "photo_out_end": slot + PHOTO_OUT,
            "points_t0": slot + PHOTO_HOLD,
            "points_t1": slot + PHOTO_HOLD + POINT_FLY,
        })
        frame_meta.append(m)
        sel = frame_ids == used_ids[j]
        pt_t0[sel] = m["points_t0"]
        pt_t1[sel] = m["points_t1"]
    mont = [j for j in range(n) if j not in hero_set]
    for rank, j in enumerate(mont):
        t0 = PROMO_MONT_T0 + (PROMO_MONT_T1 - PROMO_MONT_T0) \
            * rank / max(len(mont) - 1, 1)
        sel = frame_ids == used_ids[j]
        pt_t0[sel] = t0
        pt_t1[sel] = t0 + PROMO_MONT_FLY
        burst[sel] *= 0.35
    return frame_meta, pt_t0, pt_t1


def _promo_closeups(frame_meta):
    """Up to 4 push-in close-ups on the accent hero frames (windows:
    in 0.85 s before the photo lands, out 0.75 s after its points
    settle)."""
    if not frame_meta:
        return []
    idxs = sorted({int(round(f * (len(frame_meta) - 1)))
                   for f in PROMO_CLOSEUP_FRACS})
    out = []
    for i in idxs:
        m = frame_meta[i]
        C = np.array(m["centre"]); V = np.array(m["z"]); X = np.array(m["x"])
        d = m["dmed"]
        plane_c = C + V * d
        half_w = d * max(m["cx"], m["W"] - m["cx"]) / m["f"]
        half_h = d * max(m["cy"], m["H"] - m["cy"]) / m["f"]
        k = float(np.clip(2.05 * max(half_w, half_h), 0.7, 0.92 * d))
        eye = (plane_c - V * k + X * (half_w * 0.38)
               + np.array([0.0, 1.0, 0.0]) * (half_h * 0.22))
        out.append({
            "stem": m["stem"],
            "t0": m["slot"] - 0.85,
            "t2": m["points_t1"] + 0.75,
            "slot": m["slot"],
            "target": [float(v) for v in plane_c],
            "eye": [float(v) for v in eye],
        })
    return out


def build_preview_data(job_dir: Path, provenance_dir: Path | None,
                       promo: bool = False):
    """Collect every array the page needs. Returns (manifest, blob, atlas_jpeg)."""
    job_dir = Path(job_dir)
    # master-timeline values; promo mode re-times the same real data
    # for the 17.22 s cut (the speeds of the individual effects are
    # unchanged)
    tl_dur, tl_card = DUR, (CARD_T0, CARD_T1)
    tl_route = (ROUTE_T0, ROUTE_T1)
    tl_mesh, tl_dim = (MESH_T0, MESH_T1), (DIM_T0, DIM_T1)
    if promo:
        tl_dur = PROMO_DUR
        tl_card, tl_route = PROMO_CARD, PROMO_ROUTE
        tl_mesh, tl_dim = PROMO_MESH, PROMO_DIM
    prov_dir = Path(provenance_dir) if provenance_dir else job_dir
    poses = json.loads((job_dir / "poses.json").read_text())
    frames_dir = job_dir / "frames"
    depth_dir = job_dir / "depth"

    align_path = job_dir / "alignment.json"
    rot_align = None
    if align_path.exists():
        rot_align = np.array(json.loads(align_path.read_text())["rotation"],
                             dtype=np.float64)

    def to_export(v):
        """Row-vector transform COLMAP world -> exported (upright) frame."""
        return v @ rot_align.T if rot_align is not None else v

    # -- cameras -------------------------------------------------------------
    stems_sorted = sorted(Path(k).stem for k in poses)
    cams = {}
    for stem in stems_sorted:
        p = poses[stem + ".jpg"]
        R = _quat_rot(p["cam_from_world_quat_xyzw"])     # cam_from_world
        t = np.asarray(p["cam_from_world_t"], dtype=np.float64)
        centre = -R.T @ t
        cam = p["camera"]
        f, cx, cy = (float(cam["params"][0]), float(cam["params"][1]),
                     float(cam["params"][2]))
        cams[stem] = {
            "stem": stem,
            "centre": to_export(centre),
            # camera axes in world are the rows of cam_from_world
            "x": to_export(R[0]), "y": to_export(R[1]), "z": to_export(R[2]),
            "f": f, "cx": cx, "cy": cy,
            "W": int(cam["width"]), "H": int(cam["height"]),
        }

    # -- fused points with per-frame provenance ------------------------------
    final_pts, final_rgb = _read_points_bin(job_dir / "points.bin")
    frame_ids = None
    prov_note = ""
    ids_path = prov_dir / "points_frame_ids.npy"
    if ids_path.exists() and (prov_dir / "points.bin").exists():
        raw_pts, raw_rgb = _read_points_bin(prov_dir / "points.bin")
        frame_ids = np.load(ids_path).astype(np.int64)
        rgb = raw_rgb
        if len(raw_pts) == len(final_pts):
            # Two provenance conventions exist: ids stored beside a
            # pre-alignment cloud (a sibling provenance directory; the
            # rotation in alignment.json carries that cloud onto the
            # final points), and ids written by stage_fuse into the job
            # itself, which already index the final aligned cloud.
            # Whichever reading matches the job's points wins.
            diff_raw = float(np.abs(raw_pts - final_pts).max())
            diff_rot = (float(np.abs(to_export(raw_pts) - final_pts).max())
                        if rot_align is not None else float("inf"))
            if diff_raw <= diff_rot:
                pts = raw_pts
                prov_note = (f"provenance exact: source cloud already in "
                             f"the exported frame (max diff {diff_raw:.2e})")
            else:
                pts = to_export(raw_pts)
                prov_note = (f"provenance exact: rotated source cloud "
                             f"matches the job's aligned points.bin "
                             f"(max diff {diff_rot:.2e})")
        else:
            prov_note = ("WARNING: provenance cloud size differs from the "
                         "job's points.bin; using the provenance cloud")
            pts = to_export(raw_pts)
            final_pts, final_rgb = pts, rgb
    else:
        # Approximate fallback: nearest in-frustum camera wins the point.
        prov_note = ("no points_frame_ids.npy found -- points assigned "
                     "approximately to the nearest in-frustum camera")
        pts, rgb = final_pts, final_rgb
        frame_ids = _assign_points_approx(pts, cams, stems_sorted)

    # dissolve frames = stems that own >=1 point AND have a depth file,
    # in the same order stage_fuse assigned frame_index (sorted stems).
    index_path = job_dir / "frame_index.json"
    if index_path.exists():
        # stage_fuse's own id table (written since v0.2): ids cover only
        # frames that contributed points, so re-deriving the table from
        # the poses/depth files would misalign every later frame the
        # moment one posed frame owns no points.
        id_to_stem = {int(k): v for k, v in
                      json.loads(index_path.read_text()).items()}
    else:
        cand_stems = [s for s in stems_sorted
                      if (depth_dir / f"{s}.npy").exists()]
        id_to_stem = {k: s for k, s in enumerate(cand_stems)}
    used_ids = sorted(set(int(i) for i in np.unique(frame_ids)) & set(id_to_stem))
    diss_stems = [id_to_stem[k] for k in used_ids]
    n_diss = len(diss_stems)

    # -- per-frame geometry: photo-plane start positions ----------------------
    n_pts = len(pts)
    start_pos = np.empty_like(pts)
    burst = np.zeros((n_pts, 3), dtype=np.float64)
    extent_pts = float((pts.max(axis=0) - pts.min(axis=0)).max())
    rng = np.random.default_rng(20261031)
    frame_geo = []
    for j, stem in enumerate(diss_stems):
        fid_here = used_ids[j]
        sel = frame_ids == fid_here
        P = pts[sel]
        C = cams[stem]
        centre, view = C["centre"], C["z"]
        rel = P - centre
        dist = np.linalg.norm(rel, axis=1)
        # photo plane sits a touch inside the median surface so the
        # projected photo tucks into the room instead of towering out of
        # it; start positions use the same plane, so points still emerge
        # from exactly their own photo pixels
        d_med = float(np.median(dist)) * 0.92
        denom = rel @ view
        denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        start_pos[sel] = centre + rel * (d_med / denom)[:, None]
        # crush-burst vector per point: mostly off the photo toward the
        # camera, plus a radial spread from the photo's centre, plus
        # noise; the vector's length is the burst amplitude. In the page
        # this decays to zero exactly as the point reaches its target,
        # so every particle lands on its real position.
        plane_c = centre + view * d_med
        radial = start_pos[sel] - plane_c
        rn = np.linalg.norm(radial, axis=1, keepdims=True)
        radial_n = np.where(rn > 1e-9, radial / np.maximum(rn, 1e-9), 0.0)
        n_here = int(sel.sum())
        rnd = rng.standard_normal((n_here, 3))
        rnd /= np.maximum(np.linalg.norm(rnd, axis=1, keepdims=True), 1e-9)
        bdir = -view[None, :] * 0.9 + radial_n * 0.55 + rnd * 0.35
        bdir /= np.maximum(np.linalg.norm(bdir, axis=1, keepdims=True), 1e-9)
        amp = extent_pts * (0.02 + 0.05 * rng.random(n_here))
        burst[sel] = bdir * amp[:, None]
        frame_geo.append({
            "stem": stem,
            "centre": [float(v) for v in centre],
            "x": [float(v) for v in C["x"]], "y": [float(v) for v in C["y"]],
            "z": [float(v) for v in C["z"]],
            "f": C["f"], "cx": C["cx"], "cy": C["cy"], "W": C["W"], "H": C["H"],
            "dmed": d_med,
            "n_points": int(sel.sum()),
        })

    # per-frame schedule -> frame_meta (photo events) + per-point timing
    if promo:
        frame_meta, pt_t0, pt_t1 = _promo_timing(frame_geo, frame_ids,
                                                 used_ids, burst)
    else:
        frame_meta = []
        slot_span = (DISS_T1 - DISS_T0) / max(n_diss, 1)
        for j, g in enumerate(frame_geo):
            slot = DISS_T0 + slot_span * j
            m = dict(g)
            m.update({
                "slot": slot,
                "photo_in_end": slot + PHOTO_IN,
                "dissolve_begin": slot + PHOTO_HOLD,
                "photo_out_end": slot + PHOTO_OUT,
                "points_t0": slot + PHOTO_HOLD,
                "points_t1": slot + PHOTO_HOLD + POINT_FLY,
            })
            frame_meta.append(m)
        pt_t0 = np.zeros(n_pts, dtype=np.float32)
        pt_t1 = np.zeros(n_pts, dtype=np.float32)
        for j, stem in enumerate(diss_stems):
            sel = frame_ids == used_ids[j]
            pt_t0[sel] = frame_meta[j]["points_t0"]
            pt_t1[sel] = frame_meta[j]["points_t1"]
    # -- cinematic close-ups: a rotating subset of dissolve frames -----------
    # The dissolve runs ~5 frames/second, far too fast to push in on every
    # frame, so ~9 point-rich frames spread across the act get a close-up:
    # the camera eases in before the photo lands, holds through the crush,
    # and eases back out after the points have settled.
    closeups = []
    if frame_meta:
        med_n = float(np.median([m["n_points"] for m in frame_meta]))
        chosen_slots = []
        for frac in (0.05, 0.16, 0.27, 0.38, 0.49, 0.60, 0.71, 0.82, 0.93):
            t_target = DISS_T0 + (DISS_T1 - DISS_T0) * frac
            best = None
            for m in frame_meta:
                if abs(m["slot"] - t_target) > 2.2 or m["n_points"] < 0.55 * med_n:
                    continue
                if any(abs(m["slot"] - s) < 4.2 for s in chosen_slots):
                    continue
                if best is None or m["n_points"] > best["n_points"]:
                    best = m
            if best is None:
                continue
            chosen_slots.append(best["slot"])
            C = np.array(best["centre"]); V = np.array(best["z"])
            X = np.array(best["x"]); d = best["dmed"]
            plane_c = C + V * d
            half_w = d * max(best["cx"], best["W"] - best["cx"]) / best["f"]
            half_h = d * max(best["cy"], best["H"] - best["cy"]) / best["f"]
            k = float(np.clip(2.05 * max(half_w, half_h), 0.7, 0.92 * d))
            eye = (plane_c - V * k + X * (half_w * 0.38)
                   + np.array([0.0, 1.0, 0.0]) * (half_h * 0.22))
            closeups.append({
                "stem": best["stem"],
                "t0": best["slot"] - 1.5,
                "t2": best["points_t1"] + 2.0,
                "slot": best["slot"],
                "target": [float(v) for v in plane_c],
                "eye": [float(v) for v in eye],
            })

    if promo:
        closeups = _promo_closeups(frame_meta)

    # -- cards: every posed frame --------------------------------------------
    card_stems = stems_sorted
    card_atlas_idx = {s: i for i, s in enumerate(card_stems)}
    for m in frame_meta:
        m["atlas"] = card_atlas_idx[m["stem"]]
    card_centre = np.array([cams[s]["centre"] for s in card_stems])
    card_x = np.array([cams[s]["x"] for s in card_stems])
    card_y = np.array([cams[s]["y"] for s in card_stems])
    n_cards = len(card_stems)
    card_t = (tl_card[0] + (tl_card[1] - tl_card[0])
              * np.arange(n_cards) / max(n_cards - 1, 1)).astype(np.float32)

    # -- route: camera centres in walking order --------------------------------
    route_pos = card_centre.astype(np.float64)
    seg = np.linalg.norm(np.diff(route_pos, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    route_frac = (cum / max(cum[-1], 1e-9)).astype(np.float32)

    # -- thumbnail atlas --------------------------------------------------------
    rows = int(np.ceil(n_cards / ATLAS_COLS))
    atlas = Image.new("RGB", (ATLAS_COLS * ATLAS_TILE_W, rows * ATLAS_TILE_H),
                      (8, 12, 20))
    for i, stem in enumerate(card_stems):
        img = Image.open(frames_dir / f"{stem}.jpg").convert("RGB")
        img = img.resize((ATLAS_TILE_W, ATLAS_TILE_H), Image.LANCZOS)
        atlas.paste(img, ((i % ATLAS_COLS) * ATLAS_TILE_W,
                          (i // ATLAS_COLS) * ATLAS_TILE_H))
    buf = io.BytesIO()
    atlas.save(buf, "JPEG", quality=80)
    atlas_jpeg = buf.getvalue()

    # -- mesh (optional fade-in) -------------------------------------------------
    mesh_blob_parts = []
    mesh_counts = [0, 0]
    mv_path, mi_path = job_dir / "mesh_vertices.bin", job_dir / "mesh_indices.bin"
    if mv_path.exists() and mi_path.exists():
        mv, mc = _read_points_bin(mv_path)
        tris = np.fromfile(mi_path, dtype="<u4").reshape(-1, 3)
        dv, dc, dt, full_tris = _decimate_mesh(mv, mc, tris)
        mesh_blob_parts = [dv.tobytes(), dc.tobytes(), dt.tobytes()]
        mesh_counts = [len(dv), len(dt)]
        mesh_note = f"mesh decimated {full_tris} -> {len(dt)} tris for the page"
    else:
        mesh_note = "no mesh bins in job dir; mesh fade-in skipped"

    # -- pack the binary blob -----------------------------------------------------
    parts = [
        pts.astype(np.float32).tobytes(),          # end positions
        start_pos.astype(np.float32).tobytes(),    # photo-plane start positions
        rgb.astype(np.uint8).tobytes(),            # colours
        pt_t0.tobytes(), pt_t1.tobytes(),          # per-point timing
        burst.astype(np.float32).tobytes(),        # per-point crush-burst vector
        card_centre.astype(np.float32).tobytes(),
        card_x.astype(np.float32).tobytes(),
        card_y.astype(np.float32).tobytes(),
        card_t.tobytes(),
        route_pos.astype(np.float32).tobytes(),
        route_frac.tobytes(),
    ] + mesh_blob_parts
    # pad every part to a 4-byte boundary: the page reads the blob back as
    # Float32Array/Uint32Array views, which require aligned offsets
    blob = b"".join(p + b"\x00" * (-len(p) % 4) for p in parts)

    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    manifest = {
        "duration": tl_dur,
        "n_points": int(n_pts),
        "n_frames": n_diss,
        "n_cards": int(n_cards),
        "mesh_verts": int(mesh_counts[0]),
        "mesh_tris": int(mesh_counts[1]),
        "bbox_min": [float(v) for v in lo],
        "bbox_max": [float(v) for v in hi],
        "floor_y": _floor_level(pts[:, 1]),
        "route_t0": tl_route[0], "route_t1": tl_route[1],
        "mesh_t0": tl_mesh[0], "mesh_t1": tl_mesh[1],
        "dim_t0": tl_dim[0], "dim_t1": tl_dim[1],
        "atlas": {"cols": ATLAS_COLS, "rows": rows,
                  "tile_w": ATLAS_TILE_W, "tile_h": ATLAS_TILE_H},
        "frames": frame_meta,
        "closeups": closeups,
        "cards": [{"stem": s, "atlas": card_atlas_idx[s]} for s in card_stems],
        "provenance": prov_note,
        "mesh_note": mesh_note,
    }
    return manifest, blob, atlas_jpeg


def _assign_points_approx(pts, cams, stems_sorted):
    """Fallback grouping: nearest camera whose frustum contains the point."""
    n = len(pts)
    best = np.full(n, -1, dtype=np.int64)
    best_d = np.full(n, np.inf)
    order = {s: k for k, s in enumerate(stems_sorted)}
    for stem in stems_sorted:
        C = cams[stem]
        # world -> cam: rows of R are the axes, so cam coords = (p-C).axes
        rel = pts - C["centre"]
        xc = rel @ C["x"]; yc = rel @ C["y"]; zc = rel @ C["z"]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = C["f"] * xc / np.where(zc > 0, zc, np.inf) + C["cx"]
            v = C["f"] * yc / np.where(zc > 0, zc, np.inf) + C["cy"]
        inside = (zc > 0) & (u >= 0) & (u < C["W"]) & (v >= 0) & (v < C["H"])
        d = np.linalg.norm(rel, axis=1)
        upd = inside & (d < best_d)
        best[upd] = order[stem]
        best_d[upd] = d[upd]
    best[best < 0] = 0
    return best


# ---------------------------------------------------------------------------
# page template
# ---------------------------------------------------------------------------
_PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>ECHO 重建过程回放</title>
<style>
  html, body { margin: 0; height: 100%; background: #04070d; overflow: hidden; }
  #gl { display: block; width: 100%; height: 100%; touch-action: none; cursor: grab; }
  #gl:active { cursor: grabbing; }
  #cap {
    position: fixed; left: 14px; top: 12px; color: #bcd7ff;
    font: 14px/1.5 ui-monospace, Menlo, Consolas, monospace;
    text-shadow: 0 1px 4px #000; pointer-events: none; white-space: pre-line;
  }
  #cap b { color: #6fb2ff; font-size: 15px; }
  #bar {
    position: fixed; left: 50%; bottom: 12px; transform: translateX(-50%);
    display: flex; align-items: center; gap: 10px;
    max-width: calc(100vw - 16px); box-sizing: border-box;
    background: rgba(6, 12, 22, 0.82); border: 1px solid rgba(111, 178, 255, 0.35);
    border-radius: 12px; padding: 8px 12px;
    font: 13px/1.4 ui-monospace, Menlo, Consolas, monospace; color: #cfe4ff;
    user-select: none; -webkit-user-select: none; touch-action: manipulation;
  }
  #bar button {
    font: inherit; color: #dceeff; background: rgba(111, 178, 255, 0.12);
    border: 1px solid rgba(111, 178, 255, 0.45); border-radius: 8px;
    padding: 6px 12px; cursor: pointer; min-height: 32px; white-space: nowrap;
  }
  #bar button:active { background: rgba(111, 178, 255, 0.3); }
  #segs {
    display: flex; gap: 2px; align-items: center; cursor: pointer;
    width: min(46vw, 520px); padding: 8px 2px; touch-action: none;
  }
  .seg {
    flex: 1 1 0; height: 12px; background: rgba(140, 175, 225, 0.30);
    transform: scaleY(0.42); pointer-events: none;
  }
  .seg.on { background: #e9f3ff; transform: none; }
  .seg.head { background: #6fb2ff; transform: none; }
  #clock { min-width: 86px; text-align: center; color: #8fc2ff; }
  #hud {
    position: fixed; right: 14px; top: 12px; text-align: right;
    color: rgba(168, 204, 255, 0.60); pointer-events: none;
    font: 11px/1.9 ui-monospace, Menlo, Consolas, monospace;
    letter-spacing: 0.08em; white-space: pre-line;
    text-shadow: 0 1px 4px #000;
  }
  #remain {
    position: fixed; left: 50%; top: 12px; transform: translateX(-50%);
    text-align: center; pointer-events: none;
    font: 13px/1.5 ui-monospace, Menlo, Consolas, monospace;
    color: rgba(205, 228, 255, 0.85); text-shadow: 0 1px 4px #000;
  }
  #remainLine { width: 132px; height: 2px; background: rgba(160, 195, 240, 0.25); margin: 0 auto 3px; }
  #remainFill { height: 100%; width: 0%; background: #e9f3ff; }
</style>
</head>
<body>
<canvas id="gl"></canvas>
<div id="cap">加载中…</div>
<div id="hud"></div>
<div id="remain"><div id="remainLine"><div id="remainFill"></div></div><span id="remainTxt"></span></div>
<div id="bar">
  <button id="btnPlay">⏸ 暂停</button>
  <div id="segs"></div>
  <span id="clock">0.0s</span>
</div>
<script>
'use strict';
const MANIFEST = /*__MANIFEST__*/{};
const DATA_B64 = "/*__DATA_B64__*/";
const ATLAS_B64 = "/*__ATLAS_B64__*/";
const cap = document.getElementById('cap');
const hud = document.getElementById('hud');
const remainTxt = document.getElementById('remainTxt');
const remainFill = document.getElementById('remainFill');
const bar = document.getElementById('bar');
const canvas = document.getElementById('gl');
const gl = canvas.getContext('webgl2', { antialias: true });
if (!gl) { cap.textContent = '此设备不支持 WebGL2，无法播放回放。'; throw new Error('no webgl2'); }

function b64ToBytes(s) {
  const bin = atob(s); const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
const raw = b64ToBytes(DATA_B64);
let off = 0;
function takeF32(n) { const a = new Float32Array(raw.buffer, raw.byteOffset + off, n); off += 4 * n; return a; }
function takeU8(n) { const a = new Uint8Array(raw.buffer, raw.byteOffset + off, n); off += n; off = (off + 3) & ~3; return a; }
function takeU32(n) { const a = new Uint32Array(raw.buffer, raw.byteOffset + off, n); off += 4 * n; return a; }
const NP = MANIFEST.n_points, NC = MANIFEST.n_cards;
const endPos = takeF32(3 * NP), startPos = takeF32(3 * NP), ptCol = takeU8(3 * NP);
const ptT0 = takeF32(NP), ptT1 = takeF32(NP), ptBurst = takeF32(3 * NP);
const cardC = takeF32(3 * NC), cardX = takeF32(3 * NC), cardY = takeF32(3 * NC), cardT = takeF32(NC);
const routePos = takeF32(3 * NC), routeFrac = takeF32(NC);
const NV = MANIFEST.mesh_verts, NT = MANIFEST.mesh_tris;
const meshPos = NV ? takeF32(3 * NV) : null, meshCol = NV ? takeU8(3 * NV) : null;
const meshIdx = NT ? takeU32(3 * NT) : null;
const FRAMES = MANIFEST.frames, DUR = MANIFEST.duration;
const CLOSEUPS = MANIFEST.closeups || [];

/* ---------------- mat4 (column-major) ---------------- */
function mMul(a, b) { const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k]; o[c * 4 + r] = s; } return o; }
function mPersp(fovy, asp, n, f) { const t = 1 / Math.tan(fovy / 2), o = new Float32Array(16);
  o[0] = t / asp; o[5] = t; o[10] = (f + n) / (n - f); o[11] = -1; o[14] = 2 * f * n / (n - f); return o; }
function mLook(eye, ctr, up) { let zx = eye[0] - ctr[0], zy = eye[1] - ctr[1], zz = eye[2] - ctr[2];
  let l = Math.hypot(zx, zy, zz); zx /= l; zy /= l; zz /= l;
  let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
  l = Math.hypot(xx, xy, xz); xx /= l; xy /= l; xz /= l;
  const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
  const o = new Float32Array(16);
  o[0] = xx; o[1] = yx; o[2] = zx; o[4] = xy; o[5] = yy; o[6] = zy; o[8] = xz; o[9] = yz; o[10] = zz;
  o[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  o[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  o[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]); o[15] = 1; return o; }

/* ---------------- shaders ---------------- */
function prog(vs, fs) {
  function sh(t, s) { const h = gl.createShader(t); gl.shaderSource(h, s); gl.compileShader(h);
    if (!gl.getShaderParameter(h, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(h) + '\n' + s); return h; }
  const p = gl.createProgram(); gl.attachShader(p, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p); if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p)); return p;
}
const VS_HEAD = '#version 300 es\nprecision highp float;\nuniform mat4 uMVP;\n';
const gridP = prog(VS_HEAD + `
  layout(location=0) in vec3 aPos; layout(location=1) in float aShade;
  out float vS;
  void main(){ vS = aShade; gl_Position = uMVP * vec4(aPos, 1.0); }`,
  `#version 300 es
  precision highp float; uniform float uAlpha; in float vS; out vec4 o;
  void main(){ o = vec4(vec3(0.24, 0.5, 1.0) * (0.35 + 0.65 * vS), uAlpha * (0.25 + 0.75 * vS)); }`);
const cardP = prog(VS_HEAD + `
  layout(location=0) in vec3 aPos; layout(location=1) in vec2 aUV; layout(location=2) in vec2 aLocal;
  layout(location=3) in float aT; layout(location=4) in float aIdx;
  uniform float uTime, uActive, uDim;
  out vec2 vUV; out vec2 vLoc; out float vA; out float vHot;
  void main(){
    float ap = smoothstep(aT, aT + 0.6, uTime);
    vA = ap * (1.0 - 0.85 * uDim);
    vHot = (abs(aIdx - uActive) < 0.5) ? 1.0 : 0.0;
    vUV = aUV; vLoc = aLocal;
    gl_Position = uMVP * vec4(aPos, 1.0); }`,
  `#version 300 es
  precision highp float; uniform sampler2D uTex;
  in vec2 vUV; in vec2 vLoc; in float vA; in float vHot; out vec4 o;
  void main(){
    vec3 c = texture(uTex, vUV).rgb;
    float lum0 = dot(c, vec3(0.299, 0.587, 0.114));
    c = mix(c, vec3(lum0) * vec3(0.80, 0.94, 1.22), 0.30);
    float e = min(min(vLoc.x, 1.0 - vLoc.x), min(vLoc.y, 1.0 - vLoc.y));
    float border = 1.0 - smoothstep(0.015, 0.05, e);
    c = mix(c, vec3(0.35, 0.65, 1.0), border * (0.55 + 0.45 * vHot));
    c *= 0.82 + 0.5 * vHot;
    o = vec4(c, vA * (0.92 + 0.08 * vHot)); if (o.a < 0.004) discard; }`);
const routeP = prog(VS_HEAD + `
  layout(location=0) in vec3 aPos; layout(location=1) in float aS;
  uniform float uRoute; out float vA;
  void main(){ vA = (aS <= uRoute) ? 1.0 : 0.0; gl_Position = uMVP * vec4(aPos, 1.0); }`,
  `#version 300 es
  precision highp float; uniform float uFade; in float vA; out vec4 o;
  void main(){ if (vA <= 0.0) discard; o = vec4(0.35, 0.66, 1.0, 0.9 * uFade); }`);
const nodeP = prog(VS_HEAD + `
  layout(location=0) in vec3 aPos; layout(location=1) in float aS;
  uniform float uRoute, uScale; out float vA; out float vHead;
  void main(){
    vA = (aS <= uRoute) ? 1.0 : 0.0;
    vHead = exp(-abs(aS - uRoute) * 55.0);
    vec4 mv = uMVP * vec4(aPos, 1.0);
    gl_PointSize = clamp((5.0 + 26.0 * vHead) * uScale * (10.0 / -mv.z), 1.0, 42.0);
    gl_Position = mv; }`,
  `#version 300 es
  precision highp float; uniform float uFade; in float vA; in float vHead; out vec4 o;
  void main(){ if (vA <= 0.0) discard;
    float d = length(gl_PointCoord - 0.5);
    float a = smoothstep(0.5, 0.06, d);
    vec3 c = mix(vec3(0.4, 0.7, 1.0), vec3(0.85, 0.95, 1.0), vHead);
    o = vec4(c, a * (0.5 + 0.5 * vHead) * uFade); }`);
const ptsP = prog(VS_HEAD + `
  layout(location=0) in vec3 aEnd; layout(location=1) in vec3 aStart; layout(location=2) in vec3 aCol;
  layout(location=3) in float aT0; layout(location=4) in float aT1; layout(location=5) in vec3 aBurst;
  uniform float uTime, uScale; out vec3 vC; out float vA;
  void main(){
    float p = clamp((uTime - aT0) / max(aT1 - aT0, 1e-4), 0.0, 1.0);
    if (uTime < aT0) { gl_Position = vec4(0.0, 0.0, -10.0, 1.0); gl_PointSize = 0.0; vC = aCol; vA = 0.0; return; }
    /* crush motion: the burst offset kicks in fast and decays to exactly
       zero, while the pull toward the target accelerates (p*p) like
       gravity -- so every particle lands exactly on its real position */
    float trav = p * p;
    vec3 pos = mix(aStart, aEnd, trav) + aBurst * sin(3.14159265 * pow(p, 0.7));
    vec4 mv = uMVP * vec4(pos, 1.0);
    gl_PointSize = clamp(3.4 * uScale * (10.0 / -mv.z), 1.5, 9.0);
    vC = aCol; vA = clamp(p / 0.15, 0.0, 1.0);
    gl_Position = mv; }`,
  `#version 300 es
  precision highp float; in vec3 vC; in float vA; out vec4 o;
  void main(){ float d = length(gl_PointCoord - 0.5);
    float a = smoothstep(0.5, 0.3, d) * vA; if (a < 0.01) discard;
    o = vec4(vC, a); }`);
const photoP = prog(VS_HEAD + `
  layout(location=0) in vec3 aPos; layout(location=1) in vec2 aUV;
  out vec2 vUV; out vec2 vLoc;
  void main(){ vUV = aUV; vLoc = aUV; gl_Position = uMVP * vec4(aPos, 1.0); }`,
  `#version 300 es
  precision highp float; uniform sampler2D uTex; uniform float uOpacity; uniform vec4 uRect;
  in vec2 vUV; in vec2 vLoc; out vec4 o;
  void main(){
    vec2 uv = vec2(mix(uRect.x, uRect.z, vUV.x), mix(uRect.y, uRect.w, vUV.y));
    vec3 c = texture(uTex, uv).rgb;
    float lum0 = dot(c, vec3(0.299, 0.587, 0.114));
    c = mix(c, vec3(lum0) * vec3(0.80, 0.94, 1.22), 0.30);
    float e = min(min(vLoc.x, 1.0 - vLoc.x), min(vLoc.y, 1.0 - vLoc.y));
    float border = 1.0 - smoothstep(0.004, 0.02, e);
    c = mix(c, vec3(0.45, 0.72, 1.0), border * 0.8);
    o = vec4(c, uOpacity); if (o.a < 0.004) discard; }`);
const meshP = prog(VS_HEAD + `
  layout(location=0) in vec3 aPos; layout(location=1) in vec3 aCol;
  out vec3 vC; out vec3 vW;
  void main(){ vC = aCol; vW = aPos; gl_Position = uMVP * vec4(aPos, 1.0); }`,
  `#version 300 es
  precision highp float; uniform float uOpacity; in vec3 vC; in vec3 vW; out vec4 o;
  void main(){
    vec3 n = normalize(cross(dFdx(vW), dFdy(vW)));
    float li = 0.45 + 0.55 * abs(dot(n, normalize(vec3(0.4, 0.8, 0.45))));
    o = vec4(vC * li, uOpacity); if (o.a < 0.004) discard; }`);

/* ---------------- buffers ---------------- */
function vao(build) { const v = gl.createVertexArray(); gl.bindVertexArray(v); build(); gl.bindVertexArray(null); return v; }
function attrBuf(loc, data, size, type, norm) {
  const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, type || gl.FLOAT, !!norm, 0, 0);
  return b;
}
function idxBuf(data) { const b = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, b);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data, gl.STATIC_DRAW); return b; }
function uni(p, name) { return gl.getUniformLocation(p, name); }
function tileUV(idx, lx, ly) {
  const A = MANIFEST.atlas;
  const col = idx % A.cols, row = Math.floor(idx / A.cols);
  const aw = A.cols * A.tile_w, ah = A.rows * A.tile_h;
  const u0 = (col * A.tile_w + 1.5) / aw, u1 = ((col + 1) * A.tile_w - 1.5) / aw;
  const vTop = 1 - (row * A.tile_h + 1.5) / ah, vBot = 1 - ((row + 1) * A.tile_h - 1.5) / ah;
  return [u0 + (u1 - u0) * lx, vTop + (vBot - vTop) * ly];
}

/* grid: floor grid + grid-filled walls */
const bmin = MANIFEST.bbox_min, bmax = MANIFEST.bbox_max, floorY = MANIFEST.floor_y;
let gridVAO, gridCount = 0;
{
  const cx = (bmin[0] + bmax[0]) / 2, cz = (bmin[2] + bmax[2]) / 2;
  const hx = (bmax[0] - bmin[0]) / 2 + 1.2, hz = (bmax[2] - bmin[2]) / 2 + 1.2;
  const topY = floorY + (bmax[1] - floorY);
  const pos = [], shade = [];
  function line(a, b, s) { pos.push(a[0], a[1], a[2], b[0], b[1], b[2]); shade.push(s, s); }
  const step = 0.75;
  for (let x = cx - hx; x <= cx + hx + 1e-4; x += step)
    line([x, floorY, cz - hz], [x, floorY, cz + hz], Math.abs(x - cx) < step / 2 ? 0.9 : 0.35);
  for (let z = cz - hz; z <= cz + hz + 1e-4; z += step)
    line([cx - hx, floorY, z], [cx + hx, floorY, z], Math.abs(z - cz) < step / 2 ? 0.9 : 0.35);
  /* walls: every vertical plane is FILLED with grid squares (Detroit
     android-vision style) -- verticals + horizontals at the same step
     as the floor, dimmer than the floor, bright base/top/corner edges */
  const cs = [[cx - hx, cz - hz], [cx + hx, cz - hz], [cx + hx, cz + hz], [cx - hx, cz + hz]];
  for (let i = 0; i < 4; i++) {
    const a = cs[i], b = cs[(i + 1) % 4];
    line([a[0], floorY, a[1]], [b[0], floorY, b[1]], 1.0);
    line([a[0], topY, a[1]], [b[0], topY, b[1]], 0.85);
    line([a[0], floorY, a[1]], [a[0], topY, a[1]], 0.85);
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    for (let s = step; s < len - 1e-4; s += step) {
      const u = s / len, x = a[0] + (b[0] - a[0]) * u, z = a[1] + (b[1] - a[1]) * u;
      line([x, floorY, z], [x, topY, z], 0.30);
    }
    for (let y = floorY + step; y < topY - 1e-4; y += step)
      line([a[0], y, a[1]], [b[0], y, b[1]], 0.22);
  }
  /* scattered blue line shards strewn through the whole volume --
     ambient dressing like in the android-vision shots: thin streaks of
     random length and orientation, deterministic (seeded), biased
     toward the periphery, suspended in the air (not data) */
  {
    let seed = 987654321;
    const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
    const extL = Math.max(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]);
    for (let i = 0; i < 150; i++) {
      const push = 0.45 + 0.75 * rnd();
      const px = cx + (rnd() * 2 - 1) * hx * push;
      const pz = cz + (rnd() * 2 - 1) * hz * push;
      const py = floorY + rnd() * (topY - floorY);
      let dx = rnd() * 2 - 1, dy = (rnd() * 2 - 1) * 0.55, dz = rnd() * 2 - 1;
      const dl = Math.hypot(dx, dy, dz) || 1; dx /= dl; dy /= dl; dz /= dl;
      const L = extL * (0.04 + 0.20 * Math.pow(rnd(), 1.5));
      const s = 0.25 + 0.75 * rnd() * rnd();
      line([px - dx * L / 2, py - dy * L / 2, pz - dz * L / 2],
           [px + dx * L / 2, py + dy * L / 2, pz + dz * L / 2], s);
    }
  }
  gridCount = pos.length / 3;
  const P = new Float32Array(pos), S = new Float32Array(shade);
  gridVAO = vao(() => { attrBuf(0, P, 3); attrBuf(1, S, 1); });
}

/* photo cards */
let cardVAO, cardCount = 0;
{
  const hw = 0.34, hh = hw / (1080 / 1920);
  const pos = [], uv = [], loc = [], tt = [], ix = [], idx = [];
  for (let i = 0; i < NC; i++) {
    const C = [cardC[3 * i], cardC[3 * i + 1], cardC[3 * i + 2]];
    const X = [cardX[3 * i], cardX[3 * i + 1], cardX[3 * i + 2]];
    const Y = [cardY[3 * i], cardY[3 * i + 1], cardY[3 * i + 2]];
    const cs = [[-1, -1, 0, 0], [1, -1, 1, 0], [-1, 1, 0, 1], [1, 1, 1, 1]];
    for (const [sx, sy, lx, ly] of cs) {
      pos.push(C[0] + X[0] * hw * sx + Y[0] * hh * sy,
               C[1] + X[1] * hw * sx + Y[1] * hh * sy,
               C[2] + X[2] * hw * sx + Y[2] * hh * sy);
      const [u, v] = tileUV(i, lx, ly); uv.push(u, v); loc.push(lx, ly);
      tt.push(cardT[i]); ix.push(i);
    }
    const b = 4 * i; idx.push(b, b + 2, b + 1, b + 1, b + 2, b + 3);
  }
  cardCount = idx.length;
  cardVAO = vao(() => {
    attrBuf(0, new Float32Array(pos), 3); attrBuf(1, new Float32Array(uv), 2);
    attrBuf(2, new Float32Array(loc), 2); attrBuf(3, new Float32Array(tt), 1);
    attrBuf(4, new Float32Array(ix), 1); idxBuf(new Uint32Array(idx));
  });
}

/* route line + nodes */
let routeVAO, routeSegVerts = 0, nodeVAO;
{
  const pos = [], s = [];
  for (let i = 0; i < NC - 1; i++) {
    pos.push(routePos[3 * i], routePos[3 * i + 1], routePos[3 * i + 2],
             routePos[3 * i + 3], routePos[3 * i + 4], routePos[3 * i + 5]);
    s.push(routeFrac[i], routeFrac[i + 1]);
  }
  routeSegVerts = pos.length / 3;
  routeVAO = vao(() => { attrBuf(0, new Float32Array(pos), 3); attrBuf(1, new Float32Array(s), 1); });
  nodeVAO = vao(() => { attrBuf(0, routePos, 3); attrBuf(1, routeFrac, 1); });
}

/* scene points */
const ptsVAO = vao(() => {
  attrBuf(0, endPos, 3); attrBuf(1, startPos, 3);
  attrBuf(2, ptCol, 3, gl.UNSIGNED_BYTE, true);
  attrBuf(3, ptT0, 1); attrBuf(4, ptT1, 1); attrBuf(5, ptBurst, 3);
});

/* photo quad (single, dynamic) */
const photoVAO = gl.createVertexArray();
const photoPosBuf = gl.createBuffer(), photoUVBuf = gl.createBuffer();
{
  gl.bindVertexArray(photoVAO);
  gl.bindBuffer(gl.ARRAY_BUFFER, photoPosBuf);
  gl.bufferData(gl.ARRAY_BUFFER, 48, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, photoUVBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 0, 0);
  idxBuf(new Uint16Array([0, 2, 1, 1, 2, 3]));
  gl.bindVertexArray(null);
}
function updatePhotoQuad(f) {
  const C = f.centre, X = f.x, Y = f.y, Z = f.z, d = f.dmed;
  const corner = (u, v) => {
    const lx = (u - f.cx) / f.f * d, ly = (v - f.cy) / f.f * d;
    return [C[0] + Z[0] * d + X[0] * lx + Y[0] * ly,
            C[1] + Z[1] * d + X[1] * lx + Y[1] * ly,
            C[2] + Z[2] * d + X[2] * lx + Y[2] * ly];
  };
  const pts = [corner(0, 0), corner(f.W, 0), corner(0, f.H), corner(f.W, f.H)];
  const arr = new Float32Array(12);
  pts.forEach((p, i) => { arr[3 * i] = p[0]; arr[3 * i + 1] = p[1]; arr[3 * i + 2] = p[2]; });
  gl.bindBuffer(gl.ARRAY_BUFFER, photoPosBuf);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, arr);
}

/* mesh */
let meshVAO = null;
if (NV > 0) {
  meshVAO = vao(() => {
    attrBuf(0, meshPos, 3); attrBuf(1, meshCol, 3, gl.UNSIGNED_BYTE, true);
    idxBuf(meshIdx);
  });
}

/* atlas texture */
const atlasTex = gl.createTexture();
{
  const img = new Image();
  img.onload = () => {
    gl.bindTexture(gl.TEXTURE_2D, atlasTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    atlasReady = true;
  };
  img.src = 'data:image/jpeg;base64,' + ATLAS_B64;
}
let atlasReady = false;

/* ---------------- camera ---------------- */
const centre = [(bmin[0] + bmax[0]) / 2, floorY + 0.34 * (bmax[1] - floorY), (bmin[2] + bmax[2]) / 2];
const extent = Math.max(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]);
let az = 0.75, el = 0.36, radius = extent * 1.28, target = centre.slice();
let azPhase = 0.75; const radius0 = extent * 1.30;
let lastInteract = -10;
function bindOrbit() {
  let drag = null;
  canvas.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, b: e.button }; canvas.setPointerCapture(e.pointerId); lastInteract = perfT; });
  canvas.addEventListener('pointermove', e => {
    if (!drag) return; lastInteract = perfT;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
    if (drag.b === 2 || e.shiftKey) {
      const s = radius * 0.0016, ce = Math.cos(el), se = Math.sin(el), ca = Math.cos(az), sa = Math.sin(az);
      const right = [ca, 0, -sa], upv = [-se * sa, ce, -se * ca];
      for (let k = 0; k < 3; k++) target[k] += (-dx * right[k] + dy * upv[k]) * s;
    } else {
      az -= dx * 0.005; el = Math.max(-1.4, Math.min(1.5, el + dy * 0.005));
    }
  });
  const end = () => { drag = null; };
  canvas.addEventListener('pointerup', end); canvas.addEventListener('pointercancel', end);
  canvas.addEventListener('wheel', e => { e.preventDefault(); lastInteract = perfT;
    radius = Math.max(extent * 0.15, Math.min(extent * 4, radius * (e.deltaY > 0 ? 1.12 : 0.9))); }, { passive: false });
  canvas.addEventListener('contextmenu', e => e.preventDefault());
}

/* ---------------- timeline ---------------- */
const hash = new URLSearchParams(location.hash.slice(1));
let time = 0, playing = true, speed = 1, perfT = 0;
if (hash.has('t')) time = Math.max(0, Math.min(1, parseFloat(hash.get('t')))) * DUR;
if (hash.has('speed')) speed = parseFloat(hash.get('speed')) || 1;
const still = hash.get('still') === '1';
if (still) { playing = false; document.getElementById('btnPlay').style.display = 'none'; }
const btnPlay = document.getElementById('btnPlay'), clock = document.getElementById('clock');
function setPlayUI() { btnPlay.textContent = playing ? '⏸ 暂停' : '▶ 播放'; }
btnPlay.addEventListener('click', () => { if (!playing && time >= DUR - 0.01) time = 0; playing = !playing; setPlayUI(); });
setPlayUI();

/* segmented tick timeline (Detroit-style): played = solid blocks,
   unplayed = dim ticks; pointer drag seeks */
const NSEG = 56, segsEl = document.getElementById('segs'), segDivs = [];
for (let i = 0; i < NSEG; i++) { const d = document.createElement('div'); d.className = 'seg'; segsEl.appendChild(d); segDivs.push(d); }
let segIdx = -1;
function updateSegs() {
  const idx = Math.min(NSEG, Math.floor(time / DUR * NSEG));
  if (idx === segIdx) return;
  for (let i = 0; i < NSEG; i++)
    segDivs[i].className = 'seg' + (i < idx ? ' on' : i === idx ? ' head' : '');
  segIdx = idx;
}
function seekEvt(e) {
  const r = segsEl.getBoundingClientRect();
  time = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * DUR;
}
segsEl.addEventListener('pointerdown', e => { segsEl.setPointerCapture(e.pointerId); seekEvt(e); });
segsEl.addEventListener('pointermove', e => { if (e.buttons & 1) seekEvt(e); });

/* floating HUD: real job data only (frame numbers, point counts,
   stage states) -- the job carries no video fps/timecodes, so none
   are shown */
function fmtClock(s) {
  s = Math.max(0, Math.ceil(s));
  return String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0');
}
const settleEv = FRAMES.map(f => [f.points_t1, f.n_points]).sort((a, b) => a[0] - b[0]);
function settledAt(t) { let s = 0; for (const [tt, n] of settleEv) { if (tt <= t) s += n; else break; } return s; }
let hudKey = -1;
function updateHUD(act) {
  const key = Math.floor(time * 4);
  if (key === hudKey) return; hudKey = key;
  const dissT0 = FRAMES.length ? FRAMES[0].slot : 15;
  const st = time < MANIFEST.route_t0 ? 0 : time < dissT0 ? 1 :
             time < MANIFEST.mesh_t0 ? 2 : time < MANIFEST.mesh_t1 ? 3 : 4;
  const mark = i => (st === 4 || i < st) ? '✓' : i === st ? '▸' : '·';
  const lines = [
    'PROCESS DIRECTORY',
    mark(0) + ' 切帧 EXTRACT · ' + NC + ' 帧',
    mark(1) + ' 路线 ROUTE · ' + NC + ' 机位',
    mark(2) + ' 碎点 DISSOLVE · ' + MANIFEST.n_frames + ' 帧',
    mark(3) + ' 表面 MESH · ' + NT + ' 面',
    '',
    '已落点 ' + settledAt(time).toLocaleString('en-US') + ' / ' + NP.toLocaleString('en-US'),
  ];
  if (act) lines.push('FRAME ' + act.stem.replace('frame_', '') + ' · +' + act.n_points.toLocaleString('en-US') + ' 点');
  hud.textContent = lines.join('\n');
  remainTxt.textContent = '-' + fmtClock(DUR - time);
  remainFill.style.width = (time / DUR * 100).toFixed(1) + '%';
}

function caption(t) {
  if (t < MANIFEST.route_t0) return '正在把视频切成一帧帧画面…';
  if (t < (FRAMES.length ? FRAMES[0].slot : 15)) return '正在找回拍摄的路线…';
  if (t < MANIFEST.mesh_t0) return '正在把照片贴到它对着的墙上，再碎成点…';
  if (t < MANIFEST.mesh_t1) return '正在合上表面…';
  return '完成 ✓';
}
function smooth(a, b, x) { const p = Math.max(0, Math.min(1, (x - a) / (b - a))); return p * p * (3 - 2 * p); }
function activeFrame(t) {
  let act = null;
  for (const f of FRAMES) { if (f.slot <= t && t <= f.photo_out_end) act = f; else if (f.slot > t) break; }
  return act;
}
function photoOpacity(f, t) {
  const tt = t - f.slot;
  if (tt < 0 || t > f.photo_out_end) return 0;
  if (tt < 0.10) return tt / 0.10;
  const db = f.dissolve_begin - f.slot;
  if (tt <= db) return 1;
  return Math.max(0, 1 - (tt - db) / Math.max(f.photo_out_end - f.dissolve_begin, 1e-4));
}
let photoFrame = null;

/* ---------------- render loop ---------------- */
function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.floor(canvas.clientWidth * dpr), h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  gl.viewport(0, 0, w, h);
}
bindOrbit();
let lastNow = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.05, (now - lastNow) / 1000); lastNow = now; perfT = now / 1000;
  if (playing) { time += dt * speed; if (time >= DUR) { time = DUR; playing = false; setPlayUI(); } }
  resize();
  /* cinematic camera: if the user grabbed the view in the last 6s use
     their orbit; otherwise a slow ambient drift. Close-up windows then
     blend the camera toward the dissolving frame's wall and back out. */
  const manual = perfT - lastInteract < 6;
  let baseEye, baseTgt;
  if (manual) {
    baseTgt = target;
    baseEye = [target[0] + radius * Math.cos(el) * Math.sin(az),
               target[1] + radius * Math.sin(el),
               target[2] + radius * Math.cos(el) * Math.cos(az)];
    azPhase = az - time * 0.028;
  } else {
    const azA = azPhase + time * 0.028;
    const elA = 0.34 + 0.09 * Math.sin(time * 0.13);
    const rA = radius0 * (1.0 + 0.05 * Math.sin(time * 0.07));
    baseTgt = centre;
    baseEye = [centre[0] + rA * Math.cos(elA) * Math.sin(azA),
               centre[1] + rA * Math.sin(elA),
               centre[2] + rA * Math.cos(elA) * Math.cos(azA)];
  }
  let eye = baseEye, lookTgt = baseTgt;
  if (!manual) {
    for (const c of CLOSEUPS) {
      if (time >= c.t0 && time <= c.t2) {
        const w = Math.min(smooth(c.t0, c.t0 + 1.1, time),
                           1 - smooth(c.t2 - 1.1, c.t2, time));
        if (w > 0.001) {
          eye = [0, 1, 2].map(k => baseEye[k] + (c.eye[k] - baseEye[k]) * w);
          lookTgt = [0, 1, 2].map(k => baseTgt[k] + (c.target[k] - baseTgt[k]) * w);
        }
        break;
      }
    }
  }
  const mvp = mMul(mPersp(0.96, canvas.width / canvas.height, 0.1, 300), mLook(eye, lookTgt, [0, 1, 0]));
  const uScale = canvas.height / 850;
  const uRoute = Math.max(0, Math.min(1, (time - MANIFEST.route_t0) / (MANIFEST.route_t1 - MANIFEST.route_t0)));
  const uMesh = smooth(MANIFEST.mesh_t0, MANIFEST.mesh_t1, time);
  const uDim = smooth(MANIFEST.dim_t0, MANIFEST.dim_t1, time);
  const act = activeFrame(time);
  gl.clearColor(0.012, 0.022, 0.042, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST); gl.disable(gl.CULL_FACE);

  /* grid */
  gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.depthMask(false);
  gl.useProgram(gridP); gl.uniformMatrix4fv(uni(gridP, 'uMVP'), false, mvp);
  gl.uniform1f(uni(gridP, 'uAlpha'), 0.30 * smooth(0, 2, time));
  gl.bindVertexArray(gridVAO); gl.drawArrays(gl.LINES, 0, gridCount);

  /* mesh (fades in at the end) */
  if (meshVAO && uMesh > 0.003) {
    gl.depthMask(true);
    gl.useProgram(meshP); gl.uniformMatrix4fv(uni(meshP, 'uMVP'), false, mvp);
    gl.uniform1f(uni(meshP, 'uOpacity'), uMesh * 0.92);
    gl.bindVertexArray(meshVAO); gl.drawElements(gl.TRIANGLES, NT * 3, gl.UNSIGNED_INT, 0);
    gl.depthMask(false);
  }

  /* scene points */
  gl.useProgram(ptsP); gl.uniformMatrix4fv(uni(ptsP, 'uMVP'), false, mvp);
  gl.uniform1f(uni(ptsP, 'uTime'), time); gl.uniform1f(uni(ptsP, 'uScale'), uScale);
  gl.bindVertexArray(ptsVAO); gl.drawArrays(gl.POINTS, 0, NP);

  /* photo on the wall */
  if (act && atlasReady) {
    const op = photoOpacity(act, time);
    if (op > 0.003) {
      if (photoFrame !== act) { updatePhotoQuad(act); photoFrame = act; }
      const A = MANIFEST.atlas, col = act.atlas % A.cols, row = Math.floor(act.atlas / A.cols);
      gl.useProgram(photoP); gl.uniformMatrix4fv(uni(photoP, 'uMVP'), false, mvp);
      gl.uniform1f(uni(photoP, 'uOpacity'), op);
      gl.uniform4f(uni(photoP, 'uRect'), col / A.cols, 1 - row / A.rows, (col + 1) / A.cols, 1 - (row + 1) / A.rows);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, atlasTex);
      gl.uniform1i(uni(photoP, 'uTex'), 0);
      gl.bindVertexArray(photoVAO); gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
    }
  }

  /* frame cards */
  if (atlasReady) {
    gl.useProgram(cardP); gl.uniformMatrix4fv(uni(cardP, 'uMVP'), false, mvp);
    gl.uniform1f(uni(cardP, 'uTime'), time);
    gl.uniform1f(uni(cardP, 'uActive'), act ? act.atlas : -1);
    gl.uniform1f(uni(cardP, 'uDim'), uDim);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, atlasTex);
    gl.uniform1i(uni(cardP, 'uTex'), 0);
    gl.bindVertexArray(cardVAO); gl.drawElements(gl.TRIANGLES, cardCount, gl.UNSIGNED_INT, 0);
  }

  /* route: additive glow */
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
  gl.useProgram(routeP); gl.uniformMatrix4fv(uni(routeP, 'uMVP'), false, mvp);
  gl.uniform1f(uni(routeP, 'uRoute'), uRoute);
  gl.uniform1f(uni(routeP, 'uFade'), 1.0 - 0.8 * uDim);
  gl.bindVertexArray(routeVAO); gl.drawArrays(gl.LINES, 0, routeSegVerts);
  gl.useProgram(nodeP); gl.uniformMatrix4fv(uni(nodeP, 'uMVP'), false, mvp);
  gl.uniform1f(uni(nodeP, 'uRoute'), uRoute); gl.uniform1f(uni(nodeP, 'uScale'), uScale);
  gl.uniform1f(uni(nodeP, 'uFade'), 1.0 - 0.8 * uDim);
  gl.bindVertexArray(nodeVAO); gl.drawArrays(gl.POINTS, 0, NC);
  gl.depthMask(true); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  cap.innerHTML = '<b>ECHO · 重建过程回放</b>\n' + caption(time);
  clock.textContent = time.toFixed(1) + 's / ' + DUR.toFixed(0) + 's';
  updateSegs();
  updateHUD(act);
}
requestAnimationFrame(frame);
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def write_preview(job_dir, out_path, provenance_dir=None, promo=False) -> dict:
    manifest, blob, atlas_jpeg = build_preview_data(
        Path(job_dir), Path(provenance_dir) if provenance_dir else None,
        promo=promo)
    html = (_PAGE
            .replace("/*__MANIFEST__*/{}", json.dumps(manifest, ensure_ascii=False))
            .replace("/*__DATA_B64__*/", base64.b64encode(blob).decode())
            .replace("/*__ATLAS_B64__*/", base64.b64encode(atlas_jpeg).decode()))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return {
        "ok": True,
        "out": str(out_path),
        "bytes": out_path.stat().st_size,
        "points": manifest["n_points"],
        "frames": manifest["n_frames"],
        "cards": manifest["n_cards"],
        "mesh_tris": manifest["mesh_tris"],
        "provenance": manifest["provenance"],
        "mesh_note": manifest["mesh_note"],
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    prov = None
    if "--provenance" in argv:
        k = argv.index("--provenance")
        prov = argv[k + 1]
        del argv[k:k + 2]
    promo = "--promo" in argv
    if promo:
        argv.remove("--promo")
    if len(argv) != 2:
        print(__doc__)
        return 2
    info = write_preview(argv[0], argv[1], prov, promo=promo)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
