"""True-scale stage （真尺度）: anchor the scene to real metres.

Method (proven by Pioneer A on the room job, 2026-10-31; productised
here unchanged in spirit):

  * DISCOVER candidate rectangles in the job's frames (LSD straight
    segments -> line clustering -> rectangle assembly; deterministic,
    no learned detector, no voting).
  * MEASURE each candidate per frame with that frame's own depth,
    aligned to the COLMAP frame by the pipeline's own anchor fit
    (fuse._collect_anchor_pairs + _robust_affine_fit -- the same
    machinery stage_fuse uses). Corners are back-projected at their
    true pixels; depth is sampled a few px inward because monocular
    depth smears across silhouette edges.
  * VERIFY, per candidate: planarity, agreement between the scale
    implied by its height and by its width under a catalogue gang
    hypothesis, an estimate_scale cross-check along its centreline,
    and support from a wall plane fitted to the surrounding ring.
  * CORROBORATE across frames: candidates cluster by 3D position;
    a cluster needs >= 2 frames and a tight spread. Clusters whose
    implied scales agree form a group; the group must additionally
    pass a sanity gate -- the scale it implies for the room itself
    (ceiling height, plan extents, route length, all measured from
    the job's own data) must be plausible. Only then is the scale
    adopted, as the pooled median of the group's measurements.

The anchor catalogue below is DATA: a new anchor class is a new entry,
not new code. v1 ships the 国标 86-type panel family only. (A door
leaf is deliberately absent: Pioneer A never measured one with this
automatic chain -- the room footage's door frames were dropped by
frame selection -- so claiming it would be unproven.)

Evidence label: a scale found this way is an ESTIMATE （估计） and is
recorded as such, with its measurement spread. When no anchor group
survives, the job's scale stays UNKNOWN （未知） and the scene is
exactly what the previous stages produced -- this stage never blocks
a job and never invents a scale.

When a scale is adopted it is applied to the job's data in place
(points.bin, scene_points.ply, points_trim_dropped.bin,
sparse_points.ply, and the camera translations in poses.json), so
every downstream stage (mesh, voxel, present, preview, replay) and
every export inherits metres with no code of its own. Application is
idempotent: scale.json records that it happened, and a re-run will
not scale twice. If the markers stage already put the scene into
metres (markers.json scale_to_meters -- a MEASURED scale, stronger
evidence), this stage records that and changes nothing.
"""
import json
from pathlib import Path

import numpy as np

__all__ = ["ANCHOR_CATALOG", "stage_scale"]

# ---------------------------------------------------------------------------
# Anchor catalogue (data, not code). kind "gang_panel": a rectangular
# plate whose height is one unit and whose width is n units, n = 1..max_gang.
# ---------------------------------------------------------------------------
ANCHOR_CATALOG = [
    {
        "id": "cn86_panel",
        "label": "墙面开关面板",
        "kind": "gang_panel",
        "unit_m": 0.086,
        "max_gang": 4,
        "agree_tol": 0.12,
        "description": (
            "国标 86 型开关/插座面板：单联 86x86 mm，连体总宽为联数 x 86 mm"
        ),
    },
]

# Discovery / measurement constants (Pioneer A settings, frozen for v1).
_MIN_SEGMENT = 46.0        # px, shortest LSD segment used
_LINE_SUPPORT_MIN = 46.0   # px of segment support per clustered line
_AGREE_TOL_DEFAULT = 0.12  # width-vs-height scale agreement (catalogue may override)
_LINE_XCHECK_TOL = 0.20    # centreline estimate_scale cross-check tolerance
_CLUSTER_RADIUS = 0.32     # scene units, same physical anchor
_CLUSTER_SPREAD_MAX = 0.20  # IQR / median inside one cluster
_GROUP_AGREE = 0.10        # cluster medians within 10% form one scale group
_MIN_ERROR_PCT = 10        # the method's established error floor (Pioneer A)
_MAX_ERROR_PCT = 30

# Sanity gate: what the adopted scale must imply for the room itself.
_CEILING_RANGE_M = (2.0, 3.5)
_PLAN_RANGE_M = (1.6, 10.0)
_ROUTE_RANGE_M = (4.0, 80.0)


# ---------------------------------------------------------------------------
# Rectangle discovery: LSD segments -> clustered lines -> rectangles.
# ---------------------------------------------------------------------------
def _fit_line(pts, wts):
    cen = np.average(pts, axis=0, weights=wts)
    d = pts - cen
    cov = (d * wts[:, None]).T @ d
    _, vecs = np.linalg.eigh(cov)
    direction = vecs[:, -1]
    return cen, direction / np.linalg.norm(direction)


def _cluster_lines(segs):
    lines = []
    for p1, p2, length in sorted(segs, key=lambda s: -s[2]):
        mid = (p1 + p2) / 2
        placed = False
        for ln in lines:
            d = ln["dir"]
            v = (p2 - p1)
            v = v / max(np.linalg.norm(v), 1e-9)
            if abs(float(v @ d)) < np.cos(np.radians(3.5)):
                continue
            off = abs(float((mid - ln["point"]) @ np.array([-d[1], d[0]])))
            if off > 7:
                continue
            ln["members"].append((p1, p2, length))
            pts = np.concatenate(
                [m[0][None, :] for m in ln["members"]]
                + [m[1][None, :] for m in ln["members"]]
            )
            wts = np.concatenate([[m[2]] * 2 for m in ln["members"]])
            ln["point"], ln["dir"] = _fit_line(pts, np.array(wts))
            ln["support"] = sum(m[2] for m in ln["members"])
            placed = True
            break
        if not placed:
            lines.append({
                "point": mid.copy(),
                "dir": (p2 - p1) / max(np.linalg.norm(p2 - p1), 1e-9),
                "members": [(p1, p2, length)],
                "support": length,
            })
    for ln in lines:
        ts = []
        for p1, p2, _ in ln["members"]:
            ts += [float((p1 - ln["point"]) @ ln["dir"]),
                   float((p2 - ln["point"]) @ ln["dir"])]
        ln["tmin"], ln["tmax"] = min(ts), max(ts)
        ln["extent"] = ln["tmax"] - ln["tmin"]
    return [ln for ln in lines
            if ln["support"] >= _LINE_SUPPORT_MIN and ln["extent"] >= 46]


def _intersect(l1, l2):
    a = np.stack([l1["dir"], -l2["dir"]], axis=1)
    det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
    if abs(det) < 1e-9:
        return None
    t = np.linalg.solve(a, l2["point"] - l1["point"])
    return l1["point"] + t[0] * l1["dir"]


def _offset_of(line, point):
    n = np.array([-line["dir"][1], line["dir"][0]])
    return float((point - line["point"]) @ n)


def _assemble_rectangles(lines, width, height):
    quads = []
    n = len(lines)
    for i in range(n):
        for j in range(i + 1, n):
            l1, l2 = lines[i], lines[j]
            if abs(float(l1["dir"] @ l2["dir"])) < np.cos(np.radians(5)):
                continue
            sep = abs(_offset_of(l1, l2["point"]))
            if sep < 38 or sep > 750:
                continue
            c = float((l2["point"] - l1["point"]) @ l1["dir"])
            a0, a1 = c + l2["tmin"], c + l2["tmax"]
            lo = max(l1["tmin"], a0)
            hi = min(l1["tmax"], a1)
            overlap = hi - lo
            if overlap < 42 or overlap < 0.5 * min(l1["extent"], l2["extent"]):
                continue
            closers = []
            for k in range(n):
                if k in (i, j):
                    continue
                m = lines[k]
                if abs(float(m["dir"] @ l1["dir"])) > np.cos(np.radians(83)):
                    continue
                offs = []
                for p1, p2, _ in m["members"]:
                    offs += [abs(_offset_of(l1, p1)), abs(_offset_of(l1, p2))]
                if max(offs) < sep - 22 or min(offs) > 22:
                    continue
                closers.append((float((m["point"] - l1["point"]) @ l1["dir"]), m))
            closers.sort(key=lambda cm: cm[0])
            for a in range(len(closers)):
                for b in range(a + 1, len(closers)):
                    ca, ma = closers[a]
                    cb, mb = closers[b]
                    w_ab = cb - ca
                    if w_ab < 42 or w_ab > 900:
                        continue
                    if ca > lo + 26 or cb < hi - 26:
                        continue
                    corners = [_intersect(l1, ma), _intersect(l1, mb),
                               _intersect(l2, mb), _intersect(l2, ma)]
                    if any(cc is None for cc in corners):
                        continue
                    corners = np.array(corners)
                    if (corners[:, 0] < -4).any() or (corners[:, 0] > width + 4).any() \
                            or (corners[:, 1] < -4).any() or (corners[:, 1] > height + 4).any():
                        continue
                    sides = [np.linalg.norm(corners[(t + 1) % 4] - corners[t])
                             for t in range(4)]
                    wm = (sides[0] + sides[2]) / 2
                    hm = (sides[1] + sides[3]) / 2
                    if min(wm, hm) < 40 or max(wm, hm) / min(wm, hm) > 4.6:
                        continue
                    support = (l1["support"] + l2["support"]
                               + ma["support"] + mb["support"])
                    quads.append((corners, wm, hm, support))
    quads.sort(key=lambda q: -q[3])
    kept = []
    for q in quads:
        cen = q[0].mean(axis=0)
        dup = False
        for kq in kept:
            if np.linalg.norm(cen - kq[0].mean(axis=0)) < 20 \
                    and abs(q[1] - kq[1]) < 0.18 * q[1] \
                    and abs(q[2] - kq[2]) < 0.18 * q[2]:
                dup = True
                break
        if not dup:
            kept.append(q)
    return kept


def _detect_rectangles(gray):
    """Candidate rectangles in one frame (list of (corners, w_px, h_px,
    edge support)). Deterministic; pure OpenCV + numpy."""
    import cv2

    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD, scale=0.8)
    lines, *_ = lsd.detect(gray)
    if lines is None:
        return []
    segs = []
    for (x1, y1, x2, y2) in lines.reshape(-1, 4):
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        length = float(np.linalg.norm(p2 - p1))
        if length >= _MIN_SEGMENT:
            segs.append((p1, p2, length))
    if not segs:
        return []
    return _assemble_rectangles(_cluster_lines(segs),
                                gray.shape[1], gray.shape[0])


# ---------------------------------------------------------------------------
# Per-frame depth measurement (Pioneer A's machinery, via fuse's helpers).
# ---------------------------------------------------------------------------
def _frame_fits(job_dir, depth_dir):
    """(a_glob, b_glob, per-frame fits) exactly as stage_fuse computes
    them, or None when the job has too few COLMAP anchor pairs."""
    from .fuse import _collect_anchor_pairs, _fit_mad, _robust_affine_fit

    rel_vals, inv_z, per_frame = _collect_anchor_pairs(job_dir, depth_dir)
    if len(rel_vals) < 30:
        return None
    a_glob, b_glob = _robust_affine_fit(inv_z, rel_vals)
    frame_fits = {}
    for stem, (f_rel, f_inv) in per_frame.items():
        if len(f_rel) < 60:
            continue
        a_i, b_i = _robust_affine_fit(f_inv, f_rel)
        if a_i <= 0:
            continue
        if _fit_mad(f_inv, f_rel, a_i, b_i) <= _fit_mad(f_inv, f_rel, a_glob, b_glob):
            frame_fits[stem] = (a_i, b_i)
    return a_glob, b_glob, frame_fits


def _z_at(rel, px, py, a, b):
    """Median aligned depth over a 7x7 window around a pixel."""
    h, w = rel.shape
    x0, x1 = max(0, int(round(px)) - 3), min(w, int(round(px)) + 4)
    y0, y1 = max(0, int(round(py)) - 3), min(h, int(round(py)) + 4)
    patch = rel[y0:y1, x0:x1]
    denom = patch - b
    ok = denom > 1e-4
    if ok.sum() < 8:
        return None
    z = a / denom[ok]
    z = z[np.isfinite(z)]
    if len(z) < 8:
        return None
    return float(np.median(z))


def _backproject(pose, px, py, z):
    from .fuse import _quat_xyzw_to_matrix

    cam = pose["camera"]["params"]
    f, cx, cy = float(cam[0]), float(cam[1]), float(cam[2])
    rot = _quat_xyzw_to_matrix(pose["cam_from_world_quat_xyzw"])
    t = np.array(pose["cam_from_world_t"], dtype=np.float64)
    xc = np.array([(px - cx) / f * z, (py - cy) / f * z, z])
    return (xc - t) @ rot


def _order_quad(pts):
    pts = np.asarray(pts, dtype=np.float64)
    cen = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - cen[1], pts[:, 0] - cen[0])
    pts = pts[np.argsort(ang)]
    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
    return np.roll(pts, -start, axis=0)


def _measure_quad(pose, rel, fit, quad, entry):
    """Verify + measure one candidate rectangle in one frame.

    Returns a measurement dict (centroid in the COLMAP frame, width /
    height in scene units, gang count, scale samples) or None when any
    verification gate rejects the candidate.
    """
    from echo_super.ds_batch1 import estimate_scale
    from echo_super.ds_batch2 import plane_fit_lsq

    a, b = fit
    unit = float(entry["unit_m"])
    max_gang = int(entry["max_gang"])
    agree_tol = float(entry.get("agree_tol", _AGREE_TOL_DEFAULT))
    q = _order_quad(quad)
    cen2 = q.mean(axis=0)
    corners = []
    for (px, py) in q:
        # Depth is sampled a few px inward (monocular depth smears
        # across the border); the back-projection ray uses the true
        # corner pixel, so no size bias is introduced.
        sx = px + (cen2[0] - px) * 0.045
        sy = py + (cen2[1] - py) * 0.045
        z = _z_at(rel, sx, sy, a, b)
        if z is None or not (0.2 < z < 40):
            return None
        corners.append(_backproject(pose, px, py, z))
    c = np.array(corners)
    diag = np.linalg.norm(c[2] - c[0])
    if diag < 0.05:
        return None
    if np.linalg.norm(c[0] + c[2] - c[1] - c[3]) > 0.10 * diag:
        return None  # corners not coplanar: not one flat plate
    w = (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2
    h = (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])) / 2
    if min(w, h) < 0.04:
        return None
    # Gang hypothesis: height is one unit; width is n units. The two
    # implied scales must agree -- a wrong rectangle almost never does.
    s_h = unit / h
    best = None
    for n in range(1, max_gang + 1):
        s_w = n * unit / w
        d = abs(s_w - s_h) / ((s_w + s_h) / 2)
        if best is None or d < best[0]:
            best = (d, n, s_w)
    rel_diff, gang, s_w = best
    if rel_diff > agree_tol or not (0.72 * gang < w / h < 1.38 * gang):
        return None
    # Centreline cross-check with the registered estimate_scale kernel
    # (the same cross-check Pioneer A ran).
    mid0 = (q[0] + q[3]) / 2
    mid1 = (q[1] + q[2]) / 2
    pts = []
    for t in np.linspace(0.04, 0.96, 24):
        px, py = mid0 * (1 - t) + mid1 * t
        z = _z_at(rel, px, py, a, b)
        if z is not None and 0.2 < z < 40:
            pts.append(_backproject(pose, px, py, z))
    s_line = None
    if len(pts) >= 8:
        try:
            s_line, _ = estimate_scale(np.array(pts), gang * unit)
        except ValueError:
            s_line = None
    if s_line is not None and abs(s_line - s_h) / s_h > _LINE_XCHECK_TOL:
        return None
    # Wall support: the ring around the plate must fit one plane (the
    # wall) and the plate must sit on it.
    ring = []
    qw = q[:, 0].max() - q[:, 0].min()
    qh = q[:, 1].max() - q[:, 1].min()
    for gx in np.linspace(0, 1, 9):
        for gy in np.linspace(0, 1, 9):
            if 0.22 < gx < 0.78 and 0.22 < gy < 0.78:
                continue
            px = cen2[0] + (gx - 0.5) * 2 * qw * 0.62
            py = cen2[1] + (gy - 0.5) * 2 * qh * 0.62
            z = _z_at(rel, px, py, a, b)
            if z is not None and 0.2 < z < 40:
                ring.append(_backproject(pose, px, py, z))
    if len(ring) < 12:
        return None
    try:
        normal, offset, dists = plane_fit_lsq(np.array(ring))
    except ValueError:
        return None
    rms = float(np.sqrt(np.mean(dists ** 2)))
    cen3 = c.mean(axis=0)
    if rms > 0.30 * h or abs(float(normal @ cen3 + offset)) > 0.45 * h:
        return None
    samples = [s_h, s_w] + ([s_line] if s_line is not None else [])
    return {"centroid": cen3, "width_units": float(w), "height_units": float(h),
            "gang": gang, "samples": [float(s) for s in samples]}


# ---------------------------------------------------------------------------
# Corroboration: clustering, scale groups, and the room sanity gate.
# ---------------------------------------------------------------------------
def _cluster_measurements(measurements):
    clusters = []
    for r in measurements:
        best_i, best_d = None, _CLUSTER_RADIUS
        for i, cl in enumerate(clusters):
            d = float(np.linalg.norm(r["centroid"] - cl["seed"]))
            if d < best_d:
                best_i, best_d = i, d
        if best_i is None:
            clusters.append({"seed": r["centroid"].copy(), "members": [r]})
        else:
            clusters[best_i]["members"].append(r)
    accepted = []
    for cl in clusters:
        frames = {m["frame"] for m in cl["members"]}
        samples = [s for m in cl["members"] for s in m["samples"]]
        med = float(np.median(samples))
        q25, q75 = np.percentile(samples, [25, 75])
        spread = float((q75 - q25) / med) if med > 0 else 9.9
        if len(frames) >= 2 and spread <= _CLUSTER_SPREAD_MAX:
            accepted.append({
                "median": med,
                "samples": samples,
                "frames": frames,
                "centroid": np.mean([m["centroid"] for m in cl["members"]], axis=0),
                "gangs": sorted({m["gang"] for m in cl["members"]}),
            })
    return accepted


def _sanity_refs(job_dir, poses):
    """The room's own dimensions in scene units, from the job's data:
    (height, plan_a, plan_b, route). Height/plan are measured along the
    camera-estimated up direction (poses frame); when points.bin lives
    in the exported frame (alignment.json present) the points are first
    carried back, so both production and legacy jobs read consistently.
    Returns None when the cloud is unusable."""
    from .fuse import _estimate_world_up, _quat_xyzw_to_matrix

    points_path = job_dir / "points.bin"
    if not points_path.exists():
        return None
    raw = np.fromfile(points_path, dtype=np.uint8)
    if raw.size % 15:
        return None
    pts = raw.reshape(-1, 15)[:, :12].copy().view(np.float32) \
        .reshape(-1, 3).astype(np.float64)
    if len(pts) < 100:
        return None
    align_path = job_dir / "alignment.json"
    if align_path.exists():
        try:
            rot = np.array(json.loads(align_path.read_text())["rotation"],
                           dtype=np.float64)
            pts = pts @ rot  # exported = colmap @ R.T  ->  colmap = exported @ R
        except (OSError, ValueError, KeyError):
            pass
    up = _estimate_world_up(poses)
    if up is None:
        return None
    a1 = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = a1 - (a1 @ up) * up
    u /= np.linalg.norm(u)
    v = np.cross(up, u)

    def span(axis):
        s = pts @ axis
        lo, hi = np.percentile(s, [1, 99])
        return float(hi - lo)

    centres = []
    for name in sorted(poses):
        p = poses[name]
        rot_m = _quat_xyzw_to_matrix(p["cam_from_world_quat_xyzw"])
        centres.append(-rot_m.T @ np.array(p["cam_from_world_t"], dtype=np.float64))
    route = float(np.linalg.norm(np.diff(np.array(centres), axis=0), axis=1).sum()) \
        if len(centres) > 1 else 0.0
    return span(up), span(u), span(v), route


def _is_sane(scale, refs):
    height, plan_a, plan_b, route = refs
    return (
        _CEILING_RANGE_M[0] <= height * scale <= _CEILING_RANGE_M[1]
        and _PLAN_RANGE_M[0] <= plan_a * scale <= _PLAN_RANGE_M[1]
        and _PLAN_RANGE_M[0] <= plan_b * scale <= _PLAN_RANGE_M[1]
        and _ROUTE_RANGE_M[0] <= route * scale <= _ROUTE_RANGE_M[1]
    )


def _decide_scale(clusters, refs):
    """Pick the winning scale group, or None.

    Groups are formed around each accepted cluster (medians within
    _GROUP_AGREE), strongest first (most clusters, then most samples).
    A group wins only if the scale it implies for the room itself is
    plausible AND it is corroborated by >= 2 distinct anchors or
    >= 3 distinct frames. Deterministic throughout -- corroboration
    and consistency gates, never a vote that could move a value: the
    adopted scale is the pooled median of the winning group's own
    measurements.
    """
    if not clusters or refs is None:
        return None
    groups = []
    for cl in clusters:
        members = [o for o in clusters
                   if abs(o["median"] - cl["median"]) <= _GROUP_AGREE * cl["median"]]
        groups.append(members)
    groups.sort(key=lambda g: (-len(g), -sum(len(o["samples"]) for o in g)))
    seen = set()
    for g in groups:
        key = tuple(sorted(id(o) for o in g))
        if key in seen:
            continue
        seen.add(key)
        samples = [s for o in g for s in o["samples"]]
        med = float(np.median(samples))
        frames = set().union(*[o["frames"] for o in g])
        if not _is_sane(med, refs):
            continue
        if not (len(g) >= 2 or len(frames) >= 3):
            continue
        p16, p84 = np.percentile(samples, [16, 84])
        half_spread = float((p84 - p16) / 2 / med) if med > 0 else 1.0
        return {
            "scale": med,
            "samples": samples,
            "clusters": g,
            "frames": frames,
            "half_spread": half_spread,
        }
    return None


# ---------------------------------------------------------------------------
# Applying the scale to the job's data (idempotent).
# ---------------------------------------------------------------------------
def _scale_points_file(path, scale):
    from .fuse import _write_points_bin

    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % 15:
        return False
    raw = raw.reshape(-1, 15)
    pts = raw[:, :12].copy().view(np.float32).reshape(-1, 3).astype(np.float64)
    colors = raw[:, 12:15].copy()
    _write_points_bin(pts * scale, colors, path)
    return True


def _scale_ply(path, scale):
    from .fuse import _write_binary_ply

    text = path.read_bytes()
    marker = b"end_header\n"
    idx = text.find(marker)
    if idx < 0 or b"binary_little_endian" not in text[:idx]:
        return False
    body = np.frombuffer(text[idx + len(marker):], dtype=np.uint8)
    if body.size % 15:
        return False
    body = body.reshape(-1, 15)
    pts = body[:, :12].copy().view(np.float32).reshape(-1, 3).astype(np.float64)
    colors = body[:, 12:15].copy()
    _write_binary_ply(pts * scale, colors, path)
    return True


def _scale_sparse_ply(path, scale):
    lines = path.read_text().splitlines()
    out = []
    in_body = False
    for line in lines:
        if not in_body:
            out.append(line)
            if line.strip() == "end_header":
                in_body = True
            continue
        parts = line.split()
        if len(parts) < 6:
            out.append(line)
            continue
        x, y, z = (float(parts[0]) * scale, float(parts[1]) * scale,
                   float(parts[2]) * scale)
        out.append(f"{x:.4f} {y:.4f} {z:.4f} "
                   f"{int(float(parts[3]))} {int(float(parts[4]))} {int(float(parts[5]))}")
    path.write_text("\n".join(out) + "\n")
    return True


def _apply_scale(job_dir, scale):
    """Multiply every spatial artefact of the job by `scale`.

    points.bin order (and therefore points_frame_ids.npy) is
    untouched -- only coordinates change. Camera translations in
    poses.json scale with the world, so mesh/replay/preview camera
    geometry stays consistent with the points.
    """
    applied = []
    if _scale_points_file(job_dir / "points.bin", scale):
        applied.append("points.bin")
    ply = job_dir / "scene_points.ply"
    if ply.exists() and _scale_ply(ply, scale):
        applied.append("scene_points.ply")
    dropped = job_dir / "points_trim_dropped.bin"
    if dropped.exists() and _scale_points_file(dropped, scale):
        applied.append("points_trim_dropped.bin")
    sparse = job_dir / "sparse_points.ply"
    if sparse.exists():
        try:
            if _scale_sparse_ply(sparse, scale):
                applied.append("sparse_points.ply")
        except (OSError, ValueError):
            pass
    poses_path = job_dir / "poses.json"
    if poses_path.exists():
        poses = json.loads(poses_path.read_text())
        for pose in poses.values():
            pose["cam_from_world_t"] = [float(v) * scale
                                        for v in pose["cam_from_world_t"]]
        poses_path.write_text(json.dumps(poses, ensure_ascii=False, indent=2))
        applied.append("poses.json")
    return applied


# ---------------------------------------------------------------------------
# The stage.
# ---------------------------------------------------------------------------
def _write_record(job_dir, record):
    (job_dir / "scale.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2))
    info = {"stage": "scale", **record}
    (job_dir / "job.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return info


def stage_scale(job_dir) -> dict:
    """Find a true scale for the job and, when found, put the scene
    into metres. Always writes scale.json; returns the stage info.

    Never raises for missing/degenerate inputs -- those are simply a
    job whose scale stays unknown. (The pipeline wrapper additionally
    guarantees the job proceeds regardless.)
    """
    job_dir = Path(job_dir)
    record = {
        "status": "unknown",
        "scale_to_meters": None,
        "evidence": "未知",
        "error_pct": None,
        "anchor": None,
        "measurements": 0,
        "clusters": 0,
        "frames": 0,
        "applied": False,
        "applied_to": [],
        "reason": None,
    }

    existing_path = job_dir / "scale.json"
    if existing_path.exists():
        try:
            existing = json.loads(existing_path.read_text())
        except (OSError, ValueError):
            existing = {}
        if existing.get("status") == "set" and existing.get("applied"):
            record.update({k: existing.get(k) for k in record if k in existing})
            record["reason"] = "scale already applied by an earlier run"
            return _write_record(job_dir, record)

    markers_path = job_dir / "markers.json"
    if markers_path.exists():
        try:
            markers = json.loads(markers_path.read_text())
        except (OSError, ValueError):
            markers = {}
        if markers.get("scale_to_meters"):
            record.update({
                "status": "set",
                "scale_to_meters": float(markers["scale_to_meters"]),
                "evidence": "实测",
                "anchor": {"id": "markers", "label": "标记纸"},
                "reason": "scene already in metres via the markers stage",
            })
            return _write_record(job_dir, record)

    poses_path = job_dir / "poses.json"
    if not poses_path.exists() or not (job_dir / "points.bin").exists():
        record["reason"] = "poses.json or points.bin missing"
        return _write_record(job_dir, record)
    poses = json.loads(poses_path.read_text())
    fits = _frame_fits(job_dir, job_dir / "depth")
    if fits is None:
        record["reason"] = "too few COLMAP anchor pairs to align depth"
        return _write_record(job_dir, record)
    a_glob, b_glob, frame_fits = fits
    refs = _sanity_refs(job_dir, poses)

    import cv2

    winner = None
    winner_entry = None
    for entry in ANCHOR_CATALOG:
        if entry.get("kind") != "gang_panel":
            continue
        measurements = []
        for name in sorted(poses.keys()):
            stem = Path(name).stem
            img_path = job_dir / "frames" / name
            npy = job_dir / "depth" / f"{stem}.npy"
            if not img_path.exists() or not npy.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            quads = _detect_rectangles(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            if not quads:
                continue
            rel = np.load(npy).astype(np.float64)
            fit = frame_fits.get(stem, (a_glob, b_glob))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for corners, wm, hm, _support in quads:
                # Loose interior-uniformity gate: a plate's face is one
                # surface (plugs and print are allowed, clutter is not).
                cen = corners.mean(axis=0)
                shrunk = (cen + (corners - cen) * 0.6).astype(np.int32)
                mask = np.zeros(gray.shape, np.uint8)
                cv2.fillPoly(mask, [shrunk], 1)
                vals = gray[mask > 0]
                if len(vals) < 60 or float(vals.std()) > 68:
                    continue
                r = _measure_quad(poses[name], rel, fit, corners, entry)
                if r is not None:
                    r["frame"] = stem
                    measurements.append(r)
        clusters = _cluster_measurements(measurements)
        decided = _decide_scale(clusters, refs)
        if decided is not None:
            winner = decided
            winner_entry = entry
            break

    if winner is None:
        record["reason"] = "no anchor group survived verification"
        return _write_record(job_dir, record)

    scale = winner["scale"]
    error_pct = int(np.ceil(max(_MIN_ERROR_PCT,
                                min(_MAX_ERROR_PCT, winner["half_spread"] * 100))))
    applied = _apply_scale(job_dir, scale)
    height, plan_a, plan_b, route = refs
    record.update({
        "status": "set",
        "scale_to_meters": scale,
        "evidence": "估计",
        "error_pct": error_pct,
        "anchor": {
            "id": winner_entry["id"],
            "label": winner_entry["label"],
            "gangs": sorted({g for cl in winner["clusters"] for g in cl["gangs"]}),
        },
        "measurements": len(winner["samples"]),
        "clusters": len(winner["clusters"]),
        "frames": len(winner["frames"]),
        "spread": winner["half_spread"],
        "samples_range": [float(min(winner["samples"])),
                          float(max(winner["samples"]))],
        "implied": {
            "ceiling_m": height * scale,
            "room_m": [plan_a * scale, plan_b * scale],
            "route_m": route * scale,
        },
        "applied": True,
        "applied_to": applied,
    })
    return _write_record(job_dir, record)
