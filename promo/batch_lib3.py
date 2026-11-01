#!/usr/bin/env python3
"""V3 segment-A library: same data/effects as batch_lib (v2) except the
user's three binding fixes.

1. NO BLACK PHOTOS -- hero photos and route cards are re-sampled in
   TRUE colour (v3_assets.npz; v2 baked in a darkening blue grade) and
   the hero photo is quincunx-filled in its own plane so it reads as a
   continuous photograph. A photo appears fully lit at its wall,
   holds INTACT until its crush accent, then dissolves as its points
   fly. Cards pop in lit during the opening and hand over to the
   first hero (fade out across its landing).
2. SEGMENTED CAMERA -- explicit keyframes: for every hero a HOLD
   segment [photo_t - settle, points_t1 + post] at a FIXED 3/4 watch
   pose (breathing only); travels are smootherstep glides between
   holds with route via-keys on long gaps; too-close accent pairs
   merge into one group hold. The camera is therefore always settled
   before a photo appears and holds through the whole crush.
   Schedules are re-spaced: 'accent' = crush start (points_t0),
   pins exact; photo leads the accent by 0.25-0.45s.
3. SOLIDIFICATION SWELL -- settled points grow over SWELL_DUR into
   filled axis-aligned squares of world size PITCH*g (perspective
   correct), drawn by a local njit splat and z-composited against a
   depth buffer of the operator pass. The v2 mesh-points fade is
   gone; the room itself becomes solid, front travelling per hero.

Particle/crush motion math is byte-identical to v2 (p^2 path +
sin(pi * p^0.7) burst, alpha ramp p/0.15); lattice/shards/route
treatment identical; base pass uses the same operator call.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import numpy as np
from numba import njit
import batch_lib as BL

WORK = HERE / 'assets'
W, H = BL.W, BL.H
BG = BL.BG
clamp01, smooth, smoother = BL.clamp01, BL.smooth, BL.smoother
render_frame = BL.render_frame

PTS, START, RGB = BL.PTS, BL.START, BL.RGB
DELTA = BL.DELTA
FRAME_IDS, USED_IDS = BL.FRAME_IDS, BL.USED_IDS
FRAME_GEO = BL.FRAME_GEO
ROUTE_POS, ROUTE_FRAC = BL.ROUTE_POS, BL.ROUTE_FRAC
LAT_LINES, LAT_SHADE = BL.LAT_LINES, BL.LAT_SHADE
SHARD_LINES, SHARD_SHADE = BL.SHARD_LINES, BL.SHARD_SHADE
CENTRE, FLOOR_Y = BL.CENTRE, BL.FLOOR_Y
STEMS = BL.STEMS
NC = BL.NC
route_point = BL.route_point
WIDE_POS = BL.WIDE_POS

_V3 = np.load(WORK / 'v3_assets.npz')
PHOTO_PTS = _V3['photo_pts'].astype(np.float64)
PHOTO_RGB = _V3['photo_rgb'].astype(np.float64)
PHOTO_OWNER = _V3['photo_owner']
PHOTO_FEATHER = _V3['photo_feather'].astype(np.float64)
PHOTO_DU = _V3['photo_du'].astype(np.float64)
PHOTO_DV = _V3['photo_dv'].astype(np.float64)
CARD_PTS = _V3['card_pts'].astype(np.float64)
CARD_RGB = _V3['card_rgb'].astype(np.float64)
CARD_OWNER = _V3['card_owner']
CARD_FEATHER = _V3['card_feather'].astype(np.float64)
CARD_DU = _V3['card_du'].astype(np.float64)
CARD_DV = _V3['card_dv'].astype(np.float64)
CARD_C = _V3['card_c'].astype(np.float64)

def smooth_v(a, b, x):
    p = np.clip((x - a) / (b - a), 0.0, 1.0); return p * p * (3 - 2 * p)
def smoother_v(a, b, x):
    p = np.clip((x - a) / (b - a), 0.0, 1.0); return p * p * p * (p * (p * 6 - 15) + 10)

SWELL_DUR = 1.5
SWELL_PITCH = 0.11
PHOTO_OFFS = [(0.0, 0.0), (0.5, 0.5), (0.5, -0.5), (-0.5, 0.5), (-0.5, -0.5)]
CARD_OFFS = [(a, b) for a in (-0.5, 0.0, 0.5) for b in (-0.5, 0.0, 0.5)]

# ---------------------------------------------------------------------------
# schedule v3
def build_schedule3(A_dur, n_hero, t_start, t_end, pins, mont0,
                    card_start, card_end, route_win, sentimental=False):
    n = len(FRAME_GEO)
    cand = []
    for s in range(n_hero):
        lo = int(n * s / n_hero); hi = max(lo + 1, int(n * (s + 1) / n_hero))
        cand.append(max(range(lo, hi), key=lambda k: FRAME_GEO[k]['n_points']))
    cand = sorted(set(cand)); nh = len(cand)
    # ideal accents by anchor interpolation in rank fraction (as v2)
    anch = [(0.0, t_start)]
    for p in pins:
        q = clamp01((p - t_start) / max(t_end - t_start, 1e-9))
        anch.append((min(max(q, 0.04), 0.98), p))
    anch.append((1.0, t_end)); anch.sort()
    ideal, is_pin = [], [False] * nh
    for r in range(nh):
        fr = r / max(nh - 1, 1); a = t_end
        for (f0, t0), (f1, t1) in zip(anch, anch[1:]):
            if fr <= f1:
                a = t0 + (t1 - t0) * (fr - f0) / max(f1 - f0, 1e-9); break
        ideal.append(a)
    for p in pins:
        k = int(np.argmin([abs(a - p) for a in ideal]))
        ideal[k] = p; is_pin[k] = True
    # spacing repair: pins immovable; fillers need >=1.32s after the
    # previous accent and >=0.97s before the next pin, else dropped
    MIN_GAP, PIN_GAP = 1.32, 0.97
    pin_acc = sorted(pins)
    heroes, accents, pin_flags = [], [], []
    last = -1e18
    for r in range(nh):
        a = ideal[r]
        if is_pin[r]:
            heroes.append(cand[r]); accents.append(a); pin_flags.append(True)
            last = a; continue
        a = max(a, last + MIN_GAP)
        nxt = min([p for p in pin_acc if p > last] or [1e18])
        if a > nxt - PIN_GAP or a > t_end + 0.2:
            continue
        heroes.append(cand[r]); accents.append(a); pin_flags.append(False)
        last = a
    # per-hero timing: photo leads the crush accent
    frame_meta = []
    pt_t0 = np.zeros(len(FRAME_IDS)); pt_t1 = np.zeros(len(FRAME_IDS))
    burst = BL.BURST0.copy()
    prev_t1 = None
    for r, j in enumerate(heroes):
        acc = float(accents[r])
        lead = 0.45 if prev_t1 is None else float(min(0.45, max(0.25, acc - prev_t1 - 0.15)))
        photo_t = acc - lead
        m = dict(FRAME_GEO[j]); m.update({
            'slot': photo_t, 'accent': acc, 'lead': lead,
            'photo_in_end': photo_t + 0.06, 'dissolve_begin': acc,
            'photo_out_end': acc + 0.30, 'points_t0': acc,
            'points_t1': acc + 0.75, 'hero_idx': j, 'is_pin': pin_flags[r]})
        frame_meta.append(m)
        sel = FRAME_IDS == USED_IDS[j]
        pt_t0[sel] = acc; pt_t1[sel] = acc + 0.75
        prev_t1 = acc + 0.75
    hero_set = set(heroes)
    mont = [j for j in range(n) if j not in hero_set]
    for rank, j in enumerate(mont):
        t0 = mont0 + (A_dur - 1.55 - mont0) * rank / max(len(mont) - 1, 1)
        sel = FRAME_IDS == USED_IDS[j]
        pt_t0[sel] = t0; pt_t1[sel] = t0 + 0.45
        burst[sel] *= 0.35
    # watch poses + hold windows (tracker geometry identical to v2).
    # v3 rhythm: EVERY hero is watched from its OWN pose (no merging).
    # The post-hold after a crush shrinks when the next photo follows
    # quickly, and the next settle shrinks with the remaining glide time,
    # so tight accent pairs get a short glide between neighbouring route
    # poses instead of one shared (wrong-wall) pose.
    trackers = []
    prev_hold_end = None
    for i, m in enumerate(frame_meta):
        C = np.array(m['centre']); V = np.array(m['z']); X = np.array(m['x']); d = m['dmed']
        plane_c = C + V * d
        half_w = d * max(m['cx'], m['W'] - m['cx']) / m['f']
        half_h = d * max(m['cy'], m['H'] - m['cy']) / m['f']
        kk = float(np.clip(2.05 * max(half_w, half_h), 0.7, 0.92 * d))
        eye = plane_c - V * kk + X * (half_w * 0.38) + np.array([0.0, 1.0, 0.0]) * (half_h * 0.22)
        route_u = STEMS['cards'].index(m['stem']) / max(NC - 1, 1)
        if i + 1 < len(frame_meta):
            D = frame_meta[i + 1]['slot'] - m['points_t1']
            post = 0.30 if D >= 0.85 else max(0.02, D - 0.55)
        else:
            post = 0.30
        hold_end = m['points_t1'] + post
        if prev_hold_end is None:
            settle = 0.30
        else:
            rem = m['slot'] - prev_hold_end
            settle = float(min(0.30, max(0.02, 0.30 * rem))) if rem > 0.35 else 0.02
        tr = {'slot': m['slot'], 'accent': m['accent'], 'points_t0': m['points_t0'],
              'points_t1': m['points_t1'], 'route_u': route_u, 'target': plane_c,
              'eye': eye, 'is_pin': m['is_pin'],
              'hold_start': m['slot'] - settle, 'hold_end': hold_end}
        trackers.append(tr)
        prev_hold_end = tr['hold_end']
    card_t = card_start + (card_end - card_start) * np.arange(NC) / max(NC - 1, 1)
    return {'A': A_dur, 'frame_meta': frame_meta, 'pt_t0': pt_t0, 'pt_t1': pt_t1,
            'burst': burst, 'trackers': trackers, 'card_t': card_t,
            'cards_out': (frame_meta[0]['slot'] - 0.25, frame_meta[0]['slot'] + 0.55),
            'route': route_win, 'close': (A_dur - 1.55, A_dur - 0.45),
            'sentimental': sentimental}

# ---------------------------------------------------------------------------
# camera v3: hold / travel keyframes
def _cam_keys(sched):
    if 'ckeys' in sched:
        return sched['ckeys']
    # one hold per tracker — no pose merging (v3 fix 2)
    groups = [{'hold_start': tr['hold_start'], 'hold_end': tr['hold_end'],
               'eye': tr['eye'], 'tgt': tr['target'], 'u': tr['route_u']}
              for tr in sched['trackers']]
    keys = []
    u0 = max(groups[0]['u'] - 0.05, 0.0)
    e0 = route_point(u0).copy(); e0[1] = max(e0[1], FLOOR_Y + 0.55)
    ahead = route_point(min(u0 + 0.05, 1.0))
    keys.append((0.0, e0, 0.5 * ahead + 0.5 * groups[0]['tgt']))
    prev_end, prev_u = 0.0, u0
    for g in groups:
        if g['hold_start'] - prev_end > 1.0:
            um = 0.5 * (prev_u + g['u'])
            em = route_point(um).copy(); em[1] = max(em[1], FLOOR_Y + 0.55)
            ah = route_point(min(um + 0.05, 1.0))
            keys.append((0.5 * (prev_end + g['hold_start']), em,
                         0.55 * ah + 0.45 * CENTRE))
        keys.append((g['hold_start'], g['eye'], g['tgt']))
        keys.append((g['hold_end'], g['eye'], g['tgt']))
        prev_end, prev_u = g['hold_end'], g['u']
    keys.append((sched['A'] - 0.45, WIDE_POS, CENTRE))
    out = []
    for k in keys:
        if out and k[0] <= out[-1][0]:
            continue
        out.append(k)
    sched['ckeys'] = out
    sched['groups'] = groups
    return out

def _cam_from(eye, tgt):
    fwd = tgt - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd]); tv = -R @ eye
    fy = (H / 2) / np.tan(0.48)
    K = np.array([[fy, 0, W / 2], [0, fy, H / 2], [0, 0, 1.0]])
    return {'K': K, 'R': R, 't': tv}

def _camera3(sched, t):
    keys = _cam_keys(sched)
    if t <= keys[0][0]:
        eye, tgt = keys[0][1], keys[0][2]
    elif t >= keys[-1][0]:
        eye, tgt = keys[-1][1], keys[-1][2]
    else:
        eye, tgt = keys[-1][1], keys[-1][2]
        for (t0, e0, g0), (t1, e1, g1) in zip(keys, keys[1:]):
            if t0 <= t <= t1:
                u = smoother(t0, t1, t) if t1 > t0 else 1.0
                eye = e0 + (e1 - e0) * u; tgt = g0 + (g1 - g0) * u
                break
    eye = np.array(eye, dtype=np.float64); tgt = np.array(tgt, dtype=np.float64)
    d = eye - tgt
    eye = tgt + d * (1 + 0.010 * np.sin(t * 1.7))   # breathing, ~1%
    return _cam_from(eye, tgt)

# ---------------------------------------------------------------------------
# solid layer (njit splats + depth)
def _project(cam, P):
    Xc = P @ cam['R'].T + cam['t']
    z = Xc[:, 2]
    ok = z > 1e-4
    zs = np.where(ok, z, 1.0)
    u = cam['K'][0, 0] * Xc[:, 0] / zs + cam['K'][0, 2]
    v = cam['K'][1, 1] * Xc[:, 1] / zs + cam['K'][1, 2]
    return u, v, z, ok

@njit(cache=True)
def _depth_splat(zbuf, u, v, z, r, width, height):
    r2 = r * r
    for i in range(u.shape[0]):
        x_lo = int(np.floor(u[i] - r)); x_hi = int(np.ceil(u[i] + r))
        y_lo = int(np.floor(v[i] - r)); y_hi = int(np.ceil(v[i] + r))
        if x_lo < 0: x_lo = 0
        if y_lo < 0: y_lo = 0
        if x_hi > width - 1: x_hi = width - 1
        if y_hi > height - 1: y_hi = height - 1
        for py in range(y_lo, y_hi + 1):
            dy = (py + 0.5) - v[i]; dy2 = dy * dy
            if dy2 > r2: continue
            for px in range(x_lo, x_hi + 1):
                dx = (px + 0.5) - u[i]
                if dx * dx + dy2 <= r2 and z[i] < zbuf[py, px]:
                    zbuf[py, px] = z[i]

@njit(cache=True)
def _solid_splat(colbuf, zbuf, u, v, z, rgb, side, width, height):
    for i in range(u.shape[0]):
        h = 0.5 * side[i]
        if h < 0.75: h = 0.75
        x_lo = int(np.floor(u[i] - h)); x_hi = int(np.ceil(u[i] + h))
        y_lo = int(np.floor(v[i] - h)); y_hi = int(np.ceil(v[i] + h))
        if x_lo < 0: x_lo = 0
        if y_lo < 0: y_lo = 0
        if x_hi > width - 1: x_hi = width - 1
        if y_hi > height - 1: y_hi = height - 1
        zi = z[i]
        for py in range(y_lo, y_hi + 1):
            for px in range(x_lo, x_hi + 1):
                if zi < zbuf[py, px]:
                    zbuf[py, px] = zi
                    colbuf[py, px, 0] = rgb[i, 0]
                    colbuf[py, px, 1] = rgb[i, 1]
                    colbuf[py, px, 2] = rgb[i, 2]

def _solid_composite(base_img, cam, base_pts, sol_pos, sol_col, sol_g):
    """Draw swollen points as filled squares; keep whichever of
    (solid, base pass) is nearer per pixel."""
    fy = cam['K'][0, 0]
    bu, bv, bz, bok = _project(cam, base_pts)
    base_z = np.full((H, W), np.inf, dtype=np.float64)
    if bok.any():
        _depth_splat(base_z, bu[bok], bv[bok], bz[bok], 2.4, W, H)
    su, sv, sz, sok = _project(cam, sol_pos)
    sol_colb = np.zeros((H, W, 3), dtype=np.float32)
    sol_z = np.full((H, W), np.inf, dtype=np.float64)
    if sok.any():
        idx = np.nonzero(sok)[0]
        order = idx[np.argsort(-sz[idx], kind='stable')]
        side = np.clip(fy * SWELL_PITCH * sol_g[order] / sz[order], 1.5, 150.0)
        _solid_splat(sol_colb, sol_z, su[order], sv[order], sz[order],
                     sol_col[order].astype(np.float32), side, W, H)
    mask = sol_z < base_z
    out = base_img.astype(np.float32)
    out[mask] = sol_colb[mask]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------------
# segment A frame
def _active_hero(sched, t):
    act = None
    for m in sched['frame_meta']:
        if m['slot'] <= t <= m['photo_out_end']: act = m
        elif m['slot'] > t: break
    return act

def _photo_opacity(m, t):
    if t < m['slot'] or t > m['photo_out_end']: return 0.0
    if t < m['photo_in_end']: return (t - m['slot']) / max(m['photo_in_end'] - m['slot'], 1e-4)
    if t <= m['dissolve_begin']: return 1.0
    return max(0.0, 1 - (t - m['dissolve_begin']) / max(m['photo_out_end'] - m['dissolve_begin'], 1e-4))

def render_a_frame3(sched, t, still=False):
    if still:
        cam = _cam_from(WIDE_POS, CENTRE)
        close_u = 1.0; fade_in = 1.0
    else:
        cam = _camera3(sched, t)
        close_u = smooth(sched['close'][0], sched['close'][1], t)
        fade_in = smooth(0, 4.2, t) if sched['sentimental'] else smooth(0, 2, t)
    # ---- lines: global lattice + shards + route (identical to v2) ----
    scol = np.array([0.30, 0.55, 1.0]) * 255.0
    lat_a = fade_in * (1.0 - 0.25 * close_u)
    lat_rgb = BG[None, :] + (scol[None, :] - BG[None, :]) * (0.55 * LAT_SHADE * lat_a)[:, None]
    sh_col = (np.array([0.24, 0.5, 1.0]) * 255.0) * (0.35 + 0.65 * SHARD_SHADE)[:, None]
    sh_a = (0.30 * fade_in) * (0.25 + 0.75 * SHARD_SHADE)[:, None]
    shard_rgb = BG[None, :] + (sh_col - BG[None, :]) * sh_a
    line_list = [LAT_LINES, SHARD_LINES]; line_cols = [lat_rgb, shard_rgb]
    r0, r1 = sched['route']
    u_route = 1.0 if still else clamp01((t - r0) / (r1 - r0))
    route_fade = 1.0 - 0.8 * close_u
    if u_route > 0:
        segs, scols = [], []
        rc = np.array([0.35, 0.66, 1.0]) * 255.0
        eff = np.clip(BG + rc * (0.9 * route_fade), 0, 255)
        for i in range(NC - 1):
            f0, f1 = ROUTE_FRAC[i], ROUTE_FRAC[i + 1]
            if u_route <= f0: continue
            a, b = ROUTE_POS[i], ROUTE_POS[i + 1]
            if u_route < f1:
                b = a + (b - a) * ((u_route - f0) / max(f1 - f0, 1e-9))
            segs.append((a, b)); scols.append(eff)
        if segs:
            line_list.append(np.array(segs)); line_cols.append(np.array(scols))
    lines = np.concatenate(line_list); lines_rgb = np.concatenate(line_cols)
    # ---- scene points: flying (operator) vs settled (solid layer) ----
    chunks_p, chunks_c = [], []
    sol_parts = []
    if still:
        sol_parts.append((PTS, RGB, np.ones(len(PTS))))
    else:
        t0, t1 = sched['pt_t0'], sched['pt_t1']
        act = t >= t0
        if act.any():
            idx = np.nonzero(act)[0]
            tt0, tt1 = t0[idx], t1[idx]
            p = np.clip((t - tt0) / np.maximum(tt1 - tt0, 1e-4), 0, 1)
            g = smoother_v(tt1, tt1 + SWELL_DUR, t)
            fly = g <= 0.001
            if fly.any():
                fi = idx[fly]; pf = p[fly]
                pos = START[fi] + DELTA[fi] * (pf * pf)[:, None] \
                    + sched['burst'][fi] * np.sin(np.pi * pf ** 0.7)[:, None]
                al = np.clip(pf / 0.15, 0, 1)[:, None]
                chunks_p.append(pos)
                chunks_c.append(BG[None, :] + (RGB[fi] - BG[None, :]) * al)
            if (~fly).any():
                si = idx[~fly]
                sol_parts.append((PTS[si], RGB[si], g[~fly]))
        # ---- hero photo on its wall: lit, intact, then dissolving ----
        m = _active_hero(sched, t)
        if m is not None:
            op = _photo_opacity(m, t)
            if op > 0.02:
                sel = PHOTO_OWNER == m['hero_idx']
                base = PHOTO_PTS[sel]
                du, dv = PHOTO_DU[m['hero_idx']], PHOTO_DV[m['hero_idx']]
                pp = np.concatenate([base + a * du + b * dv for a, b in PHOTO_OFFS])
                cc = BG[None, :] + (PHOTO_RGB[sel] - BG[None, :]) \
                    * (op * PHOTO_FEATHER)[:, None]
                chunks_p.append(pp); chunks_c.append(np.tile(cc, (len(PHOTO_OFFS), 1)))
        # ---- route cards: lit mini-photos of the opening ----
        out0, out1 = sched['cards_out']
        card_alpha = smooth_v(sched['card_t'], sched['card_t'] + 0.15, t) \
            * (1 - smooth(out0, out1, t))
        keep = np.nonzero(card_alpha > 0.03)[0]
        if len(keep):
            eye = -cam['R'].T @ cam['t']
            cen = CARD_C[keep]
            Xc = cen @ cam['R'].T + cam['t']
            dist = np.linalg.norm(cen - eye[None, :], axis=1)
            keep = keep[(Xc[:, 2] > 0.3) & (dist <= 7.0)]
            for ci in keep:
                sel = CARD_OWNER == ci
                base = CARD_PTS[sel]
                du, dv = CARD_DU[ci], CARD_DV[ci]
                pp = np.concatenate([base + a * du + b * dv for a, b in CARD_OFFS])
                cc = BG[None, :] + (CARD_RGB[sel] - BG[None, :]) \
                    * (card_alpha[ci] * CARD_FEATHER)[:, None]
                chunks_p.append(pp); chunks_c.append(np.tile(cc, (len(CARD_OFFS), 1)))
        # ---- route nodes ----
        if u_route > 0:
            nsel = ROUTE_FRAC <= u_route
            if nsel.any():
                ncol = BG + (np.array([0.4, 0.7, 1.0]) * 255.0 - BG) * (0.75 * route_fade)
                cols = np.tile(ncol, (int(nsel.sum()), 1))
                cols[-1] = BG + (np.array([0.85, 0.95, 1.0]) * 255.0 - BG) * route_fade
                chunks_p.append(ROUTE_POS[nsel]); chunks_c.append(cols)
    if chunks_p:
        all_p = np.concatenate(chunks_p).astype(np.float32)
        all_c = np.clip(np.concatenate(chunks_c), 0, 255).astype(np.uint8)
    else:
        all_p = np.zeros((0, 3), np.float32); all_c = np.zeros((0, 3), np.uint8)
    img = render_frame(all_p, all_c, lines.astype(np.float32),
                       np.clip(lines_rgb, 0, 255).astype(np.uint8),
                       camera=cam, width=W, height=H, point_radius_px=2.4,
                       glow_strength=0.5, glow_radius_factor=2.0,
                       background_rgb=(3, 6, 11))
    if sol_parts:
        sol_pos = np.concatenate([p for p, c, gg in sol_parts])
        sol_col = np.concatenate([c for p, c, gg in sol_parts])
        sol_g = np.concatenate([gg for p, c, gg in sol_parts])
        img = _solid_composite(img, cam, all_p.astype(np.float64), sol_pos, sol_col, sol_g)
    return img
