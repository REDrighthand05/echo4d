#!/usr/bin/env python3
"""Shared library for the 4 batch promos: per-track schedules pinned to
each track's beat map, segment-A frames via the Echo Render operator (points+lines),
segment-B frames via PIL. Choreography mirrors echo_scene/preview.py's
promo mode (same geometry, same effect speeds)."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
APP = HERE.parent
ASSETS = HERE / 'assets'
TMP = HERE / 'work'
sys.path.insert(0, str(APP))
sys.path.insert(0, str(HERE))
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
try:
    from echo_super.kernels.echo_render import render_frame
except ImportError:  # pre-rename tree
    from echo_super.kernels.mus_render import render_frame

WORK = ASSETS  # frozen layout: data assets live in promo/assets
W, H, FPS = 1920, 1080, 25
BG = np.array([3.0, 6.0, 11.0])

D = np.load(WORK / 'data.npz')
FRAME_GEO = json.loads((WORK / 'frame_geo.json').read_text())
PTS = D['pts'].astype(np.float64); START = D['start_pos'].astype(np.float64)
RGB = D['rgb'].astype(np.float64); BURST0 = D['burst'].astype(np.float64)
FRAME_IDS = D['frame_ids']; USED_IDS = D['used_ids']
CARD_PTS = D['card_pts'].astype(np.float64); CARD_RGB = D['card_rgb'].astype(np.float64)
CARD_OWNER = D['card_owner']; NC = int(CARD_OWNER.max()) + 1
ROUTE_POS = D['route_pos'].astype(np.float64); ROUTE_FRAC = D['route_frac'].astype(np.float64)
MESH_POS = D['mesh_pos'].astype(np.float64); MESH_COL = D['mesh_col'].astype(np.float64)
GRID_LINES = D['grid_lines'].astype(np.float64); GRID_SHADE = D['grid_shade'].astype(np.float64)
PHOTO_PTS = D['photo_pts'].astype(np.float64); PHOTO_RGB = D['photo_rgb'].astype(np.float64)
PHOTO_OWNER = D['photo_owner']
BMIN = D['bmin']; BMAX = D['bmax']; FLOOR_Y = float(D['floor_y'])
_S = np.load(WORK / 'surface.npz')
SURF_LINES = _S['surf_lines'].astype(np.float64)   # (v1 pilot only; v2 does not draw these)
PHOTO_FEATHER = _S['photo_feather'].astype(np.float64)
_G2 = np.load(WORK / 'grid2.npz')
LAT_LINES = _G2['lat_lines'].astype(np.float64)    # v2: GLOBAL lattice filling the space
LAT_SHADE = _G2['lat_shade'].astype(np.float64)
SHARD_LINES = GRID_LINES[-150:]                    # scattered shards stay
SHARD_SHADE = GRID_SHADE[-150:]
STEMS = json.loads((WORK / 'stems.json').read_text())
CENTRE = np.array([(BMIN[0] + BMAX[0]) / 2, FLOOR_Y + 0.34 * (BMAX[1] - FLOOR_Y),
                   (BMIN[2] + BMAX[2]) / 2])
EXTENT = float(np.max(BMAX - BMIN))
RADIUS0 = EXTENT * 1.30
DELTA = PTS - START

def clamp01(x): return float(np.clip(x, 0.0, 1.0))
def smooth(a, b, x):
    p = clamp01((x - a) / (b - a)); return p * p * (3 - 2 * p)
def smoother(a, b, x):
    """Smootherstep (quintic): zero 1st+2nd derivative at both ends."""
    p = clamp01((x - a) / (b - a)); return p * p * p * (p * (p * 6 - 15) + 10)
def _orbit_phase(t, ramp=2.0):
    """Integral of a smoothstep velocity ramp: the ambient orbit starts
    at rest and eases up to full drift speed -- no constant-speed jump."""
    if t <= 0: return 0.0
    if t >= ramp:
        return t - ramp * 0.5
    u = t / ramp
    return ramp * (u ** 3 - 0.5 * u ** 4)
def ease_io(p): return 2 * p * p if p < 0.5 else 1 - ((-2 * p + 2) ** 2) / 2
def ease_out(p): return 1 - (1 - p) ** 3
def lerp(a, b, u): return a + (b - a) * u

# ---------------------------------------------------------------------------
# schedules
def pick_pins(accents, t_lo, t_hi, gap, cap=5, min_strength=0.85):
    cand = [(t, s) for t, s in accents if min_strength <= s and t_lo <= t <= t_hi]
    cand.sort(key=lambda p: -p[1])
    chosen = []
    for t, s in cand:
        if all(abs(t - c) >= gap for c in chosen):
            chosen.append(t)
        if len(chosen) >= cap:
            break
    return sorted(chosen)

def build_schedule(A_dur, accents, n_hero, t_start, t_end, pins, mont0,
                   card_start, card_span, route_win, sentimental=False):
    n = len(FRAME_GEO)
    heroes = []
    for s in range(n_hero):
        lo = int(n * s / n_hero); hi = max(lo + 1, int(n * (s + 1) / n_hero))
        heroes.append(max(range(lo, hi), key=lambda k: FRAME_GEO[k]['n_points']))
    heroes = sorted(set(heroes)); nh = len(heroes)
    hero_set = set(heroes)
    # anchor interpolation in (rank_frac, time), pins snapped after
    anch = [(0.0, t_start)]
    for p in pins:
        q = clamp01((p - 0.10 - t_start) / max(t_end - t_start, 1e-9))
        anch.append((min(max(q, 0.04), 0.98), p - 0.10))
    anch.append((1.0, t_end))
    anch.sort()
    slots = []
    for r, j in enumerate(heroes):
        fr = r / max(nh - 1, 1)
        t_slot = t_end
        for (f0, t0), (f1, t1) in zip(anch, anch[1:]):
            if fr <= f1:
                t_slot = t0 + (t1 - t0) * (fr - f0) / max(f1 - f0, 1e-9); break
        slots.append(t_slot)
    # snap heroes nearest each pin exactly onto it
    for p in pins:
        k = int(np.argmin([abs(s - (p - 0.10)) for s in slots]))
        slots[k] = p - 0.10
    slots = list(np.maximum.accumulate(np.array(slots)))
    slots = [min(s, A_dur - 1.0) for s in slots]
    frame_meta = []
    pt_t0 = np.zeros(len(FRAME_IDS)); pt_t1 = np.zeros(len(FRAME_IDS))
    burst = BURST0.copy()
    for r, j in enumerate(heroes):
        slot = float(slots[r])
        m = dict(FRAME_GEO[j]); m.update({
            'slot': slot, 'photo_in_end': slot + 0.10, 'dissolve_begin': slot + 0.10,
            'photo_out_end': slot + 0.28, 'points_t0': slot + 0.10,
            'points_t1': slot + 0.10 + 0.75, 'hero_idx': j})
        frame_meta.append(m)
        sel = FRAME_IDS == USED_IDS[j]
        pt_t0[sel] = m['points_t0']; pt_t1[sel] = m['points_t1']
    mont = [j for j in range(n) if j not in hero_set]
    for rank, j in enumerate(mont):
        t0 = mont0 + (A_dur - 1.55 - mont0) * rank / max(len(mont) - 1, 1)
        sel = FRAME_IDS == USED_IDS[j]
        pt_t0[sel] = t0; pt_t1[sel] = t0 + 0.45
        burst[sel] *= 0.35
    # tracking poses for EVERY hero: the camera follows each photo as it
    # lands on its wall and crushes into points (user's look spec)
    trackers = []
    for m in frame_meta:
        C = np.array(m['centre']); V = np.array(m['z']); X = np.array(m['x']); d = m['dmed']
        plane_c = C + V * d
        half_w = d * max(m['cx'], m['W'] - m['cx']) / m['f']
        half_h = d * max(m['cy'], m['H'] - m['cy']) / m['f']
        kk = float(np.clip(2.05 * max(half_w, half_h), 0.7, 0.92 * d))
        eye = plane_c - V * kk + X * (half_w * 0.38) + np.array([0.0, 1.0, 0.0]) * (half_h * 0.22)
        route_u = STEMS['cards'].index(m['stem']) / max(NC - 1, 1)
        trackers.append({'slot': m['slot'], 'points_t0': m['points_t0'],
                         'points_t1': m['points_t1'], 'route_u': route_u,
                         'target': plane_c, 'eye': eye})
    # close-ups on pin heroes (kept for reference; camera uses trackers)
    closeups = []
    used_h = set()
    for p in pins[:4]:
        k = int(np.argmin([abs(s - (p - 0.10)) for s in slots]))
        if k in used_h: continue
        used_h.add(k)
        tr = trackers[k]
        closeups.append({'t0': frame_meta[k]['slot'] - 0.85,
                         't2': frame_meta[k]['points_t1'] + 0.75,
                         'target': tr['target'], 'eye': tr['eye']})
    card_t = card_start + card_span * np.arange(NC) / max(NC - 1, 1)
    return {'A': A_dur, 'frame_meta': frame_meta, 'pt_t0': pt_t0, 'pt_t1': pt_t1,
            'burst': burst, 'closeups': closeups, 'trackers': trackers,
            'card_t': card_t,
            'route': route_win, 'mesh': (A_dur - 1.55, A_dur - 0.45),
            'sentimental': sentimental}

# ---------------------------------------------------------------------------
# segment A
# ---------------------------------------------------------------------------
# v2 camera: ONE continuous journey along the photographer's route.
# A Catmull-Rom spline through the real camera positions; the journey
# parameter eases (smootherstep) between hero stops, arriving at each
# hero's own spot exactly as its photo lands. Near a hero, position and
# target blend to the 3/4 watch pose and push in gently during the
# crush. Finale: pull back off-route to the wide reveal.
_RP = np.vstack([ROUTE_POS[0], ROUTE_POS, ROUTE_POS[-1]])
def route_point(u):
    n = len(ROUTE_POS)
    x = clamp01(u) * (n - 1)
    i = min(int(x), n - 2); f = x - i
    p0, p1, p2, p3 = _RP[i], _RP[i + 1], _RP[i + 2], _RP[i + 3]
    f2, f3 = f * f, f * f * f
    return 0.5 * ((2 * p1) + (-p0 + p2) * f
                  + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * f3)

WIDE_POS = CENTRE + (RADIUS0 * 1.05) * np.array(
    [np.cos(0.40) * np.sin(1.10), np.sin(0.40), np.cos(0.40) * np.cos(1.10)])

def _journey_ctrl(sched):
    if 'jctrl' in sched:
        return sched['jctrl']
    trs = sched['trackers']
    m0, m1 = sched['mesh']
    us = [tr['route_u'] for tr in trs]
    ctrl = [(0.0, max(0.0, us[0] - 0.05))]
    for tr, u in zip(trs, us):
        ctrl.append((tr['slot'] + 0.10, u))     # arrive as the photo lands
    ctrl.append((m0 - 0.9, us[-1]))
    ctrl.append((m1 + 2.0, us[-1]))
    out = []
    for tt, u in ctrl:
        if out and tt <= out[-1][0]:
            if u > out[-1][1]:
                out[-1] = (out[-1][0], u)
            continue
        out.append((tt, u))
    sched['jctrl'] = out
    return out

def _journey_u(sched, t):
    ctrl = _journey_ctrl(sched)
    if t <= ctrl[0][0]:
        return ctrl[0][1]
    for (t0, u0), (t1, u1) in zip(ctrl, ctrl[1:]):
        if t <= t1:
            return u0 + (u1 - u0) * smoother(t0, t1, t)
    return ctrl[-1][1]

def _camera(sched, t):
    m0, m1 = sched['mesh']
    u = _journey_u(sched, t)
    eye = route_point(u)
    eye[1] = max(eye[1], FLOOR_Y + 0.55)
    ahead = route_point(min(u + 0.05, 1.0))
    tgt = 0.55 * ahead + 0.45 * CENTRE
    # watch envelopes: blend to each hero's 3/4 pose while its photo
    # lands and crushes; gentle push-in across the crush window
    release = 1 - smoother(m0 - 0.9, m0 - 0.1, t)
    w_sum = 0.0; eye_acc = np.zeros(3); tgt_acc = np.zeros(3)
    for tr in sched.get('trackers', []):
        w = min(smoother(tr['slot'] - 1.05, tr['slot'] - 0.10, t),
                1 - smoother(tr['points_t1'] + 0.10, tr['points_t1'] + 0.95, t)) * release
        if w > 0.001:
            crush = smooth(tr['points_t0'], tr['points_t1'], t)
            c_eye = tr['target'] + (tr['eye'] - tr['target']) * (1 - 0.16 * crush)
            c_eye = c_eye + (c_eye - tr['target']) \
                * (0.010 * np.sin((t - tr['slot']) * 1.8))
            eye_acc += w * c_eye; tgt_acc += w * tr['target']; w_sum += w
    w_max = 0.0
    if w_sum > 0.001:
        w_eff = clamp01(w_sum)
        eye = eye + (eye_acc / w_sum - eye) * w_eff
        tgt = tgt + (tgt_acc / w_sum - tgt) * w_eff
        w_max = w_eff
    # finale: pull back along/off the route to the wide reveal
    fin = smoother(m0 - 0.9, m1, t)
    if fin > 0.001:
        eye = eye + (WIDE_POS - eye) * fin
        tgt = tgt + (CENTRE - tgt) * fin
    fwd = tgt - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd]); tv = -R @ eye
    fy = (H / 2) / np.tan(0.48)
    K = np.array([[fy, 0, W / 2], [0, fy, H / 2], [0, 0, 1.0]])
    return {'K': K, 'R': R, 't': tv}, w_max

def _active_hero(sched, t):
    act = None
    for m in sched['frame_meta']:
        if m['slot'] <= t <= m['photo_out_end']: act = m
        elif m['slot'] > t: break
    return act

def _photo_opacity(m, t):
    tt = t - m['slot']
    if tt < 0 or t > m['photo_out_end']: return 0.0
    if tt < 0.10: return tt / 0.10
    db = m['dissolve_begin'] - m['slot']
    if tt <= db: return 1.0
    return max(0.0, 1 - (tt - db) / max(m['photo_out_end'] - m['dissolve_begin'], 1e-4))

def render_a_frame(sched, t, still=False):
    cam, w_cu = _camera(sched, (sched['A'] - 0.45) if still else t)
    A = sched['A']
    u_dim = 1.0 if still else smooth(sched['mesh'][0], sched['mesh'][1], t)
    u_mesh = 1.0 if still else smooth(sched['mesh'][0], sched['mesh'][1], t)
    # lines (v2): GLOBAL blue lattice filling the whole space (Detroit:
    # independent of the scene) + meaningless scattered shards + route.
    # The scene's own surface wireframe is NOT drawn.
    fade_in = 1.0 if still else (smooth(0, 4.2, t) if sched['sentimental'] else smooth(0, 2, t))
    scol = np.array([0.30, 0.55, 1.0]) * 255.0
    lat_a = fade_in * (1.0 - 0.25 * u_mesh)   # grid persists; dims slightly as surface closes
    lat_rgb = BG[None, :] + (scol[None, :] - BG[None, :]) * (0.55 * LAT_SHADE * lat_a)[:, None]
    sh_col = (np.array([0.24, 0.5, 1.0]) * 255.0) * (0.35 + 0.65 * SHARD_SHADE)[:, None]
    sh_a = (0.30 * fade_in) * (0.25 + 0.75 * SHARD_SHADE)[:, None]
    shard_rgb = BG[None, :] + (sh_col - BG[None, :]) * sh_a
    line_list = [LAT_LINES, SHARD_LINES]; line_cols = [lat_rgb, shard_rgb]
    r0, r1 = sched['route']
    u_route = 1.0 if still else clamp01((t - r0) / (r1 - r0))
    route_fade = 1.0 - 0.8 * u_dim
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
    # points
    chunks_p, chunks_c = [], []
    if still:
        chunks_p.append(PTS); chunks_c.append(RGB)
    else:
        t0, t1 = sched['pt_t0'], sched['pt_t1']
        act = t >= t0
        if act.any():
            p = np.clip((t - t0[act]) / np.maximum(t1[act] - t0[act], 1e-4), 0, 1)
            pos = START[act] + DELTA[act] * (p * p)[:, None] \
                + sched['burst'][act] * np.sin(np.pi * p ** 0.7)[:, None]
            al = np.clip(p / 0.15, 0, 1)[:, None]
            col = BG[None, :] + (RGB[act] - BG[None, :]) * al
            chunks_p.append(pos); chunks_c.append(col)
    # photo on the wall (one active hero)
    if not still:
        m = _active_hero(sched, t)
        if m is not None:
            op = _photo_opacity(m, t)
            if op > 0.02:
                sel = PHOTO_OWNER == m['hero_idx']
                chunks_p.append(PHOTO_PTS[sel])
                # true colours, edges feathered out to transparent
                # (black-gradient falloff), no frame
                chunks_c.append(BG[None, :] + (PHOTO_RGB[sel] - BG[None, :])
                                * (op * PHOTO_FEATHER)[:, None])
    # cards
    card_alpha = np.zeros(NC)
    if still:
        card_alpha[:] = 0.15
    else:
        fade_d = 0.9 if sched['sentimental'] else 0.6
        ap = np.array([smooth(sched['card_t'][i], sched['card_t'][i] + fade_d, t) for i in range(NC)])
        card_alpha = ap * (1 - 0.85 * u_dim)
        m = _active_hero(sched, t)
    keep = card_alpha > 0.03
    if keep.any():
        sel = keep[CARD_OWNER]
        cc = CARD_RGB[sel].copy()
        al = card_alpha[CARD_OWNER][sel][:, None]
        cc = BG[None, :] + (cc - BG[None, :]) * al
        if not still:
            m = _active_hero(sched, t)
            if m is not None:
                from json import loads
                stems = loads((WORK / 'stems.json').read_text())['cards']
                ai = stems.index(m['stem'])
                hot = CARD_OWNER[sel] == ai
                cc[hot] = np.clip(cc[hot] * 1.35, 0, 255)
        chunks_p.append(CARD_PTS[sel]); chunks_c.append(cc)
    # route nodes
    if u_route > 0:
        nsel = ROUTE_FRAC <= u_route
        if nsel.any():
            ncol = BG + (np.array([0.4, 0.7, 1.0]) * 255.0 - BG) * (0.75 * route_fade)
            cols = np.tile(ncol, (int(nsel.sum()), 1))
            idx = np.nonzero(nsel)[0]
            cols[-1] = BG + (np.array([0.85, 0.95, 1.0]) * 255.0 - BG) * route_fade
            chunks_p.append(ROUTE_POS[idx]); chunks_c.append(cols)
    # mesh points closing the surface
    if u_mesh > 0.02:
        chunks_p.append(MESH_POS)
        chunks_c.append(BG[None, :] + (MESH_COL - BG[None, :]) * (u_mesh * 0.92))
    if chunks_p:
        all_p = np.concatenate(chunks_p).astype(np.float32)
        all_c = np.clip(np.concatenate(chunks_c), 0, 255).astype(np.uint8)
    else:
        all_p = np.zeros((0, 3), np.float32); all_c = np.zeros((0, 3), np.uint8)
    radius = 2.4 * (1 + 0.5 * w_cu)
    img = render_frame(all_p, all_c, lines.astype(np.float32),
                       np.clip(lines_rgb, 0, 255).astype(np.uint8),
                       camera=cam, width=W, height=H, point_radius_px=radius,
                       glow_strength=0.5, glow_radius_factor=2.0,
                       background_rgb=(3, 6, 11))
    return img

# ---------------------------------------------------------------------------
# text helpers (PIL)
FONT_B = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
FONT_R = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
_font_cache = {}
def get_font(path, size):
    key = (path, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, size)
    return _font_cache[key]

def text_layer(text, size, color, tracking=0, bold=True, glow_color=(120, 180, 255),
               glow_radius=12, glow_alpha=0.85, sharp_alpha=1.0):
    """RGBA layer with tracked text + blurred glow underlay; returns (layer, (tw, th))."""
    font = get_font(FONT_B if bold else FONT_R, size)
    tmp = ImageDraw.Draw(Image.new('RGBA', (8, 8)))
    widths = [tmp.textlength(ch, font=font) for ch in text]
    asc, desc = font.getmetrics()
    tw = int(sum(widths) + tracking * (len(text) - 1)) + 8
    th = asc + desc + 8
    pad = glow_radius * 3
    layer = Image.new('RGBA', (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    x = pad + 4
    for ch, wch in zip(text, widths):
        dr.text((x, pad + 4), ch, font=font, fill=color + (255,))
        x += wch + tracking
    glow = layer.copy()
    if glow_alpha > 0:
        ga = glow.getchannel('A').point(lambda a: int(a * glow_alpha))
        glow = Image.merge('RGBA', (Image.new('L', glow.size, glow_color[0]),
                                     Image.new('L', glow.size, glow_color[1]),
                                     Image.new('L', glow.size, glow_color[2]), ga))
        glow = glow.filter(ImageFilter.GaussianBlur(glow_radius))
        out = Image.new('RGBA', layer.size, (0, 0, 0, 0))
        out.alpha_composite(glow); out.alpha_composite(glow)
        sa = layer.getchannel('A').point(lambda a: int(a * sharp_alpha))
        lay2 = Image.merge('RGBA', (layer.getchannel('R'), layer.getchannel('G'),
                                    layer.getchannel('B'), sa))
        out.alpha_composite(lay2)
        return out, (tw, th), pad
    return layer, (tw, th), pad

def paste_center(base, layer, cx, cy, alpha=1.0, scale=1.0):
    if alpha <= 0.003: return
    lay = layer
    if scale != 1.0:
        lay = lay.resize((max(1, int(lay.width * scale)), max(1, int(lay.height * scale))), Image.LANCZOS)
    if alpha < 0.999:
        a = lay.getchannel('A').point(lambda v: int(v * alpha))
        lay = lay.copy(); lay.putalpha(a)
    base.alpha_composite(lay, (int(cx - lay.width / 2), int(cy - lay.height / 2)))

_halo_cache = {}
def halo_sprite(color, radius, strength):
    """Radial-gradient halo RGBA of size 2*radius."""
    key = (color, radius)
    if key not in _halo_cache:
        r = radius
        yy, xx = np.mgrid[-r:r, -r:r]
        d = np.sqrt(xx ** 2 + yy ** 2) / r
        a = np.clip(1 - d, 0, 1) ** 2
        arr = np.zeros((2 * r, 2 * r, 4), np.uint8)
        arr[..., 0], arr[..., 1], arr[..., 2] = color
        arr[..., 3] = (a * 255).astype(np.uint8)
        _halo_cache[key] = Image.fromarray(arr, 'RGBA')
    sp = _halo_cache[key]
    if strength < 0.999:
        sp = sp.copy()
        sp.putalpha(sp.getchannel('A').point(lambda v: int(v * strength)))
    return sp
