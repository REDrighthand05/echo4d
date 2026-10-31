#!/usr/bin/env python3
"""Pioneer C -- live-preview feasibility prototype.

Question: can the FROZEN promo pipeline (promo/, Echo Render operator)
be driven INCREMENTALLY by a running job's intermediate products, fast
enough that a person watching sees the scene grow live?

Method: replay the room2 job's recorded intermediates (as baked into
promo/assets by the frozen prep chain: per-point frame provenance,
crush start positions + burst field, true-colour per-frame photos and
route cards, global Detroit lattice) as if they were arriving live, in
pipeline order:

  extract/select  -> frame cards appear in space
  poses           -> route draws stroke by stroke
  depth           -> per frame: photo lands on its wall, crushes into
                     points (approved math, untouched), points settle
  fuse            -> settled points swell solid (v3 layer), wide reveal

Every visible update is rendered at native 1280x720 with the operator
(render_frame) + the v3 solid composite, and timed: prep / operator /
composite / total.  The look reuses promo code as-is (batch_lib data +
lattice/shards/route treatment, batch_lib3 photo/card assets, crush
math byte-identical, swell pitch/duration identical).  Nothing is
retuned; this driver only re-sequences the frozen pieces into an
incremental, arrival-driven order with a live journey camera.

Usage:
  live_driver.py bench   -> cadence table -> cadence.json (+ stdout)
  live_driver.py video   -> simulated live-build demo video (silent)
"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
APP = HERE.parent.parent
PROMO = APP / 'promo'
sys.path.insert(0, str(PROMO))
sys.path.insert(0, str(APP))
import numpy as np

import batch_lib as BL
import batch_lib3 as BL3

# ---- live preview renders at native 720p (promo renders 1080p and
# downscales at encode; a live view has no encode step to hide in) ----
W, H = 1280, 720
BL3.W, BL3.H = W, H          # _cam_from / _solid_composite read these
BG = BL.BG
render_frame = BL.render_frame

PTS, START, RGB = BL.PTS, BL.START, BL.RGB
DELTA, BURST0 = BL.DELTA, BL.BURST0
FRAME_IDS, USED_IDS = BL.FRAME_IDS, BL.USED_IDS
ROUTE_POS, ROUTE_FRAC = BL.ROUTE_POS, BL.ROUTE_FRAC
LAT_LINES, LAT_SHADE = BL.LAT_LINES, BL.LAT_SHADE
SHARD_LINES, SHARD_SHADE = BL.SHARD_LINES, BL.SHARD_SHADE
CENTRE, FLOOR_Y = BL.CENTRE, BL.FLOOR_Y
NC = BL.NC                                # 230 selected frames / cards
NF = len(USED_IDS)                        # 229 frames that produced points

assert np.all(np.diff(USED_IDS) > 0), 'USED_IDS must be chronological'
PT_RANK = np.searchsorted(USED_IDS, FRAME_IDS).astype(np.int64)
_counts = np.bincount(PT_RANK, minlength=NF)
CUM = np.concatenate([[0], np.cumsum(_counts)])     # settled prefix sizes
ORDER = np.argsort(PT_RANK, kind='stable')          # points by arrival

# v3 true-colour assets (photo per used frame, card per selected frame)
PHOTO_PTS, PHOTO_RGB = BL3.PHOTO_PTS, BL3.PHOTO_RGB
PHOTO_OWNER, PHOTO_FEATHER = BL3.PHOTO_OWNER, BL3.PHOTO_FEATHER
PHOTO_DU, PHOTO_DV = BL3.PHOTO_DU, BL3.PHOTO_DV
CARD_PTS3, CARD_RGB3 = BL3.CARD_PTS, BL3.CARD_RGB
CARD_OWNER3, CARD_FEATHER3 = BL3.CARD_OWNER, BL3.CARD_FEATHER
PHOTO_OFFS = BL3.PHOTO_OFFS

# per-owner slices (owners are dense 0..N-1 in both assets)
def _slices(owner, n):
    o = np.argsort(owner, kind='stable')
    c = np.bincount(owner, minlength=n)
    return o, np.concatenate([[0], np.cumsum(c)])
PHOTO_ORD, PHOTO_CUM = _slices(PHOTO_OWNER, NF)
CARD_ORD, CARD_CUM = _slices(CARD_OWNER3, NC)

# static line colours (fade_in = 1: a live view is already "on")
_scol = np.array([0.30, 0.55, 1.0]) * 255.0
_sh_col = (np.array([0.24, 0.5, 1.0]) * 255.0) * (0.35 + 0.65 * SHARD_SHADE)[:, None]
_sh_a = 0.30 * (0.25 + 0.75 * SHARD_SHADE)[:, None]
SHARD_RGB = BG[None, :] + (_sh_col - BG[None, :]) * _sh_a
_route_col = np.clip(BG + np.array([0.35, 0.66, 1.0]) * 255.0 * 0.9, 0, 255)
_node_col = BG + (np.array([0.4, 0.7, 1.0]) * 255.0 - BG) * 0.75
_head_col = BG + (np.array([0.85, 0.95, 1.0]) * 255.0 - BG)


def _lat_rgb(close_u):
    lat_a = 1.0 - 0.25 * close_u
    return BG[None, :] + (_scol[None, :] - BG[None, :]) * (0.55 * LAT_SHADE * lat_a)[:, None]


def live_camera(u_cam, finale=0.0):
    """Eased journey camera (v2 route-journey treatment): eye rides the
    photographer's route at head height, target blends ahead + centre;
    finale pulls back to the wide reveal pose."""
    eye = BL.route_point(u_cam).copy()
    eye[1] = max(eye[1], FLOOR_Y + 0.55)
    ahead = BL.route_point(min(u_cam + 0.05, 1.0))
    tgt = 0.55 * ahead + 0.45 * CENTRE
    if finale > 0.0:
        f = BL.smoother(0.0, 1.0, finale)
        eye = eye + (BL.WIDE_POS - eye) * f
        tgt = tgt + (CENTRE - tgt) * f
    return BL3._cam_from(eye, tgt)


def render_live(cards_k=0, route_u=0.0, settled_k=0, flying_rank=None,
                flying_p=0.5, photo_rank=None, photo_op=0.0,
                settled_g=1.0, cam=None, finale=0.0,
                solid_stride=1, settled_as_points=False):
    """One visible live update. Returns (img, timings dict).

    cards_k      frame cards visible (extract phase)
    route_u      route drawn fraction 0..1 (poses phase)
    settled_k    frames fully arrived & settled (their points swell)
    flying_rank  frame currently crushing (its points mid-flight)
    photo_rank   frame whose photo is on its wall (opacity photo_op)
    settled_g    swell growth of settled points (scalar or per-point)
    """
    t_prep0 = time.perf_counter()
    if cam is None:
        cam = live_camera(0.0, finale)
    close_u = finale
    # ---- lines: global lattice + shards + route-so-far ----
    line_list = [LAT_LINES, SHARD_LINES]
    line_cols = [_lat_rgb(close_u), SHARD_RGB]
    if route_u > 0:
        segs, scols = [], []
        eff = _route_col * (1.0 - 0.8 * close_u)
        for i in range(NC - 1):
            f0, f1 = ROUTE_FRAC[i], ROUTE_FRAC[i + 1]
            if route_u <= f0:
                continue
            a, b = ROUTE_POS[i], ROUTE_POS[i + 1]
            if route_u < f1:
                b = a + (b - a) * ((route_u - f0) / max(f1 - f0, 1e-9))
            segs.append((a, b)); scols.append(eff)
        if segs:
            line_list.append(np.array(segs)); line_cols.append(np.array(scols))
    lines = np.concatenate(line_list)
    lines_rgb = np.concatenate(line_cols)
    chunks_p, chunks_c = [], []
    # ---- cards (extract phase): lit true-colour, feathered edges ----
    if cards_k > 0:
        idx = CARD_ORD[:CARD_CUM[cards_k]]
        fea = np.tile(CARD_FEATHER3, cards_k)   # same 48x84 layout per card
        chunks_p.append(CARD_PTS3[idx].astype(np.float64))
        chunks_c.append(BG[None, :] + (CARD_RGB3[idx].astype(np.float64) - BG[None, :])
                        * fea[:, None])
    # ---- route nodes ----
    if route_u > 0:
        nsel = ROUTE_FRAC <= route_u
        if nsel.any():
            cols = np.tile(_node_col * (1 - 0.8 * close_u) + BG * 0.8 * close_u,
                           (int(nsel.sum()), 1))
            cols[-1] = _head_col
            chunks_p.append(ROUTE_POS[nsel]); chunks_c.append(cols)
    # ---- hero photo on its wall (arriving frame) ----
    if photo_rank is not None and photo_op > 0.02:
        lo, hi = PHOTO_CUM[photo_rank], PHOTO_CUM[photo_rank + 1]
        idx = PHOTO_ORD[lo:hi]
        base = PHOTO_PTS[idx].astype(np.float64)
        du, dv = PHOTO_DU[photo_rank], PHOTO_DV[photo_rank]
        pp = np.concatenate([base + a * du + b * dv for a, b in PHOTO_OFFS])
        cc = BG[None, :] + (PHOTO_RGB[idx].astype(np.float64) - BG[None, :]) \
            * (photo_op * PHOTO_FEATHER)[:, None]
        chunks_p.append(pp); chunks_c.append(np.tile(cc, (len(PHOTO_OFFS), 1)))
    # ---- arriving frame's points mid-crush (approved math) ----
    if flying_rank is not None:
        sel = ORDER[CUM[flying_rank]:CUM[flying_rank + 1]]
        if len(sel):
            p = flying_p
            pos = START[sel] + DELTA[sel] * (p * p) \
                + BURST0[sel] * np.sin(np.pi * p ** 0.7)
            al = np.clip(p / 0.15, 0, 1)
            chunks_p.append(pos)
            chunks_c.append(BG[None, :] + (RGB[sel] - BG[None, :]) * al)
    # ---- settled points ----
    sol_parts = None
    if settled_k > 0:
        sidx = ORDER[:CUM[settled_k]]
        if settled_as_points:
            chunks_p.append(PTS[sidx]); chunks_c.append(RGB[sidx])
        else:
            if solid_stride > 1:
                sidx = sidx[::solid_stride]
            g = settled_g
            if np.ndim(g):
                g = g[sidx] if len(g) == len(PTS) else g
            else:
                g = np.full(len(sidx), float(g))
            sol_parts = (PTS[sidx], RGB[sidx], np.asarray(g, dtype=np.float64))
    if chunks_p:
        all_p = np.concatenate(chunks_p).astype(np.float32)
        all_c = np.clip(np.concatenate(chunks_c), 0, 255).astype(np.uint8)
    else:
        all_p = np.zeros((0, 3), np.float32); all_c = np.zeros((0, 3), np.uint8)
    t_prep = time.perf_counter() - t_prep0
    t_r0 = time.perf_counter()
    img = render_frame(all_p, all_c, lines.astype(np.float32),
                       np.clip(lines_rgb, 0, 255).astype(np.uint8),
                       camera=cam, width=W, height=H, point_radius_px=2.4,
                       glow_strength=0.5, glow_radius_factor=2.0,
                       background_rgb=(3, 6, 11))
    t_op = time.perf_counter() - t_r0
    t_comp = 0.0
    if sol_parts is not None:
        t_c0 = time.perf_counter()
        img = BL3._solid_composite(img, cam, all_p.astype(np.float64),
                                   sol_parts[0], sol_parts[1], sol_parts[2])
        t_comp = time.perf_counter() - t_c0
    return img, {'prep_ms': t_prep * 1e3, 'operator_ms': t_op * 1e3,
                 'composite_ms': t_comp * 1e3,
                 'total_ms': (t_prep + t_op + t_comp) * 1e3,
                 'n_points': int(len(all_p)) + (0 if sol_parts is None else int(len(sol_parts[0])))}


# ---------------------------------------------------------------------------
# bench: cadence of ONE visible update, per stage, at real data sizes
def _median_run(fn, reps=3):
    fn()                                   # warmup (njit already cached)
    ts = []
    for _ in range(reps):
        _, tm = fn()
        ts.append(tm)
    med = {k: float(np.median([t[k] for t in ts])) for k in ts[0]}
    med['reps_total_ms'] = [round(t['total_ms'], 1) for t in ts]
    return med


def _k_for_settled(target):
    """Arrival count whose settled prefix first reaches >= target points."""
    return int(np.searchsorted(CUM, target))


def bench():
    rows = []
    # cold start: the very first update in a fresh process pays njit
    # dispatch + buffer setup (kernels are disk-cached from the promo).
    t0 = time.perf_counter()
    render_live(cards_k=1, cam=live_camera(0.0))
    cold = (time.perf_counter() - t0) * 1e3
    print(f'cold first update: {cold:.0f} ms')

    def row(stage, note, **kw):
        cam = kw.pop('cam')
        m = _median_run(lambda: render_live(cam=cam, **kw))
        rows.append({'stage': stage, 'note': note, **m})
        print(f"{stage:28s} {note:44s} total {m['total_ms']:7.0f} ms "
              f"(prep {m['prep_ms']:.0f} / op {m['operator_ms']:.0f} / "
              f"solid {m['composite_ms']:.0f})  pts={m['n_points']} "
              f"reps={m['reps_total_ms']}")
        return m

    cam_start = live_camera(0.02)
    # extract: cards arriving (batches in a real job; cost shown per update)
    for k in (30, 120, NC):
        row('extract / cards', f'{k}/{NC} cards visible, base-res',
            cards_k=k, cam=cam_start)
    # poses: route drawing over bare nodes
    for u in (0.5, 1.0):
        row('poses / route', f'route fraction {u:.1f}, nodes only',
            route_u=u, cam=live_camera(u * 0.9))
    # depth: the heavy stage -- settled cloud growing, one frame crushing
    for target in (0, 25_000, 50_000, 100_000, 150_000, len(PTS)):
        k = min(_k_for_settled(target), NF - 1)
        row('depth / points landing',
            f'settled {int(CUM[k]):,}/208,666 pts (frame {k + 1}/{NF} crushing, g=1)',
            route_u=1.0, settled_k=k, flying_rank=k if k < NF else None,
            cam=live_camera(k / NF))
    # depth: photo of the arriving frame intact on its wall (heaviest)
    for target in (100_000, len(PTS)):
        k = min(_k_for_settled(target), NF - 1)
        row('depth / photo landing',
            f'settled {int(CUM[k]):,} pts + photo of frame {k + 1} intact (163k px)',
            route_u=1.0, settled_k=k, photo_rank=k, photo_op=1.0,
            cam=live_camera(k / NF))
    # fuse finale: everything settled + swollen, wide camera
    row('fuse / finale', 'all 208,666 pts settled+solid, wide view',
        route_u=1.0, settled_k=NF, cam=live_camera(1.0, finale=1.0), finale=1.0)
    # fallbacks at full size, in case the solid layer is the bottleneck
    k = NF - 1
    row('fallback / solid stride 2', 'settled solid layer at every 2nd point',
        route_u=1.0, settled_k=NF, cam=live_camera(1.0), solid_stride=2)
    row('fallback / solid stride 3', 'settled solid layer at every 3rd point',
        route_u=1.0, settled_k=NF, cam=live_camera(1.0), solid_stride=3)
    row('fallback / points only', 'settled drawn as operator points, no swell',
        route_u=1.0, settled_k=NF, cam=live_camera(1.0), settled_as_points=True)
    out = {'resolution': [W, H], 'cold_first_update_ms': cold, 'rows': rows,
           'points_total': int(len(PTS)), 'frames_used': int(NF), 'cards': int(NC)}
    (HERE / 'cadence.json').write_text(json.dumps(out, indent=1))
    print('wrote cadence.json')
    return out


# ---------------------------------------------------------------------------
# video: simulated live build, replayed at watchable pacing.
# NOTE: playback pacing is illustrative compression -- the measured
# per-update costs are in cadence.json / C_live_preview.md.
FPS = 30

def _overlay(img, lines):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, 'RGBA')
    f1 = BL.get_font(BL.FONT_B, 25); f2 = BL.get_font(BL.FONT_R, 21)
    dr.rectangle([14, H - 66, 560, H - 14], fill=(2, 6, 14, 150))
    dr.text((26, H - 60), lines[0], font=f1, fill=(190, 220, 255, 255))
    if len(lines) > 1:
        dr.text((26, H - 32), lines[1], font=f2, fill=(140, 175, 220, 255))
    return np.asarray(im)


def video():
    import subprocess
    out_path = HERE / 'live_preview_demo.mp4'
    cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
           '-an', '-c:v', 'libx264', '-profile:v', 'high', '-crf', '23',
           '-maxrate', '5M', '-bufsize', '2.5M', '-preset', 'medium',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    vt = 0.0            # video clock (s) -- drives swell growth
    settle_vt = np.full(NF, 1e18)   # video time each frame finished crushing

    def emit(img, hold_s, caption):
        nonlocal vt
        fr = _overlay(img, caption)
        n = max(1, int(round(hold_s * FPS)))
        for _ in range(n):
            proc.stdin.write(fr.tobytes())
        vt += hold_s

    def swell_g():
        # per-point growth from its frame's settle time (v3: 1.5 s swell)
        age = vt - settle_vt[PT_RANK]
        return BL3.smoother_v(settle_vt[PT_RANK], settle_vt[PT_RANK] + BL3.SWELL_DUR,
                              np.where(age >= 0, vt, settle_vt[PT_RANK]))

    # -- extract: cards arrive in batches of 10 --
    for k in range(10, NC + 1, 10):
        img, _ = render_live(cards_k=k, cam=live_camera(0.02 + 0.05 * k / NC))
        emit(img, 0.13, ['① 抽帧选帧 — 照片卡陆续到达', f'已选 {k} / {NC} 帧'])
    # -- poses: route draws itself --
    for i in range(1, 13):
        u = i / 12
        img, _ = render_live(route_u=u, cam=live_camera(0.05 + 0.3 * u))
        emit(img, 0.16, ['② 位姿 — 拍摄路线逐笔描出', f'路线 {int(u * 100)}%'])
    # -- depth: per frame, photo lands -> crushes -> settles --
    for r in range(NF):
        u = r / NF
        cam = live_camera(0.05 + 0.9 * u)
        g = swell_g()
        img, _ = render_live(route_u=1.0, settled_k=r, photo_rank=r,
                             photo_op=1.0, settled_g=g, cam=cam)
        emit(img, 0.085, ['③ 深度 — 每帧照片贴上它面对的墙',
                          f'第 {r + 1} / {NF} 帧 · 已落点 {int(CUM[r]):,}'])
        img, _ = render_live(route_u=1.0, settled_k=r, flying_rank=r,
                             flying_p=0.55, photo_rank=r, photo_op=0.3,
                             settled_g=g, cam=cam)
        settle_vt[r] = vt + 0.085
        emit(img, 0.085, ['③ 深度 — 照片在原地碎成点，落回真实位置',
                          f'第 {r + 1} / {NF} 帧 · 已落点 {int(CUM[r]):,}'])
    # -- fuse: swell completes, pull back wide --
    for i in range(7):
        f = i / 6
        img, _ = render_live(route_u=1.0, settled_k=NF, settled_g=swell_g(),
                             cam=live_camera(1.0, finale=f), finale=f)
        emit(img, 0.3, ['④ 融合 — 点膨大成实体，表面合拢',
                        '208,666 点全部落位'])
    img, _ = render_live(route_u=1.0, settled_k=NF, settled_g=1.0,
                         cam=live_camera(1.0, finale=1.0), finale=1.0)
    emit(img, 1.6, ['④ 融合完成 — 场景建成', '208,666 点 · 与正式成品同一份数据'])
    proc.stdin.close(); proc.wait()
    print(f'wrote {out_path}  (video clock {vt:.1f}s)')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'bench'
    if mode == 'bench':
        bench()
    elif mode == 'video':
        video()
    else:
        sys.exit(f'unknown mode {mode}')
