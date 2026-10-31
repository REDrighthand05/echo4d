#!/usr/bin/env python3
"""Segment B: scene shrinks into a file icon, cursor drags it into the
Blender icon, closing line + credit. PIL port of /tmp/promo_work/build_b.py
canvas choreography; design space is 1920x1080 (native output size).
All motion eased (easeIO/easeOut/smoothstep), per user feedback."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from batch_lib import (W, H, clamp01, smooth, ease_io, ease_out, lerp,
                       text_layer, paste_center, halo_sprite, WORK)

FILE_C = (640, 470); BL_C = (1280, 470)
CREDIT = '本视频由 Muse 助手结合自研 Echo 渲染算子制作'
CLOSING = '完美流畅衔接 Blender、UE5 等开发流场景'

_logo = Image.open(WORK / 'blender_logo.png').convert('RGBA')

# ---- feathered scene still (rendered once by the driver) ----
_scene = None
def _load_scene():
    global _scene
    if _scene is None:
        img = Image.open(WORK / 'scene_still.png').convert('RGB')
        a = np.ones((H, W), np.float32)
        fx, fy = 110, 90
        ax = np.minimum(np.minimum(np.arange(W) / fx, (W - 1 - np.arange(W)) / fx), 1.0)
        ay = np.minimum(np.minimum(np.arange(H) / fy, (H - 1 - np.arange(H)) / fy), 1.0)
        alpha = (ax[None, :] * ay[:, None] * 255).astype(np.uint8)
        _scene = img.convert('RGBA'); _scene.putalpha(Image.fromarray(alpha))
    return _scene

# ---- file icon layer (supersampled) ----
SS = 3
def _file_icon_layer():
    w, h, fx, r = 62 * SS, 78 * SS, 30 * SS, 9 * SS
    pad = 90 * SS // 2
    lw, lh = 2 * w + 2 * pad, 2 * h + 2 * pad
    cx, cy = pad + w, pad + h
    # vertical gradient body
    top = np.array([252, 252, 253], np.float32); bot = np.array([204, 208, 214], np.float32)
    grad = (top[None, None, :] * (1 - np.linspace(0, 1, lh)[:, None, None])
            + bot[None, None, :] * np.linspace(0, 1, lh)[:, None, None])
    body = Image.fromarray(np.repeat(grad, lw, axis=1).astype(np.uint8), 'RGB').convert('RGBA')
    # outline mask with rounded corners (arc-sampled polygon)
    def arc(ccx, ccy, rr, a0, a1, n=10):
        return [(ccx + rr * np.cos(a), ccy + rr * np.sin(a))
                for a in np.linspace(a0, a1, n)]
    pts = []
    pts += [(cx - w + r, cy - h)]
    pts += [(cx + fx, cy - h), (cx + w, cy - 40 * SS)]
    pts += arc(cx + w - r, cy + h - r, r, 0, np.pi / 2)
    pts += arc(cx - w + r, cy + h - r, r, np.pi / 2, np.pi)
    pts += arc(cx - w + r, cy - h + r, r, np.pi, 3 * np.pi / 2)
    mask = Image.new('L', (lw, lh), 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    body.putalpha(mask)
    dr = ImageDraw.Draw(body)
    # dog-ear folds
    dr.polygon([(cx + fx, cy - h), (cx + fx, cy - 40 * SS), (cx + w, cy - 40 * SS)],
               fill=(170, 176, 185, 255))
    dr.polygon([(cx + fx, cy - h), (cx + fx, cy - 40 * SS), (cx + w, cy - h)],
               fill=(238, 240, 243, 255))
    # wireframe cube glyph
    c0, c1, o, yy = -18 * SS, 16 * SS, 11 * SS, 6 * SS
    col = (90, 110, 140, 217); lwln = max(2, int(2.4 * SS))
    def rect(x0, y0, x1, y1):
        dr.line([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], fill=col, width=lwln)
    rect(cx + c0, cy + yy - 14 * SS, cx + c1, cy + yy + 20 * SS)
    rect(cx + c0 + o, cy + yy - 14 * SS - o, cx + c1 + o, cy + yy + 20 * SS - o)
    for (ax_, ay_), (bx_, by_) in [
            ((c0, yy - 14 * SS), (c0 + o, yy - 14 * SS - o)),
            ((c1, yy - 14 * SS), (c1 + o, yy - 14 * SS - o)),
            ((c1, yy + 20 * SS), (c1 + o, yy + 20 * SS - o)),
            ((c0, yy + 20 * SS), (c0 + o, yy + 20 * SS - o))]:
        dr.line([(cx + ax_, cy + ay_), (cx + bx_, cy + by_)], fill=col, width=lwln)
    # content lines
    for ly in (44 * SS, 56 * SS):
        dr.rounded_rectangle([cx - w + 16 * SS, cy + ly, cx + w - 16 * SS, cy + ly + 5 * SS],
                             radius=2.5 * SS, fill=(120, 128, 140, 191))
    return body

_FILE_ICON = None
def file_icon():
    global _FILE_ICON
    if _FILE_ICON is None:
        _FILE_ICON = _file_icon_layer()
    return _FILE_ICON

def _cursor_layer():
    pad = 40
    pts = [(0, 0), (0, 34), (10, 27), (15.5, 42), (19.5, 40.5), (14.5, 26), (26, 26)]
    sc = 1.35 * SS
    lw = int(30 * sc) + 2 * pad; lh = int(46 * sc) + 2 * pad
    lay = Image.new('RGBA', (lw, lh), (0, 0, 0, 0))
    dr = ImageDraw.Draw(lay)
    pp = [(pad + x * sc, pad + y * sc) for x, y in pts]
    glow = Image.new('RGBA', (lw, lh), (0, 0, 0, 0))
    ImageDraw.Draw(glow).polygon(pp, fill=(255, 255, 255, 180))
    glow = glow.filter(ImageFilter.GaussianBlur(13 * SS / 2))
    lay.alpha_composite(glow)
    dr = ImageDraw.Draw(lay)
    dr.polygon(pp, fill=(255, 255, 255, 255), outline=(15, 18, 24, 230),
               width=max(2, int(1.6 * SS)))
    return lay

_CURSOR = None
def cursor_layer():
    global _CURSOR
    if _CURSOR is None:
        _CURSOR = _cursor_layer()
    return _CURSOR

def _paste(base, layer, cx, cy, alpha=1.0, scale=1.0):
    if alpha <= 0.003: return
    lay = layer
    if scale != 1.0:
        lay = lay.resize((max(1, int(lay.width * scale)), max(1, int(lay.height * scale))),
                         Image.LANCZOS)
    if alpha < 0.999:
        lay = lay.copy()
        lay.putalpha(lay.getchannel('A').point(lambda v: int(v * alpha)))
    base.alpha_composite(lay, (int(cx - lay.width / 2), int(cy - lay.height / 2)))

def _paste_tl(base, layer, x, y, alpha=1.0):
    if alpha <= 0.003: return
    lay = layer
    if alpha < 0.999:
        lay = lay.copy()
        lay.putalpha(lay.getchannel('A').point(lambda v: int(v * alpha)))
    base.alpha_composite(lay, (int(x), int(y)))

_text_cache = {}
def _text(text, size, color, tracking, bold, glow_alpha=0.85, glow_radius=12):
    key = (text, size, bold)
    if key not in _text_cache:
        _text_cache[key] = text_layer(text, size, color, tracking=tracking, bold=bold,
                                      glow_alpha=glow_alpha, glow_radius=glow_radius)
    return _text_cache[key]

def render_b_frame(T, B_dur, drop):
    k = B_dur / 9.34
    def S(x): return x * k
    img = Image.new('RGBA', (W, H), (0, 0, 0, 255))
    scene = _load_scene()

    # ---- scene shot: hold with a breathing glow, shrink into file icon ----
    sh_u = ease_io(smooth(S(2.1), S(3.3), T))
    sc_cx = lerp(960, FILE_C[0], sh_u); sc_cy = lerp(470, FILE_C[1], sh_u)
    sc_w = lerp(1180, 196, sh_u) * (1 + 0.015 * smooth(0, S(2.1), T))
    sc_a = (1 - 0.78 * smooth(S(2.55), S(3.3), T)) * smooth(0, S(0.5), T)
    sc_glow = 0.75 + 0.25 * np.sin(min(T / max(k, 1e-6), 2.1) * 1.6)
    if sc_a > 0.004 and T < S(3.45):
        _paste(img, halo_sprite((96, 156, 255), max(16, int(sc_w * 0.60) // 8 * 8), 1.0),
               sc_cx, sc_cy, alpha=0.22 * 0.8 * sc_glow * sc_a)
        sc_h = sc_w * H / W
        _paste(img, scene, sc_cx, sc_cy, alpha=sc_a, scale=sc_w / W)

    rec_u = smooth(drop + 0.81 * k, drop + 1.61 * k, T)
    grp_a = 1 - 0.15 * rec_u
    drag0 = drop - 0.89 * k
    grab = (8, -42)

    # ---- file icon ----
    drop_u = clamp01((T - drop) / (0.36 * k))
    fi_cx, fi_cy, fi_s = FILE_C[0], FILE_C[1], 1.0
    fi_a = smooth(S(2.55), S(3.3), T); fi_glow = 0.8
    drag_pos = None
    if drag0 <= T < drop:
        u = ease_io(clamp01((T - drag0) / (0.89 * k)))
        bx = (1 - u) ** 2 * FILE_C[0] + 2 * (1 - u) * u * 960 + u * u * BL_C[0]
        by = (1 - u) ** 2 * FILE_C[1] + 2 * (1 - u) * u * 338 + u * u * BL_C[1]
        drag_pos = (bx, by); fi_cx, fi_cy, fi_s, fi_glow = bx, by, 1.06, 1.15
    if T >= drop:
        fi_cx, fi_cy = BL_C
        fi_s = 1.06 * (1 - 0.76 * ease_out(drop_u)); fi_a *= 1 - drop_u
    if S(3.3) <= T < drag0: fi_glow = 0.95
    icon = file_icon()
    iscale = fi_s / SS
    # soft blue shadow-glow silhouette behind the icon
    sil = icon.copy()
    sil.putalpha(sil.getchannel('A').point(lambda v: int(v * 0.75 * fi_glow)))
    sil = sil.filter(ImageFilter.GaussianBlur(34 * fi_glow / 2))
    blue = Image.merge('RGBA', (Image.new('L', sil.size, 150), Image.new('L', sil.size, 195),
                                Image.new('L', sil.size, 255), sil.getchannel('A')))
    _paste(img, blue, fi_cx, fi_cy, alpha=fi_a * grp_a, scale=iscale)
    _paste(img, halo_sprite((96, 156, 255), 190, 1.0), fi_cx, fi_cy,
           alpha=0.22 * 0.55 * fi_glow * fi_a * grp_a, scale=fi_s)
    _paste(img, icon, fi_cx, fi_cy, alpha=fi_a * grp_a, scale=iscale)
    lay, _, _ = _text('voxel_model.glb', 29, (208, 230, 255), 1, False)
    paste_center(img, lay, FILE_C[0], FILE_C[1] + 128,
                 alpha=smooth(S(2.9), S(3.3), T) * (1 - smooth(drop, drop + 0.51 * k, T)) * grp_a)

    # ---- blender icon ----
    bl_cx = lerp(BL_C[0], 960, rec_u); bl_cy = lerp(BL_C[1], 316, rec_u)
    bl_s = lerp(1, 0.62, rec_u)
    pulse = np.sin(np.pi * clamp01((T - drop) / (0.75 * k))) if T >= drop else 0.0
    bl_s *= 1 + 0.10 * pulse
    bl_a = smooth(S(2.7), S(3.3), T) * grp_a
    _paste(img, halo_sprite((255, 160, 80), 185, 1.0), bl_cx, bl_cy,
           alpha=0.22 * 0.6 * (0.85 + 0.5 * pulse) * bl_a, scale=bl_s)
    lw_ = 168 * bl_s; lh_ = lw_ * _logo.height / _logo.width
    logo = _logo.resize((int(lw_), int(lh_)), Image.LANCZOS)
    _paste(img, logo, bl_cx, bl_cy, alpha=bl_a)
    lay, _, _ = _text('Blender', 29, (208, 230, 255), 2, False)
    paste_center(img, lay, BL_C[0], BL_C[1] + 128,
                 alpha=smooth(S(2.95), S(3.3), T) * (1 - rec_u) * grp_a)

    # ---- drop burst rings + flash ----
    if drop <= T < drop + 1.31 * k:
        bt = T - drop
        ov = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        for delay, r_max, a0 in ((0.0, 330, 0.85), (0.16 * k, 245, 0.6)):
            u = (bt - delay) / (1.05 * k)
            if u < 0 or u > 1: continue
            rr = 26 + r_max * ease_out(u)
            dr.ellipse([BL_C[0] - rr, BL_C[1] - rr, BL_C[0] + rr, BL_C[1] + rr],
                       outline=(150, 200, 255, int(255 * a0 * (1 - u))),
                       width=max(1, int(3.2 * (1 - u) + 0.6)))
        img.alpha_composite(ov.filter(ImageFilter.GaussianBlur(6)))
        img.alpha_composite(ov)
        fl = 1 - clamp01(bt / (0.5 * k))
        if fl > 0:
            _paste(img, halo_sprite((96, 156, 255), 330, 1.0), BL_C[0], BL_C[1],
                   alpha=0.22 * 1.4 * fl)
    # grab click ring
    if drag0 <= T < drag0 + 0.45 * k:
        u = (T - drag0) / (0.45 * k)
        ov = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        rr = 8 + 26 * u
        ImageDraw.Draw(ov).ellipse(
            [FILE_C[0] + grab[0] - rr, FILE_C[1] + grab[1] - rr,
             FILE_C[0] + grab[0] + rr, FILE_C[1] + grab[1] + rr],
            outline=(180, 215, 255, int(255 * 0.7 * (1 - u))), width=2)
        img.alpha_composite(ov)

    # ---- cursor ----
    cu_a, cu_x, cu_y = 0.0, 0.0, 0.0
    if S(3.3) <= T < drop + 0.91 * k:
        if T < drag0:
            u = ease_out(clamp01((T - S(3.3)) / (0.6 * k)))
            cu_x = lerp(1560, FILE_C[0] + grab[0], u); cu_y = lerp(940, FILE_C[1] + grab[1], u)
            cu_a = smooth(S(3.3), S(3.55), T)
        elif T < drop and drag_pos:
            cu_x, cu_y = drag_pos[0] + grab[0], drag_pos[1] + grab[1]; cu_a = 1.0
        else:
            u = ease_io(clamp01((T - drop - 0.11 * k) / (0.8 * k)))
            cu_x = lerp(BL_C[0] + grab[0], 1470, u); cu_y = lerp(BL_C[1] + grab[1], 660, u)
            cu_a = 1 - smooth(drop + 0.26 * k, drop + 0.91 * k, T)
    if cu_a > 0.003:
        cur = cursor_layer()
        _paste_tl(img, cur, cu_x - 40, cu_y - 40, alpha=cu_a)

    # ---- closing line, brand, credit ----
    tx_a = smooth(drop + 0.91 * k, drop + 1.37 * k, T)
    if tx_a > 0.004:
        breathe = 12 + 3 * np.sin(max(0.0, T - drop - 1.37 * k) * 1.4)
        lay, _, _ = _text(CLOSING, 57, (226, 240, 255), 2, True,
                          glow_alpha=0.95, glow_radius=int(breathe))
        paste_center(img, lay, 960, lerp(528, 548, tx_a), alpha=tx_a)
        paste_center(img, lay, 960, lerp(528, 548, tx_a), alpha=tx_a * 0.6)
    lay, _, _ = _text('ECHO 回声', 30, (208, 230, 255), 12, True)
    paste_center(img, lay, 960, 632,
                 alpha=0.9 * smooth(drop + 1.21 * k, drop + 1.81 * k, T))
    c0 = min(drop + 2.0 * k, B_dur - 1.7); c1 = min(drop + 2.7 * k, B_dur - 0.6)
    lay, _, _ = _text(CREDIT, 27, (190, 215, 250), 2, False,
                      glow_alpha=0.5, glow_radius=10)
    paste_center(img, lay, 960, 700, alpha=smooth(c0, c1, T))
    return np.asarray(img.convert('RGB'))
