"""Frosted-glass skin for the desktop window (decoration only).

The look the user asked for (2026-11-02): the window's background
layer is frosted glass; behind the glass, on the side facing the
user, a few freehand graffiti strokes (the wavy blue "M" scribble
above all); every card on top is translucent too, so the layers
show through -- hazy, honest, layered, never flat opaque white.

Later the same day he asked for the graffiti to come alive: the
strokes are scattered into particles, and the strokes and the
particles alike are three-dimensional -- each stroke is a tube
winding through depth, each particle a small shaded bead whose
size, colour and dodge speed depend on how near it floats. While
the app sits idle ("in presence"), the beads respond to the mouse:
wherever the cursor goes, they scatter away from it, nearer ones
dodging wider than far ones, then drift home when it leaves.

Tk/CTk has no real per-widget alpha, so the glass is faked with
images, the way the app already fakes its other motions:

* one "scene" is painted per window size: a pale gradient, soft
  colour washes, and the graffiti strokes, all sharp;
* the window backdrop is that scene frosted (blur + a milk veil),
  or, while the particle animation runs, the live particle field
  frosted the same way;
* each card gets a crop of the scene taken from exactly where the
  card sits, frosted harder and milked more, its corners blended
  back into the backdrop crop so the card melts into the wall
  behind it while its middle stays a pane of glass.

Everything here is failure-contained by the caller: if any of this
raises, the window simply keeps its flat colours.
"""

import math

from PIL import Image, ImageDraw, ImageFilter

# Graffiti inks: the Muse blue first, the rest quiet companions.
INK_BLUE = (43, 107, 255)
INK_SOFT = (110, 134, 175)
INK_PALE = (150, 168, 198)
PALE_FAR = (198, 214, 238)   # bead colour washed toward this far away

SS = 2  # supersample for stroke smoothness


def _catmull_rom(points, samples=26):
    """Smooth a freehand polyline through its control points."""
    pts = [points[0]] + list(points) + [points[-1]]
    out = []
    for i in range(len(pts) - 3):
        p0, p1, p2, p3 = pts[i], pts[i + 1], pts[i + 2], pts[i + 3]
        for j in range(samples):
            t = j / samples
            t2, t3 = t * t, t * t * t
            out.append((
                0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t
                       + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                       + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3),
                0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t
                       + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                       + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3),
            ))
    out.append(points[-1])
    return out


def graffiti_strokes(w, h):
    """The graffiti as dense polylines in window pixels.

    One definition shared by the painted scene and the particle
    field, so the living beads form exactly the strokes the static
    glass shows. Each stroke: pts, width (px), color, alpha."""
    def P(fx, fy):
        return (fx * w, fy * h)

    strokes = []
    # The main stroke: a wavy freehand "M", Muse-glyph style, sitting
    # behind the header and the hero tile (upper-left to centre).
    strokes.append({
        "pts": _catmull_rom([P(0.055, 0.335), P(0.10, 0.155),
                             P(0.155, 0.315), P(0.215, 0.135),
                             P(0.275, 0.30), P(0.335, 0.15),
                             P(0.395, 0.315), P(0.45, 0.20),
                             P(0.485, 0.30)], samples=30),
        "width": 0.030 * w, "color": INK_BLUE, "alpha": 205})
    # A quick underline swoosh under it.
    strokes.append({
        "pts": _catmull_rom([P(0.07, 0.385), P(0.20, 0.425),
                             P(0.34, 0.40), P(0.46, 0.435)], samples=30),
        "width": 0.008 * w, "color": INK_BLUE, "alpha": 120})
    # A loose spiral doodle low on the left (behind history/depth).
    cx, cy = 0.155 * w, 0.815 * h
    spiral = []
    for i in range(90):
        t = i / 89.0
        ang = t * math.pi * 5.2
        r = (1 - t) * 0.052 * w + 2
        spiral.append((cx + r * math.cos(ang),
                       cy + r * math.sin(ang) * 0.82))
    strokes.append({"pts": spiral, "width": 0.009 * w,
                    "color": INK_SOFT, "alpha": 125})
    # A light scribble low on the right (behind the marker tile).
    strokes.append({
        "pts": _catmull_rom([P(0.60, 0.86), P(0.68, 0.775),
                             P(0.76, 0.865), P(0.84, 0.79),
                             P(0.90, 0.85)], samples=30),
        "width": 0.010 * w, "color": INK_PALE, "alpha": 110})
    # Two small companion marks: an open circle, a plus.
    ccx, ccy = 0.565 * w, 0.14 * h
    circle = [(ccx + 0.02 * w * math.cos(a),
               ccy + 0.025 * h * math.sin(a))
              for a in [i / 40.0 * 2 * math.pi for i in range(41)]]
    strokes.append({"pts": circle, "width": 0.007 * w,
                    "color": INK_SOFT, "alpha": 105})
    px, py = 0.905 * w, 0.115 * h
    arm = 0.016 * w
    for seg in ([(px - arm, py), (px + arm, py)],
                [(px, py - arm), (px, py + arm)]):
        strokes.append({"pts": seg, "width": 0.007 * w,
                        "color": INK_SOFT, "alpha": 105})
    # Grow the composition (user: the background graffiti is too
    # small). Groups scale about their own centres, widths along.
    grow = [(0.27, 0.30, 1.42), (0.27, 0.30, 1.42),
            (0.155, 0.815, 1.35), (0.75, 0.83, 1.30),
            (0.565, 0.14, 1.35), (0.905, 0.115, 1.30),
            (0.905, 0.115, 1.30)]
    for st, (fx, fy, k) in zip(strokes, grow):
        cx, cy = fx * w, fy * h
        st["pts"] = [(cx + (x - cx) * k, cy + (y - cy) * k)
                     for x, y in st["pts"]]
        st["width"] *= k
    return strokes


def _base_scene(w, h):
    """Gradient + colour washes: the glass body, no graffiti."""
    grad = Image.new("RGB", (1, h))
    top = (240, 243, 247)
    bottom = (219, 226, 235)
    for y in range(h):
        t = y / max(1, h - 1)
        grad.putpixel((0, y), tuple(
            round(top[i] + (bottom[i] - top[i]) * t) for i in range(3)))
    scene = grad.resize((w, h)).convert("RGBA")

    wash = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    wd = ImageDraw.Draw(wash)
    wd.ellipse([-0.15 * w, -0.25 * h, 0.55 * w, 0.45 * h],
               fill=(168, 196, 240, 90))
    wd.ellipse([0.55 * w, 0.55 * h, 1.2 * w, 1.25 * h],
               fill=(196, 214, 238, 80))
    wd.ellipse([0.05 * w, 0.62 * h, 0.5 * w, 1.15 * h],
               fill=(226, 220, 205, 60))
    scene.alpha_composite(wash.filter(ImageFilter.GaussianBlur(70)))
    return scene


def build_scene(w, h):
    """The artwork behind the glass, at window size (sharp)."""
    scene = _base_scene(w, h)
    # Graffiti layer, supersampled so the strokes stay silky.
    g = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    gd = ImageDraw.Draw(g)
    for st in graffiti_strokes(w, h):
        gd.line([(x * SS, y * SS) for x, y in st["pts"]],
                fill=st["color"] + (st["alpha"],),
                width=max(1, int(st["width"] * SS)), joint="curve")
    g = g.resize((w, h), Image.LANCZOS).filter(
        ImageFilter.GaussianBlur(0.7))
    scene.alpha_composite(g)
    return scene.convert("RGB")


def frost_background(scene):
    """The window backdrop: the scene behind frosted glass."""
    img = scene.filter(ImageFilter.GaussianBlur(11))
    veil = Image.new("RGB", img.size, (248, 250, 252))
    return Image.blend(img, veil, 0.24)


def panel_image(bg, scene, box, radius=24, milk=104, blur=19, tint=None):
    """One card's glass pane: the scene crop from where the card
    sits, frosted harder, milked toward white, corners blended back
    into the backdrop so only the pane's middle reads as glass."""
    x, y, w, h = box
    if w < 8 or h < 8:
        return None
    pane = scene.crop((x, y, x + w, y + h)).filter(
        ImageFilter.GaussianBlur(blur))
    pane = Image.blend(pane, Image.new("RGB", (w, h), (252, 253, 255)),
                       milk / 255.0)
    if tint is not None:
        tr, tg, tb = tint
        pane = Image.blend(pane, Image.new("RGB", (w, h), (tr, tg, tb)),
                           0.38)
    mask = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(8))  # feathered rim: the frost fades into the wall, no edge line
    under = bg.crop((x, y, x + w, y + h))
    return Image.composite(pane, under, mask)


def _hex_rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


# ------------------------------------------------------------------
# The living graffiti: strokes scattered into three-dimensional
# beads that idle-drift and dodge the mouse.
# ------------------------------------------------------------------

class ParticleField:
    """The graffiti strokes as a shallow 3D volume of beads.

    Home positions trace the strokes; each home also carries a
    depth z in [-1, 1]: the stroke is a tube (its centre floats
    nearer than its edges) and the tube itself winds in depth along
    its length, so the scribble is a ribbon in space, not a flat
    decal. A bead renders as a small shaded sphere -- size, colour
    saturation and alpha all follow its current depth, and the
    mouse pushes nearer beads wider than far ones (parallax), with
    a depth kick on top, so a dodge reads as scattering *in space*.
    Springs pull everything home; a slow sway keeps the volume
    breathing while nobody touches it.
    """

    MOUSE_R = 140.0        # cursor repulsion radius, px
    SPRING = 30.0          # home spring, 1/s^2
    DAMP = 9.3             # velocity damping, 1/s
    Z_SPRING = 14.0
    Z_DAMP = 6.0

    def __init__(self, w, h, seed=20261102, calm_zones=None):
        import numpy as np

        self.np = np
        self.w, self.h = w, h
        self.calm_zones = calm_zones or []
        self.base = _base_scene(w, h).convert("RGB")
        rng = np.random.default_rng(seed)

        homes, hzs, rads, cols, alps = [], [], [], [], []
        phases, zdirs, sdirs = [], [], []
        for si, st in enumerate(graffiti_strokes(w, h)):
            pts = np.asarray(st["pts"], dtype=np.float64)
            if len(pts) < 2:
                continue
            diffs = np.diff(pts, axis=0)
            seglen = np.hypot(diffs[:, 0], diffs[:, 1])
            cum = np.concatenate([[0.0], np.cumsum(seglen)])
            total = float(cum[-1])
            if total < 4:
                continue
            spacing = 11.5 if st["width"] >= 0.02 * w else 10.0
            count = max(6, int(total / spacing))
            s_vals = (np.arange(count) + rng.random(count)) * (total / count)
            idx = np.clip(np.searchsorted(cum, s_vals), 1, len(pts) - 1)
            frac = (s_vals - cum[idx - 1]) / np.maximum(seglen[idx - 1], 1e-9)
            pos = pts[idx - 1] + (pts[idx] - pts[idx - 1]) * frac[:, None]
            tang = diffs[idx - 1]
            tang /= np.maximum(np.hypot(tang[:, 0], tang[:, 1]), 1e-9)[:, None]
            normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
            # Lateral offset across the stroke's body, centre-weighted:
            # the middle of the tube carries most beads.
            u = (rng.random(count) + rng.random(count)
                 + rng.random(count)) / 3.0 * 2.0 - 1.0
            home = pos + normal * (u * (st["width"] * 0.5 + 9.0))[:, None]
            # Depth: tube roundness across the width + the tube winding
            # in depth along its own length + a breath of jitter.
            z = (0.45 * np.cos(u * math.pi / 2.0)
                 + 0.38 * np.sin(s_vals * 0.010 + si * 1.7)
                 + rng.normal(0.0, 0.30, count))
            homes.append(home)
            hzs.append(np.clip(z, -1.0, 1.0))
            rads.append(np.full(count, max(3.4, st["width"] * 0.27))
                        * rng.uniform(0.8, 1.25, count))
            cols.append(np.tile(np.asarray(st["color"], dtype=np.float64),
                                (count, 1)))
            alps.append(np.full(count, float(st["alpha"])))
            phases.append(rng.uniform(0.0, 2 * math.pi, count))
            zu = rng.uniform(-1.0, 1.0, count)
            zdirs.append(np.sign(zu) * (0.55 + 0.45 * np.abs(zu)))
            sang = rng.uniform(0.0, 2 * math.pi, count)
            sdirs.append(np.stack([np.cos(sang), np.sin(sang)], axis=1))

        npcat = np.concatenate
        home_all = npcat(homes).astype(np.float64)
        # Calm zones (text blocks): no bead lives inside one. A home
        # that lands in a zone is pushed out through its nearest
        # edge, so the cloud parts around the words and the glass
        # behind every line of text stays calm and even -- a label's
        # flat background then has nothing to clash with.
        for (zx0, zy0, zx1, zy1) in self.calm_zones:
            inside = ((home_all[:, 0] > zx0) & (home_all[:, 0] < zx1)
                      & (home_all[:, 1] > zy0) & (home_all[:, 1] < zy1))
            if not inside.any():
                continue
            pts = home_all[inside]
            pens = np.stack([pts[:, 0] - zx0, zx1 - pts[:, 0],
                             pts[:, 1] - zy0, zy1 - pts[:, 1]], axis=1)
            side = np.argmin(pens, axis=1)
            out = pts.copy()
            out[side == 0, 0] = zx0 - 5
            out[side == 1, 0] = zx1 + 5
            out[side == 2, 1] = zy0 - 5
            out[side == 3, 1] = zy1 + 5
            home_all[inside] = out
        self.home = home_all
        self.hz = npcat(hzs)
        self.rad = npcat(rads)
        self.col = npcat(cols)
        self.alp = npcat(alps)
        self.phase = npcat(phases)
        self.zdir = npcat(zdirs)
        self.sdir = npcat(sdirs)
        self.pos = self.home.copy()
        self.vel = np.zeros_like(self.home)
        self.z = self.hz.copy()
        self.vz = np.zeros_like(self.hz)
        self.count = len(self.hz)
        self.t = 0.0
        self._sprites = {}
        self._small_base = None

    # -- physics ---------------------------------------------------

    def step(self, dt, mouse=None):
        """Advance one tick. `mouse` is (x, y) in window pixels or
        None when the cursor is elsewhere."""
        np = self.np
        dt = min(max(dt, 0.001), 0.05)
        self.t += dt
        t = self.t
        hx = self.home[:, 0] + 3.0 * np.sin(t * 0.50 + self.phase)
        hy = self.home[:, 1] + 2.1 * np.sin(t * 0.37 + self.phase * 1.7)
        hzt = self.hz + 0.12 * np.sin(t * 0.45 + self.phase * 2.3)

        ax = (hx - self.pos[:, 0]) * self.SPRING - self.vel[:, 0] * self.DAMP
        ay = (hy - self.pos[:, 1]) * self.SPRING - self.vel[:, 1] * self.DAMP
        az = (hzt - self.z) * self.Z_SPRING - self.vz * self.Z_DAMP

        if mouse is not None:
            dx = self.pos[:, 0] - mouse[0]
            dy = self.pos[:, 1] - mouse[1]
            d2 = dx * dx + dy * dy
            near = d2 < self.MOUSE_R * self.MOUSE_R
            if near.any():
                d = np.sqrt(np.maximum(d2[near], 1.0))
                f = (1.0 - d / self.MOUSE_R) ** 2
                zn = (self.z[near] + 1.0) / 2.0
                acc = f * 4200.0 * (0.5 + 0.8 * zn)   # near dodges wider
                self.vel[near, 0] += dx[near] / d * acc * dt
                self.vel[near, 1] += dy[near] / d * acc * dt
                self.vel[near] += self.sdir[near] * (f * 950.0 * dt)[:, None]
                # ...and bursts through depth, each bead decisively
                # toward the viewer or away into the glass.
                self.vz[near] += f * self.zdir[near] * 62.0 * dt

        self.vel[:, 0] += ax * dt
        self.vel[:, 1] += ay * dt
        self.vz += az * dt
        speed = np.hypot(self.vel[:, 0], self.vel[:, 1])
        over = speed > 950.0
        if over.any():
            self.vel[over] *= (950.0 / speed[over])[:, None]
        self.pos += self.vel * dt
        self.z = np.clip(self.z + self.vz * dt, -1.3, 1.3)

    # -- rendering ---------------------------------------------------

    def _sprite(self, r, zn, color, alpha):
        """One bead: a small sphere -- shaded belly, bright crown."""
        key = (round(r * 2), round(zn * 10), color,
               int(alpha) // 16)
        spr = self._sprites.get(key)
        if spr is not None:
            return spr
        if len(self._sprites) > 600:
            self._sprites.clear()
        mix = (1.0 - zn) * 0.5
        c = tuple(int(round(color[i] * (1 - mix) + PALE_FAR[i] * mix))
                  for i in range(3))
        shade = 0.80 + 0.26 * zn
        c = tuple(min(255, int(round(v * shade))) for v in c)
        dark = tuple(int(v * 0.55) for v in c)
        a = max(0, min(255, int(alpha)))
        rad = max(1.2, r)
        size = int(math.ceil(rad)) + 2
        spr = Image.new("RGBA", (size * 2, size * 2), (0, 0, 0, 0))
        d = ImageDraw.Draw(spr)
        ctr = size
        d.ellipse([ctr - rad, ctr - rad, ctr + rad, ctr + rad],
                  fill=c + (a,))
        d.ellipse([ctr - rad * 0.98, ctr - rad * 0.45,
                   ctr + rad * 0.98, ctr + rad * 0.98],
                  fill=dark + (int(a * 0.55),))
        d.ellipse([ctr - rad * 0.60, ctr - rad * 0.70,
                   ctr - rad * 0.02, ctr - rad * 0.16],
                  fill=(255, 255, 255, int(a * 0.95)))
        spr = spr.filter(ImageFilter.GaussianBlur(0.6))
        self._sprites[key] = spr
        return spr

    def render(self):
        """The field, frosted the same way as the static backdrop.
        Drawn at 0.7 scale and lifted back up: under this much
        frost the difference is invisible and the tick stays cheap."""
        np = self.np
        sc = 0.7
        if self._small_base is None:
            self._small_base = self.base.resize(
                (int(self.w * sc), int(self.h * sc)), Image.BILINEAR)
        img = self._small_base.copy()
        order = np.argsort(self.z)
        xs, ys, zs = self.pos[:, 0], self.pos[:, 1], self.z
        for i in order:
            zn = float(min(1.0, max(0.0, (zs[i] + 1.0) / 2.0)))
            r = float(self.rad[i]) * (0.5 + 1.05 * zn) * sc
            alpha = float(self.alp[i]) * (0.38 + 0.68 * zn)
            spr = self._sprite(r, zn, tuple(int(v) for v in self.col[i]),
                               alpha)
            img.paste(spr, (int(xs[i] * sc - spr.width / 2),
                            int(ys[i] * sc - spr.height / 2)), spr)
        img = img.filter(ImageFilter.GaussianBlur(4))
        img = img.resize((self.w, self.h), Image.BICUBIC)
        veil = Image.new("RGB", img.size, (248, 250, 252))
        return Image.blend(img, veil, 0.22)


class GlassWall:
    """Owns the frosted backdrop and every card pane hung on it.

    Static mode paints the graffiti once and frosts it. Animated
    mode (start_animation) runs the ParticleField behind the glass
    while the app is idle and refreshes the panes from the live
    frame, a couple per tick so nothing ever hitches.
    """

    def __init__(self, root):
        import tkinter as tk

        self.root = root
        self.panels = []      # dicts: frame, label, name, milk, blur, tint
        self.labels = {}      # name -> tk label (drop targets etc.)
        self._scene = None
        self._bg = None
        self._size = None
        self._tk = tk
        # The backdrop itself: one label under everything, so the
        # gaps between cards are glass too, not flat colour.
        self._backdrop = tk.Label(root, bd=0, highlightthickness=0)
        self._backdrop.place(x=0, y=0, relwidth=1, relheight=1)
        self._backdrop_img = None
        # Animation state.
        self._anim = False
        self._is_busy = None
        self._field = None
        self._field_size = None
        self._size_marks = 0
        self._fail = 0
        self._tick_n = 0
        self._pane_cursor = 0
        self._last_frame = None
        self._last_tick = None
        self._text_widgets = []
        self._wall_img = None
        self._grout = []
        self._textified = []

    def attach(self, frame, name=None, milk=104, blur=19, radius=24):
        """Hang a glass pane inside `frame`. Call BEFORE the frame's
        content is packed, so the content stacks above the pane."""
        label = self._tk.Label(frame, bd=0, highlightthickness=0)
        label.place(x=0, y=0, relwidth=1, relheight=1)
        panel = {"frame": frame, "label": label, "name": name,
                 "milk": milk, "blur": blur, "radius": radius,
                 "tint": None, "img": None}
        self.panels.append(panel)
        if name:
            self.labels[name] = label
        return label

    def tint(self, name, color):
        """Tint one pane (drag-over highlight); color None clears.
        While animating, the pane picks the tint up on its next
        staggered refresh -- no full repaint needed."""
        for panel in self.panels:
            if panel["name"] == name:
                panel["tint"] = _hex_rgb(color) if color else None
        if not self._anim:
            self.refresh()

    def _abs_box(self, frame):
        x = y = 0
        node = frame
        while node is not None and node is not self.root:
            x += node.winfo_x()
            y += node.winfo_y()
            node = node.master
        return x, y, frame.winfo_width(), frame.winfo_height()

    def _set_pane(self, panel, img):
        from PIL import ImageTk

        panel["last_img"] = img
        panel["last_box"] = self._abs_box(panel["frame"])
        photo = ImageTk.PhotoImage(img)
        panel["img"] = photo
        panel["label"].configure(image=photo)
        # CTk children fake their transparency from the frame's
        # fg_color: match it to the pane's average tone so text
        # sits on the glass instead of on white boxes.
        try:
            r, g, b = img.resize((1, 1)).getpixel((0, 0))[:3]
            panel["frame"].configure(fg_color=f"#{r:02x}{g:02x}{b:02x}")
        except Exception:
            pass

    def refresh(self):
        try:
            self._refresh()
        except Exception:
            pass  # decoration only: never take the window down

    def _refresh(self):
        from PIL import ImageTk

        if self._anim:
            # The animation loop owns sizes and panes; a configure
            # storm just re-crops the panes from the latest frame.
            if self._last_frame is not None:
                for panel in self.panels:
                    self._update_pane(panel, self._last_frame)
                self._rebuild_wall(self._last_frame)
            return
        w, h = self.root.winfo_width(), self.root.winfo_height()
        if w < 60 or h < 60:
            return
        if (w, h) != self._size:
            self._scene = build_scene(w, h)
            self._bg = frost_background(self._scene)
            self._size = (w, h)
            photo = ImageTk.PhotoImage(self._bg)
            self._backdrop_img = photo
            self._backdrop.configure(image=photo)
        for panel in self.panels:
            img = panel_image(self._bg, self._scene,
                              self._abs_box(panel["frame"]),
                              radius=panel["radius"], milk=panel["milk"],
                              blur=panel["blur"], tint=panel["tint"])
            if img is not None:
                self._set_pane(panel, img)
        self._rebuild_wall(self._bg)

    # -- the living backdrop -----------------------------------------

    def start_animation(self, is_busy=None):
        """Run the particle graffiti behind the glass. `is_busy`
        is a callable returning True while a job owns the machine;
        the field then freezes and costs nothing."""
        if self._anim:
            return
        self._anim = True
        self._is_busy = is_busy
        try:
            self._backdrop.lower()
        except Exception:
            pass
        self.root.after(80, self._tick)

    def _tick(self):
        if not self._anim:
            return
        try:
            self._tick_inner()
            self._fail = 0
        except Exception:
            self._fail += 1
            if self._fail >= 6:
                # Fall back to the still glass and stay there.
                self._anim = False
                self._field = None
                self.refresh()
                return
        self.root.after(40, self._tick)

    def _tick_inner(self):
        from PIL import ImageTk

        w, h = self.root.winfo_width(), self.root.winfo_height()
        if w < 200 or h < 150:
            return
        if (w, h) != self._field_size:
            # Rebuild only once the size settles, so a drag-resize
            # does not thrash scene painting.
            self._size_marks += 1
            if self._field is None or self._size_marks >= 3:
                zones = []
                for widget in self._text_widgets + self._textified:
                    try:
                        bx, by, bw, bh = self._abs_box(widget)
                    except Exception:
                        continue
                    if bw >= 4 and bh >= 4:
                        zones.append((bx - 12, by - 10,
                                      bx + bw + 12, by + bh + 10))
                self._field = ParticleField(w, h, calm_zones=zones)
                self._field_size = (w, h)
                self._size_marks = 0
                self._scene = build_scene(w, h)
                self._bg = frost_background(self._scene)
                self._size = (w, h)
                for panel in self.panels:
                    img = panel_image(self._bg, self._scene,
                                      self._abs_box(panel["frame"]),
                                      radius=panel["radius"],
                                      milk=panel["milk"],
                                      blur=panel["blur"],
                                      tint=panel["tint"])
                    if img is not None:
                        self._set_pane(panel, img)
            return
        busy = False
        if self._is_busy is not None:
            try:
                busy = bool(self._is_busy())
            except Exception:
                busy = False
        if busy:
            self._last_tick = None
            return  # frozen mid-gesture; the CPU belongs to the job
        mouse = None
        try:
            mx = self.root.winfo_pointerx() - self.root.winfo_rootx()
            my = self.root.winfo_pointery() - self.root.winfo_rooty()
            if -30 <= mx <= w + 30 and -30 <= my <= h + 30:
                mouse = (mx, my)
        except Exception:
            mouse = None
        import time

        now = time.monotonic()
        dt = 0.05 if self._last_tick is None else min(
            max(now - self._last_tick, 0.02), 0.09)
        self._last_tick = now
        self._field.step(dt, mouse)
        frame = self._field.render()
        self._last_frame = frame
        photo = ImageTk.PhotoImage(frame)
        self._backdrop_img = photo
        self._backdrop.configure(image=photo)
        # Panes follow the living wall, two per tick round-robin:
        # the frost barely moves at this cadence, and no single
        # tick ever pays for every card at once.
        self._tick_n += 1
        for _ in range(2):
            if not self.panels:
                break
            panel = self.panels[self._pane_cursor % len(self.panels)]
            self._pane_cursor += 1
            self._update_pane(panel, frame)
        if self._tick_n % 4 == 1:
            self._rebuild_wall(frame)

    def textify(self, label):
        """Make a label's background be the glass itself: on every
        wall rebuild the label is given the wall's own pixels at
        exactly where it sits, as its image, with its glyphs drawn
        over them. Outside the glyphs you see the wall -- a text
        background cannot exist."""
        self._textified.append(label)
        return label

    def track_grout(self, frames):
        """Container frames whose flat colour fills the gaps
        between panes; it is kept equal to the wall's tone."""
        self._grout = list(frames)

    def track_labels(self, widgets):
        """Register text widgets that sit on glass panes. Their
        backgrounds are matched to the glass directly beneath them,
        so no text ever sits on a visible box."""
        self._text_widgets.extend(widgets)

    def _rebuild_wall(self, base):
        """Composite backdrop + panes into one wall image and
        re-match every tracked text background against it."""
        if base is None:
            return
        try:
            wall = base.copy()
            for panel in self.panels:
                img = panel.get("last_img")
                box = panel.get("last_box")
                if img is not None and box:
                    wall.paste(img, (box[0], box[1]))
            self._wall_img = wall
            self._match_labels()
            self._paint_textified(wall)
            # The gaps between panes are part of the same sheet:
            # their flat colour tracks the wall's own tone, so the
            # seams read as joints in one glass, not grout lines.
            try:
                r, g, b = wall.resize((1, 1)).getpixel((0, 0))[:3]
                hexv = f"#{r:02x}{g:02x}{b:02x}"
                for fr in self._grout:
                    try:
                        fr.configure(fg_color=hexv)
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass

    def _paint_textified(self, wall):
        import customtkinter as ctk
        from PIL import ImageTk  # noqa: F401  (kept for parity)

        for label in self._textified:
            try:
                x, y, w, h = self._abs_box(label)
                if w < 4 or h < 4:
                    continue
                crop = wall.crop((max(0, x), max(0, y),
                                  min(wall.width, x + w),
                                  min(wall.height, y + h)))
                if crop.size != (w, h):
                    continue
                img = ctk.CTkImage(crop, size=(w, h))
                label.configure(image=img, compound="center")
                label._glass_img = img
            except Exception:
                pass

    def _match_labels(self):
        if self._wall_img is None:
            return
        W, H = self._wall_img.size
        for widget in self._text_widgets:
            try:
                x, y, w, h = self._abs_box(widget)
                if w < 4 or h < 4:
                    continue
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(W, x + w), min(H, y + h)
                if x1 - x0 < 2 or y1 - y0 < 2:
                    continue
                crop = self._wall_img.crop((x0, y0, x1, y1))
                r, g, b = crop.resize((1, 1)).getpixel((0, 0))[:3]
                hexv = f"#{r:02x}{g:02x}{b:02x}"
                try:
                    widget.configure(fg_color=hexv)
                except Exception:
                    widget.configure(bg=hexv)
            except Exception:
                pass

    def _update_pane(self, panel, frame):
        img = panel_image(frame, frame, self._abs_box(panel["frame"]),
                          radius=panel["radius"], milk=panel["milk"],
                          blur=7, tint=panel["tint"])
        if img is not None:
            self._set_pane(panel, img)
