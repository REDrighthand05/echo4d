"""Icons in the glass-bead style of the window skin.

Every pictogram speaks the same language as the frosted wall and
the living graffiti (user, 2026-11-02: all icons coordinated):
ink-blue shapes with pearl shading -- a lighter crown, a shaded
belly, a soft glow underneath -- hazy at the edges, never flat
clip-art. Drawn with PIL at supersample, so the app needs no icon
font and no emoji set (whose look would clash with the glass).

Also builds the application icon (installer/echo.ico): a frosted
pale rounded square carrying the graffiti's main stroke as a
string of pearls -- the same beads, standing still.
"""

import math

from PIL import Image, ImageDraw, ImageFilter

INK_BLUE = (43, 107, 255)
INK_SOFT = (110, 134, 175)
INK_PALE = (150, 168, 198)
PALE_FILL = (184, 203, 235)
DARK_MAIN = (127, 178, 255)     # on the preview card's dark ground
DARK_SOFT = (96, 128, 175)

SS = 3


def _shade(color, f):
    return tuple(max(0, min(255, int(round(v * f)))) for v in color)


def bead(d, cx, cy, r, color, alpha=255, crown=0.9):
    """One pearl: shaded belly, bright crown -- the graffiti's bead."""
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color + (alpha,))
    d.ellipse([cx - r * 0.98, cy - r * 0.42, cx + r * 0.98, cy + r * 0.98],
              fill=_shade(color, 0.62) + (int(alpha * 0.45),))
    d.ellipse([cx - r * 0.58, cy - r * 0.70, cx - r * 0.02, cy - r * 0.14],
              fill=(255, 255, 255, int(alpha * crown)))


def _canvas(px):
    img = Image.new("RGBA", (px * SS, px * SS), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def _finish(img):
    return img.resize((img.width // SS, img.height // SS),
                      Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.4))


def _palette(dark):
    return (DARK_MAIN, DARK_SOFT) if dark else (INK_BLUE, INK_SOFT)


def _draw(name, px, dark):
    S = px * SS
    img, d = _canvas(px)
    main, soft = _palette(dark)

    def X(f):
        return f * S

    if name == "spark":
        cx, cy = X(0.46), X(0.46)
        pts = []
        for k in range(40):
            a = k / 40.0 * 2 * math.pi
            # Four-point star: radius swells toward both axes' points.
            r = X(0.10) + X(0.32) * (abs(math.cos(a)) ** 8
                                     + abs(math.sin(a)) ** 8) ** 0.5
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        d.polygon(pts, fill=main + (255,))
        bead(d, cx, cy, X(0.075), _shade(main, 1.45))
        bead(d, X(0.78), X(0.78), X(0.07), soft)
    elif name == "list":
        for i, (ln, col) in enumerate(((0.58, main), (0.48, main),
                                       (0.38, soft))):
            y = X(0.28 + 0.22 * i)
            d.rounded_rectangle([X(0.30), y - X(0.055), X(0.30 + ln),
                                 y + X(0.055)], radius=X(0.055),
                                fill=col + (255,))
            bead(d, X(0.16), y, X(0.075), col)
    elif name == "depth":
        cx, cy, r = X(0.5), X(0.5), X(0.35)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=PALE_FILL + (255,))
        d.pieslice([cx - r, cy - r, cx + r, cy + r], 90, 270,
                   fill=main + (255,))
        d.ellipse([cx - r, cy - r, cx + r, cy + r],
                  outline=soft + (255,), width=max(2, int(X(0.03))))
        d.ellipse([cx - r * 0.62, cy - r * 0.72, cx - r * 0.1,
                   cy - r * 0.2], fill=(255, 255, 255, 130))
    elif name == "grid":
        for i in range(3):
            for j in range(3):
                col = main if (i + j) % 2 == 0 else soft
                if i == 1 and j == 1:
                    col = _shade(main, 0.8)
                bead(d, X(0.27 + 0.23 * j), X(0.27 + 0.23 * i),
                     X(0.095), col)
    elif name == "gear":
        cx, cy = X(0.5), X(0.5)
        for k in range(8):
            a = k * math.pi / 4
            x0, y0 = cx + X(0.26) * math.cos(a), cy + X(0.26) * math.sin(a)
            x1, y1 = cx + X(0.42) * math.cos(a), cy + X(0.42) * math.sin(a)
            d.line([(x0, y0), (x1, y1)], fill=soft + (255,),
                   width=int(X(0.15)))
        d.ellipse([cx - X(0.30), cy - X(0.30), cx + X(0.30), cy + X(0.30)],
                  fill=soft + (255,))
        d.ellipse([cx - X(0.13), cy - X(0.13), cx + X(0.13), cy + X(0.13)],
                  fill=(0, 0, 0, 0))
        bead(d, cx, cy, X(0.115), main)
    elif name == "folder":
        d.rounded_rectangle([X(0.08), X(0.30), X(0.92), X(0.82)],
                            radius=X(0.08), fill=PALE_FILL + (255,))
        d.rounded_rectangle([X(0.08), X(0.22), X(0.48), X(0.36)],
                            radius=X(0.06), fill=PALE_FILL + (255,))
        d.rounded_rectangle([X(0.08), X(0.42), X(0.92), X(0.82)],
                            radius=X(0.08), fill=main + (235,))
        bead(d, X(0.72), X(0.60), X(0.085), _shade(main, 1.3))
    elif name == "refresh":
        box = [X(0.14), X(0.14), X(0.86), X(0.86)]
        d.arc(box, start=300, end=235, fill=main + (255,),
              width=int(X(0.11)))
        a = math.radians(300)
        tip = (X(0.5) + X(0.36) * math.cos(a), X(0.5) + X(0.36) * math.sin(a))
        d.polygon([(tip[0] + X(0.13), tip[1] - X(0.02)),
                   (tip[0] - X(0.06), tip[1] - X(0.13)),
                   (tip[0] - X(0.02), tip[1] + X(0.11))],
                  fill=main + (255,))
        bead(d, X(0.5), X(0.5), X(0.075), soft)
    elif name in ("dot_bright", "dot_dim"):
        if name == "dot_bright":
            bead(d, X(0.5), X(0.5), X(0.30), DARK_MAIN, crown=0.95)
        else:
            bead(d, X(0.5), X(0.5), X(0.30), (61, 90, 128), crown=0.35)
    else:
        raise KeyError(name)
    return _finish(img)


_CACHE = {}


def icon_pil(name, px=84, dark=False):
    """The icon `name` as an RGBA PIL image of px x px."""
    key = (name, px, dark)
    if key not in _CACHE:
        _CACHE[key] = _draw(name, px, dark)
    return _CACHE[key]


ICON_NAMES = ("spark", "list", "depth", "grid", "gear", "folder",
              "refresh", "dot_bright", "dot_dim")


# ------------------------------------------------------------------
# The application icon: the pearl M on frosted glass.
# ------------------------------------------------------------------

def build_app_icon(path):
    """(Re)build the .ico at `path` in the coordinated style."""
    from echo_app.glass import graffiti_strokes

    S = 1024
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # Frosted pale rounded square.
    d.rounded_rectangle([8, 8, S - 8, S - 8], radius=210,
                        fill=(233, 238, 244, 255))
    base = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    bd = ImageDraw.Draw(base)
    bd.ellipse([-0.2 * S, -0.3 * S, 0.6 * S, 0.5 * S],
               fill=(168, 196, 240, 90))
    bd.ellipse([0.5 * S, 0.55 * S, 1.25 * S, 1.3 * S],
               fill=(196, 214, 238, 80))
    base = base.filter(ImageFilter.GaussianBlur(80))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([8, 8, S - 8, S - 8],
                                           radius=210, fill=255)
    img = Image.composite(Image.alpha_composite(img, base), img, mask)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([8, 8, S - 8, S - 8], radius=210,
                        outline=(168, 182, 202, 160), width=6)
    # The main graffiti stroke as a string of pearls, remapped into
    # the square with a margin, sizes breathing with its depth wave.
    stroke = graffiti_strokes(1000, 1000)[0]
    pts = stroke["pts"]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    total = 0.0
    cum = [0.0]
    for i in range(1, len(pts)):
        total += math.hypot(pts[i][0] - pts[i - 1][0],
                            pts[i][1] - pts[i - 1][1])
        cum.append(total)
    n = 21
    for k in range(n):
        s = total * k / (n - 1)
        i = max(1, next(j for j, c in enumerate(cum) if c >= s))
        f = (s - cum[i - 1]) / max(1e-9, cum[i] - cum[i - 1])
        x = pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * f
        y = pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * f
        # Uniform scale (no vertical stretch), centred: the string
        # must read as the stroke it is, not as confetti.
        us = (S - 180) / (x1 - x0)
        ix = 90 + (x - x0) * us
        iy = (S - (y1 - y0) * us) / 2 + (y - y0) * us
        zn = 0.5 + 0.5 * math.sin(s * 0.010 + 0.0)
        r = 48 + 13 * zn
        col = tuple(int(INK_BLUE[c] * (0.78 + 0.32 * zn)) for c in range(3))
        bead(d, ix, iy, r, col, crown=0.5)
    img = img.resize((256, 256), Image.LANCZOS)
    img.save(path, format="ICO",
             sizes=[(16, 16), (20, 20), (24, 24), (32, 32), (40, 40),
                    (48, 48), (64, 64), (128, 128), (256, 256)])
    return path
