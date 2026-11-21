"""Color palette, typography, and visual themes for legacy Tkinter GUI."""

import os

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except Exception:
    DND_FILES = None
    TkinterDnD = None
    HAS_DND = False

C_BG = "#E9EDF2"
C_SURFACE = "#FFFFFF"
C_BORDER = "#DADDE1"
C_INK = "#1C1E21"
C_SUBINK = "#606770"
C_ACCENT = "#0064D1"
C_ACCENT_HOVER = "#0053AD"
C_BTN2 = "#E9EDF3"
C_BTN2_HOVER = "#DCE3EC"
C_TRACK = "#E4E6EB"
C_LOG_BG = "#F4F6FA"
C_DRAG = "#E7F3FF"
C_PREVIEW_BG = "#05070E"
CORNER = 24
CORNER_SM = 16

SKINS = {
    "now": dict(
        corner=24, pane_milk=100, pane_blur=19, p_fg=C_ACCENT,
        p_hover=C_ACCENT_HOVER, p_text="#FFFFFF", s_fg=C_BTN2,
        s_hover=C_BTN2_HOVER, bar_h=10, track=C_TRACK,
    ),
    "a": dict(
        corner=28, pane_milk=58, pane_blur=14, p_fg="#AAC7FA",
        p_hover="#97B8F7", p_text="#1E4FD6", s_fg="#EFF4FB",
        s_hover="#E1EBF9", bar_h=6, track="#DFE8F3",
    ),
    "b": dict(
        corner=20, pane_milk=118, pane_blur=16, p_fg=C_ACCENT,
        p_hover=C_ACCENT_HOVER, p_text="#FFFFFF", s_fg="#FAFCFF",
        s_hover="#EBF1FA", bar_h=8, track="#E3E9F2",
    ),
    "c": dict(
        corner=32, pane_milk=26, pane_blur=10, p_fg="transparent",
        p_hover="#E1EBF9", p_text=C_ACCENT, s_fg="transparent",
        s_hover="#E7EFFA", bar_h=3, track="#E6EDF6",
    ),
}
SKIN = SKINS.get(os.environ.get("ECHO_SKIN", "a"), SKINS["now"])

FONT_PREFS = (
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "Segoe UI",
    "Noto Sans CJK SC",
    "Noto Sans Sans CJK SC",
    "WenQuanYi Zen Hei",
    "DejaVu Sans",
)
