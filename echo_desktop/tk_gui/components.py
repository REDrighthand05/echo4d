"""Custom Tkinter widgets and layout builders for legacy GUI."""

import customtkinter as ctk
from echo_desktop.tk_gui.theme import (
    C_ACCENT,
    C_BORDER,
    C_BTN2,
    C_BTN2_HOVER,
    C_INK,
    C_LOG_BG,
    C_SURFACE,
    SKIN,
)


def make_pill(master, text, font, primary=False, font_size=13, state="normal"):
    """Rounded pill button styled with current skin theme."""
    if primary:
        return ctk.CTkButton(
            master,
            text=text,
            font=font,
            height=36,
            corner_radius=18,
            fg_color=SKIN["p_fg"],
            hover_color=SKIN["p_hover"],
            text_color=SKIN["p_text"],
            state=state,
        )
    return ctk.CTkButton(
        master,
        text=text,
        font=font,
        height=36,
        corner_radius=18,
        fg_color=SKIN["s_fg"],
        hover_color=SKIN["s_hover"],
        border_width=0,
        text_color=C_INK,
        state=state,
    )


def make_card(parent, wall=None, name="card", corner=None, border=0):
    """Create a styled translucent card frame."""
    radius = corner or SKIN["corner"]
    card = ctk.CTkFrame(
        parent,
        fg_color=C_SURFACE,
        corner_radius=radius,
        border_width=border,
        border_color=C_BORDER,
    )
    if wall is not None:
        try:
            wall.attach(card, name=name, milk=SKIN["pane_milk"], blur=SKIN["pane_blur"], radius=radius)
        except Exception:
            pass
    return card


def make_log_panel(outer, font_fn, wall=None):
    """Build the slide-out log drawer on the right edge."""
    panel = ctk.CTkFrame(
        outer,
        fg_color=C_SURFACE,
        corner_radius=SKIN["corner"],
        border_width=0,
        width=340,
    )
    panel.pack_propagate(False)
    if wall is not None:
        try:
            wall.attach(panel, name="log", milk=SKIN["pane_milk"], blur=SKIN["pane_blur"], radius=SKIN["corner"])
        except Exception:
            pass

    lbl = ctk.CTkLabel(panel, text="日志", font=font_fn(12, True), text_color=C_INK)
    lbl.pack(anchor="w", padx=14, pady=(12, 4))

    logbox = ctk.CTkTextbox(
        panel,
        height=120,
        state="disabled",
        font=font_fn(11),
        corner_radius=12,
        fg_color=C_LOG_BG,
        text_color=C_INK,
    )
    logbox.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    return panel, logbox
