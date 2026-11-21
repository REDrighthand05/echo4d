"""Application entry point and event loop for legacy CustomTkinter GUI."""

import queue
from pathlib import Path
from tkinter import font as tkfont
import customtkinter as ctk

from echo_app.settings import load_settings
from echo_desktop.tk_gui.components import make_card, make_log_panel, make_pill
from echo_desktop.tk_gui.handlers import TkActionHandler
from echo_desktop.tk_gui.theme import (
    C_ACCENT,
    C_BG,
    C_INK,
    C_SUBINK,
    FONT_PREFS,
    HAS_DND,
    SKIN,
)

DEFAULT_JOBS_ROOT = Path.home() / "ECHO Jobs"


def main():
    """Start the CustomTkinter fallback GUI."""
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.title("ECHO 场景生成器")
    root.geometry("1020x800")
    root.minsize(860, 700)
    root.configure(fg_color=C_BG)

    wall = None
    try:
        from echo_app.glass import GlassWall
        wall = GlassWall(root)
    except Exception:
        pass

    settings = load_settings()
    jobs_root = settings.jobs_root_path() or DEFAULT_JOBS_ROOT

    state = {
        "job": None,
        "busy": False,
        "video": None,
        "depth_video": None,
        "panel_open": False,
        "status_text": "空闲",
    }
    ui_queue = queue.Queue()

    available = set(tkfont.families(root))
    family = next((f for f in FONT_PREFS if f in available), FONT_PREFS[0])

    def F(sz, bold=False):
        return ctk.CTkFont(family=family, size=sz, weight="bold" if bold else "normal")

    outer = ctk.CTkFrame(root, fg_color=C_BG, corner_radius=0)
    outer.pack(fill="both", expand=True)

    center = ctk.CTkFrame(outer, fg_color=C_BG, corner_radius=0)
    center.pack(side="left", fill="both", expand=True)

    panel, logbox = make_log_panel(outer, F, wall=wall)

    def log(msg: str):
        logbox.configure(state="normal")
        logbox.insert("end", str(msg) + "\n")
        logbox.see("end")
        logbox.configure(state="disabled")

    handler = TkActionHandler(state, ui_queue, jobs_root, log)

    # Header
    header = make_card(center, wall=wall, name="header")
    header.pack(fill="x", padx=8, pady=(10, 6))

    _hdr_title = ctk.CTkLabel(header, text="ECHO 回声", font=F(22, True), text_color=C_INK)
    _hdr_title.pack(anchor="w", padx=18, pady=(12, 0))
    _hdr_sub = ctk.CTkLabel(header, text="视频变 3D 场景", font=F(12), text_color=C_SUBINK)
    _hdr_sub.pack(anchor="w", padx=18, pady=(0, 12))

    # Progress bar & status
    band = ctk.CTkFrame(center, fg_color=C_BG, corner_radius=0)
    band.pack(fill="x")
    lbl_status = ctk.CTkLabel(band, text="就绪", font=F(12), text_color=C_INK)
    lbl_status.pack(anchor="w", padx=16, pady=(2, 0))

    pbar = ctk.CTkProgressBar(
        band,
        mode="determinate",
        height=SKIN["bar_h"],
        corner_radius=5,
        fg_color=SKIN["track"],
        progress_color=C_ACCENT,
    )
    pbar.set(0)
    pbar.pack(fill="x", padx=16, pady=(4, 8))

    # Tiles grid
    tiles = ctk.CTkFrame(center, fg_color=C_BG, corner_radius=0)
    tiles.pack(fill="both", expand=True, padx=8, pady=(2, 8))

    def on_pick_video():
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            filetypes=[("视频文件", "*.mp4;*.mov;*.mkv;*.avi"), ("所有文件", "*.*")]
        )
        if path:
            if handler.select_video(Path(path)):
                lbl_status.configure(text=f"已选视频：{Path(path).name}")

    def on_start():
        if state.get("video") and not state.get("busy"):
            lbl_status.configure(text="正在开始生成场景…")
            pbar.set(0.05)
            handler.start_pipeline()

    card_drop = make_card(tiles, wall=wall, name="drop")
    card_drop.pack(side="left", fill="both", expand=True, padx=4, pady=4)

    btn_pick = make_pill(card_drop, "选择视频文件", F(13), primary=False)
    btn_pick.configure(command=on_pick_video)
    btn_pick.pack(padx=20, pady=(30, 10))

    btn_run = make_pill(card_drop, "开始生成 3D 场景", F(13), primary=True)
    btn_run.configure(command=on_start)
    btn_run.pack(padx=20, pady=10)

    # Queue drain loop
    def drain():
        while not ui_queue.empty():
            try:
                kind, data = ui_queue.get_nowait()
                if kind == "log":
                    log(data)
                elif kind == "job_started":
                    state["job"] = data
                    lbl_status.configure(text=f"任务创建: {data.name}")
                elif kind == "done":
                    state["busy"] = False
                    pbar.set(1.0)
                    lbl_status.configure(text="场景制作完成！")
                    log("场景完成。")
                elif kind == "error":
                    state["busy"] = False
                    lbl_status.configure(text="生成失败，看一下日志。")
                    log(f"错误: {data}")
            except Exception:
                pass
        root.after(200, drain)

    root.after(200, drain)
    root.mainloop()


if __name__ == "__main__":
    main()
