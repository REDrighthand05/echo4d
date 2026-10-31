#!/usr/bin/env python3
"""ECHO 场景生成器 desktop shell.

The window keeps the Windows 10 Settings home layout the user picked —
an identity header band on top, then a grid of tiles, one per thing you
can do — restyled with a Meta-style rounded look: every surface is a
soft white card with big rounded corners on a pale grey window, and
the buttons are pills. The shell is built with CustomTkinter; the big
生成场景 tile is the drop zone: drag a video onto it (or click it to
pick one), watch the progress bar under the header, and the finished
scene opens by itself. Previous scenes live in the 历史场景 tile. A
separate 深度视频 tile runs the preserved depth-video feature
(echo_scene.depth_video) on a picked video and offers to play the
result. Detailed progress text goes to a collapsible terminal panel
docked on the right (the 日志 edge tab); the main area only ever shows
the plain-language status line. All heavy lifting lives in
echo_scene.pipeline / echo_scene.fuse; this file only wires the stages
together. A finished scene is plain files in its job folder -- the
standalone scene page, the model files, the process preview -- and the
window says where they are and opens the files directly (no server,
no localhost page; that was the v0.1 design the user rejected).
"""
import json
import os
import queue
import subprocess
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _HAS_DND = True
except Exception:  # tkinterdnd2 missing/broken: click-to-pick still works
    DND_FILES = None
    TkinterDnD = None
    _HAS_DND = False

JOBS_ROOT = Path.home() / "ECHO Jobs"
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}

# run_pipeline reports stage labels that read a bit technical; the window
# shows these plain-language versions instead (same order, same stages).
PLAIN_STATUS = {
    "正在抽帧": "正在看画面",
    "正在挑清晰的帧": "正在挑清楚的画面",
    "正在算相机轨迹": "正在找回拍摄的路线",
    "正在认标记纸": "正在认标记纸",
    "正在算每帧的远近": "正在量每处的远近",
    "正在拼场景": "正在拼场景",
    "正在清噪": "正在把场景收拾干净",
    "正在生成网格": "正在生成表面",
    "正在生成像素模型": "正在把场景变成一个个小方块",
    "正在收尾": "马上就好",
    "正在生成过程回放": "正在做一段过程回放",
}
STAGE_COUNT = len(PLAIN_STATUS)


def _resource_path(relative: str) -> Path:
    """Locate a bundled resource: sys._MEIPASS when frozen by
    PyInstaller, otherwise the directory holding this file (the same
    layout the spec's datas entries produce)."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base) / relative
    return Path(__file__).resolve().parent / relative


def open_marker_sheet() -> Path:
    """Open the printable marker sheet PDF with the system viewer."""
    pdf = _resource_path("assets/markers_sheet.pdf")
    if not pdf.exists():
        raise FileNotFoundError(str(pdf))
    if sys.platform.startswith("win"):
        os.startfile(str(pdf))  # noqa: S606 - opening our own bundled PDF
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(pdf)])
    else:
        subprocess.Popen(["xdg-open", str(pdf)])
    return pdf


def run_pipeline(video: Path, jobs_root: Path, log) -> Path:
    """Run every stage in order. log(str) receives human progress lines."""
    from echo_scene.pipeline import stage_depth, stage_extract, stage_present, stage_select
    from echo_scene.pipeline import stage_clean, stage_mesh, stage_poses, stage_preview, stage_voxel
    from echo_scene.markers import stage_markers
    from echo_scene.fuse import stage_fuse

    job = jobs_root / f"job_{video.stem[:24]}"
    n = 1
    base = job
    while job.exists():
        n += 1
        job = jobs_root / f"{base.name}_{n}"
    job.mkdir(parents=True)

    steps = [
        ("正在抽帧", lambda: stage_extract(video, job)),
        ("正在挑清晰的帧", lambda: stage_select(job)),
        ("正在算相机轨迹", lambda: stage_poses(job)),
        ("正在认标记纸", lambda: stage_markers(job)),
        ("正在算每帧的远近", lambda: stage_depth(job)),
        ("正在拼场景", lambda: stage_fuse(job)),
        ("正在清噪", lambda: stage_clean(job)),
        ("正在生成网格", lambda: stage_mesh(job)),
        ("正在生成像素模型", lambda: stage_voxel(job)),
        ("正在收尾", lambda: stage_present(job)),
        ("正在生成过程回放", lambda: stage_preview(job)),
    ]
    for label, fn in steps:
        log(label + "…")
        info = fn()
        if isinstance(info, dict) and info.get("ok") is False:
            raise RuntimeError(f"{label}失败：{info.get('reason', '未知原因')}")
    log("完成。")
    return job


def run_depth_video(video: Path) -> Path:
    """Turn a video into its black-and-white depth version, using the
    preserved echo_scene.depth_video feature as-is. The mp4 lands next
    to the source video (the module keeps its depth stills beside it)."""
    from echo_scene.depth_video import make_depth_video

    out = video.with_name(f"{video.stem}_depth.mp4")
    base = out
    n = 1
    while out.exists():
        n += 1
        out = base.with_name(f"{base.stem}_{n}.mp4")
    info = make_depth_video(video, out)
    if not isinstance(info, dict) or info.get("ok") is False:
        reason = info.get("reason", "未知原因") if isinstance(info, dict) else "未知原因"
        raise RuntimeError(f"生成深度视频失败：{reason}")
    return out


def play_file(path: Path):
    """Open a file with the system player/viewer."""
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # noqa: S606 - opening our own output file
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def scene_file(job: Path) -> Path:
    """The job's standalone scene page. Jobs finished by older versions
    only hold viewer.html + points.bin, so build scene.html on demand;
    it is a pure file either way."""
    scene = job / "scene.html"
    if not scene.exists() and (job / "points.bin").exists():
        try:
            from echo_scene.pipeline import stage_present

            stage_present(job)
        except Exception:
            pass
    return scene


def list_previous_jobs(jobs_root: Path):
    """Finished scenes under jobs_root, newest first. A job counts as
    finished once it holds a scene page (scene.html, or viewer.html
    for jobs finished by older versions)."""
    if not jobs_root.exists():
        return []
    jobs = [
        d for d in jobs_root.iterdir()
        if d.is_dir() and ((d / "scene.html").exists() or (d / "viewer.html").exists())
    ]
    jobs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return jobs


def _job_display_name(job: Path) -> str:
    name = job.name
    if name.startswith("job_"):
        name = name[len("job_"):]
    stamp = datetime.fromtimestamp(job.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return f"{name}（{stamp}）"


# Meta-style palette: soft grey window, white rounded cards, one blue.
C_BG = "#F0F2F5"         # window surface
C_SURFACE = "#FFFFFF"    # cards (header / tiles / log panel)
C_BORDER = "#DADDE1"     # subtle card borders
C_INK = "#1C1E21"        # primary text
C_SUBINK = "#606770"     # secondary text
C_ACCENT = "#0064D1"     # Meta blue
C_ACCENT_HOVER = "#0053AD"
C_BTN2 = "#E4E6EB"       # secondary pill buttons
C_BTN2_HOVER = "#D8DADF"
C_TRACK = "#E4E6EB"      # progress bar track
C_LOG_BG = "#F7F8FA"     # log textbox surface
C_DRAG = "#E7F3FF"       # drag-over highlight
CORNER = 18              # card corner radius

# Preferred UI font first; whatever the platform actually has wins, so
# the window looks native on Windows and still builds anywhere else.
FONT_PREFS = ("Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI",
              "Noto Sans CJK SC", "Noto Sans Sans CJK SC",
              "WenQuanYi Zen Hei", "DejaVu Sans")


def main():
    import tkinter as tk
    import customtkinter as ctk
    from tkinter import filedialog, font as tkfont, messagebox

    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")

    root = None
    dnd_ready = False
    if _HAS_DND:
        try:
            from tkinterdnd2.TkinterDnD import _require as _dnd_require

            class _DnDRoot(ctk.CTk, TkinterDnD.DnDWrapper):
                """CustomTkinter root with the tkdnd package loaded.

                DnDWrapper's methods are patched onto every widget by
                tkinterdnd2 itself, so the CTk tiles/labels below can be
                registered as drop targets exactly like plain tk ones.
                """

                def __init__(self):
                    super().__init__()
                    self.TkdndVersion = _dnd_require(self)

            root = _DnDRoot()
            dnd_ready = True
        except Exception:  # tkdnd native lib failed to load: plain window
            if root is not None:
                try:
                    root.destroy()
                except Exception:
                    pass
            root = None
    if root is None:
        root = ctk.CTk()
    root.title("ECHO 场景生成器")
    root.geometry("860x700")
    root.minsize(700, 620)
    root.configure(fg_color=C_BG)
    state = {"job": None, "busy": False, "history": [],
             "depth_video": None, "panel_open": False}
    ui_queue = queue.Queue()

    available = set(tkfont.families(root))
    family = next((f for f in FONT_PREFS if f in available), FONT_PREFS[0])

    def F(size, bold=False):
        return ctk.CTkFont(family=family, size=size,
                           weight="bold" if bold else "normal")

    def FT(size, bold=False):
        # plain tuple font for the one remaining tk widget (the Listbox)
        return (family, size, "bold") if bold else (family, size)

    def pill(master, text, primary=False, font_size=13, state="normal"):
        """A rounded pill button: blue primary, soft grey secondary."""
        if primary:
            return ctk.CTkButton(
                master, text=text, font=F(font_size), height=36,
                corner_radius=18, fg_color=C_ACCENT,
                hover_color=C_ACCENT_HOVER, text_color="#FFFFFF",
                state=state)
        return ctk.CTkButton(
            master, text=text, font=F(font_size), height=36,
            corner_radius=18, fg_color=C_BTN2, hover_color=C_BTN2_HOVER,
            text_color=C_INK, state=state)

    outer = ctk.CTkFrame(root, fg_color=C_BG, corner_radius=0)
    outer.pack(fill="both", expand=True)

    center = ctk.CTkFrame(outer, fg_color=C_BG, corner_radius=0)
    center.pack(side="left", fill="both", expand=True)

    # --- collapsible terminal panel (right edge, hidden by default) -----
    panel = ctk.CTkFrame(outer, fg_color=C_SURFACE, corner_radius=CORNER,
                         border_width=1, border_color=C_BORDER, width=340)
    panel.pack_propagate(False)
    ctk.CTkLabel(panel, text="日志", font=F(12, True),
                 text_color=C_INK).pack(anchor="w", padx=14, pady=(12, 4))
    logbox = ctk.CTkTextbox(panel, height=120, state="disabled", font=F(11),
                            corner_radius=12, fg_color=C_LOG_BG,
                            text_color=C_INK)
    logbox.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    strip = ctk.CTkFrame(outer, fg_color=C_BG, corner_radius=0)
    strip.pack(side="right", fill="y")
    btn_log = ctk.CTkButton(strip, text="日\n志\n›", font=F(11), width=36,
                            height=76, corner_radius=18, fg_color=C_BTN2,
                            hover_color=C_BTN2_HOVER, text_color=C_INK)
    btn_log.pack(side="top", padx=(0, 6), pady=10)

    def toggle_panel():
        if state["panel_open"]:
            panel.pack_forget()
            state["panel_open"] = False
            btn_log.configure(text="日\n志\n›")
        else:
            panel.pack(side="right", fill="y", padx=(0, 4), pady=10)
            state["panel_open"] = True
            btn_log.configure(text="日\n志\n‹")

    btn_log.configure(command=toggle_panel)

    # --- header band ------------------------------------------------------
    header = ctk.CTkFrame(center, fg_color=C_SURFACE, corner_radius=CORNER,
                          border_width=1, border_color=C_BORDER)
    header.pack(fill="x", padx=12, pady=(12, 8))
    ctk.CTkLabel(header, text="ECHO 回声", font=F(22, True),
                 text_color=C_INK).pack(anchor="w", padx=18, pady=(12, 0))
    ctk.CTkLabel(header, text="视频变 3D 场景", font=F(12),
                 text_color=C_SUBINK).pack(anchor="w", padx=18, pady=(0, 12))

    # --- progress band (plain-language status only) -----------------------
    band = ctk.CTkFrame(center, fg_color=C_BG, corner_radius=0)
    band.pack(fill="x")
    status = ctk.CTkLabel(band, text="还没开始。", font=F(12),
                          text_color=C_INK)
    status.pack(anchor="w", padx=16, pady=(2, 0))
    bar = ctk.CTkProgressBar(band, mode="determinate", height=10,
                             corner_radius=5, fg_color=C_TRACK,
                             progress_color=C_ACCENT)
    bar.set(0)
    bar.pack(fill="x", padx=16, pady=(4, 8))

    # --- tile grid ----------------------------------------------------------
    tiles = ctk.CTkFrame(center, fg_color=C_BG, corner_radius=0)
    tiles.pack(fill="both", expand=True, padx=12, pady=(2, 12))
    tiles.columnconfigure(0, weight=1, uniform="tile")
    tiles.columnconfigure(1, weight=1, uniform="tile")
    tiles.rowconfigure(1, weight=1)

    def make_tile(row, column, columnspan=1, rowspan=1,
                  padx=(0, 0), pady=(0, 0)):
        tile = ctk.CTkFrame(tiles, fg_color=C_SURFACE, corner_radius=CORNER,
                            border_width=1, border_color=C_BORDER)
        tile.grid(row=row, column=column, columnspan=columnspan,
                  rowspan=rowspan, sticky="nsew", padx=padx, pady=pady)
        return tile

    def tile_header(tile, glyph, title, desc):
        parts = [
            ctk.CTkLabel(tile, text=glyph, font=F(22), text_color=C_ACCENT),
            ctk.CTkLabel(tile, text=title, font=F(14, True),
                         text_color=C_INK),
            ctk.CTkLabel(tile, text=desc, font=F(11), text_color=C_SUBINK,
                         justify="left"),
        ]
        parts[0].pack(anchor="w", padx=14, pady=(12, 0))
        parts[1].pack(anchor="w", padx=14)
        parts[2].pack(anchor="w", padx=14, pady=(0, 6))
        return parts

    # Hero tile: generating a scene. It IS the drop zone.
    zone_idle = "把视频拖到这里，或者点这里选择视频\n选一段慢慢走一圈拍的视频，剩下的交给我。"
    hero = make_tile(0, 0, columnspan=2, pady=(2, 8))
    try:
        hero.configure(cursor="hand2")
    except Exception:
        pass
    hero_parts = tile_header(hero, "✦", "生成场景", zone_idle)
    hero_desc = hero_parts[2]
    hero_btn_row = ctk.CTkFrame(hero, fg_color="transparent")
    hero_btn_row.pack(anchor="w", padx=14, pady=(0, 12))
    btn_pick = pill(hero_btn_row, "选择视频", primary=True, font_size=12)
    btn_pick.pack(side="left")
    btn_open = pill(hero_btn_row, "打开场景", font_size=12, state="disabled")
    btn_open.pack(side="left", padx=(8, 0))
    btn_folder = pill(hero_btn_row, "打开文件夹", font_size=12, state="disabled")
    btn_folder.pack(side="left", padx=(8, 0))
    btn_preview = pill(hero_btn_row, "过程回放", font_size=12, state="disabled")
    btn_preview.pack(side="left", padx=(8, 0))

    def hero_bg(color):
        # CTk labels/rows are transparent, so tinting the card is enough.
        hero.configure(fg_color=color)

    # History tile: previous scenes, each reopenable.
    hist_tile = make_tile(1, 0, rowspan=2, padx=(0, 4))
    tile_header(hist_tile, "☰", "历史场景", "以前做好的场景，选一个点“打开”。")
    hist_list = tk.Listbox(hist_tile, height=4, font=FT(11), relief="flat",
                           borderwidth=0, highlightthickness=0,
                           bg=C_SURFACE, fg=C_INK,
                           selectbackground=C_ACCENT,
                           selectforeground="#FFFFFF")
    hist_list.pack(fill="both", expand=True, padx=14, pady=(2, 6))
    hist_btn_row = ctk.CTkFrame(hist_tile, fg_color="transparent")
    hist_btn_row.pack(anchor="e", padx=14, pady=(0, 12))
    btn_hist_open = pill(hist_btn_row, "打开场景", font_size=11, state="disabled")
    btn_hist_open.pack(side="left")
    btn_hist_folder = pill(hist_btn_row, "打开文件夹", font_size=11, state="disabled")
    btn_hist_folder.pack(side="left", padx=(6, 0))
    btn_hist_preview = pill(hist_btn_row, "过程回放", font_size=11, state="disabled")
    btn_hist_preview.pack(side="left", padx=(6, 0))

    # Depth-video tile.
    depth_tile = make_tile(1, 1, padx=(4, 0), pady=(0, 8))
    tile_header(depth_tile, "◐", "深度视频", "把一段视频做成黑白的深度画面。")
    depth_btn_row = ctk.CTkFrame(depth_tile, fg_color="transparent")
    depth_btn_row.pack(anchor="w", padx=14, pady=(2, 12))
    btn_depth = pill(depth_btn_row, "深度视频", primary=True, font_size=12)
    btn_depth.pack(side="left")
    btn_depth_play = pill(depth_btn_row, "播放深度视频", font_size=12,
                          state="disabled")
    btn_depth_play.pack(side="left", padx=(8, 0))

    # Marker-sheet tile.
    marker_tile = make_tile(2, 1, padx=(4, 0))
    tile_header(marker_tile, "▦", "打印标记纸",
                "打印出来贴在房间里，帮场景认准大小。")
    btn_markers = pill(marker_tile, "打印标记纸", primary=True, font_size=12)
    btn_markers.pack(anchor="w", padx=14, pady=(2, 12))

    # --- UI helpers (Tk thread only) ------------------------------------
    def append_log(text):
        logbox.configure(state="normal")
        logbox.insert("end", text + "\n")
        logbox.see("end")
        logbox.configure(state="disabled")

    def log(text):
        """Thread-safe progress sink handed to run_pipeline."""
        ui_queue.put(("log", text))

    def set_busy(busy, busy_text="正在做场景…"):
        state["busy"] = busy
        btn_pick.configure(state="disabled" if busy else "normal")
        btn_depth.configure(state="disabled" if busy else "normal")
        hero_desc.configure(text=busy_text if busy else zone_idle)

    def refresh_history():
        jobs = list_previous_jobs(JOBS_ROOT)
        state["history"] = jobs
        hist_list.delete(0, "end")
        for job in jobs:
            hist_list.insert("end", _job_display_name(job))
        for b in (btn_hist_open, btn_hist_folder, btn_hist_preview):
            b.configure(state="disabled")

    def open_job(job):
        state["job"] = job
        scene = scene_file(job)
        btn_open.configure(text="重新打开场景", state="normal")
        btn_folder.configure(state="normal")
        btn_preview.configure(
            state="normal" if (job / "preview.html").exists() else "disabled")
        if scene.exists():
            play_file(scene)

    def handle_log(text):
        line = text.rstrip("…").strip()
        if line in PLAIN_STATUS:
            plain = PLAIN_STATUS[line]
            status.configure(text=plain + "…")
            bar.set((list(PLAIN_STATUS).index(line) + 1) / STAGE_COUNT)
            append_log(plain)
        elif line == "完成。":
            status.configure(text="做好了。")
            bar.set(1)
        else:
            append_log(text)

    def handle_done(job):
        set_busy(False)
        refresh_history()
        open_job(job)
        status.configure(text=f"做好了。文件都在这个文件夹里：{job}")
        append_log(f"做好了。文件都在这个文件夹里：{job}")
        for name in ("scene.html", "voxel_model.glb", "voxel_model.ply",
                     "mesh.ply", "scene_points.ply", "preview.html"):
            p = job / name
            if p.exists():
                append_log(f"  {p}")

    def handle_error(detail):
        set_busy(False)
        bar.set(0)
        status.configure(text="没成功，这段视频没能做成场景。")
        append_log(detail)
        append_log("没成功，这段视频没能做成场景。")
        messagebox.showerror("没成功", "这段视频没能做成场景，详情看日志。")

    def handle_depth_done(out):
        set_busy(False)
        bar.stop()
        bar.configure(mode="determinate")
        bar.set(1)
        state["depth_video"] = out
        btn_depth_play.configure(state="normal")
        status.configure(text="深度视频做好了，点“播放深度视频”就能看。")
        append_log(f"深度视频做好了：{out}")

    def handle_depth_error(detail):
        set_busy(False)
        bar.stop()
        bar.configure(mode="determinate")
        bar.set(0)
        status.configure(text="没成功，深度视频没做成。")
        append_log(detail)
        append_log("没成功，深度视频没做成。")
        messagebox.showerror("没成功", "深度视频没做成，详情看日志。")

    def drain_queue():
        try:
            while True:
                kind, payload = ui_queue.get_nowait()
                if kind == "log":
                    handle_log(payload)
                elif kind == "done":
                    handle_done(payload)
                elif kind == "error":
                    handle_error(payload)
                elif kind == "depth_done":
                    handle_depth_done(payload)
                elif kind == "depth_error":
                    handle_depth_error(payload)
        except queue.Empty:
            pass
        root.after(80, drain_queue)

    # --- actions ----------------------------------------------------------
    def worker(video):
        try:
            job = run_pipeline(video, JOBS_ROOT, log)
            ui_queue.put(("done", job))
        except Exception:
            ui_queue.put(("error", traceback.format_exc(limit=2)))

    def depth_worker(video):
        try:
            out = run_depth_video(video)
            ui_queue.put(("depth_done", out))
        except Exception:
            ui_queue.put(("depth_error", traceback.format_exc(limit=2)))

    def check_video(video: Path) -> bool:
        if video.suffix.lower() not in VIDEO_SUFFIXES:
            status.configure(text="这个文件不是视频，换个视频文件试试。")
            return False
        if not video.exists():
            status.configure(text="没找到这个视频文件。")
            return False
        return True

    def start(video: Path):
        if state["busy"]:
            return
        if not check_video(video):
            return
        bar.set(0)
        status.configure(text="开始了。")
        btn_open.configure(state="disabled")
        btn_folder.configure(state="disabled")
        btn_preview.configure(state="disabled")
        set_busy(True)
        threading.Thread(target=worker, args=(video,), daemon=True).start()

    def pick(_event=None):
        if state["busy"]:
            return
        path = filedialog.askopenfilename(
            title="选择视频",
            filetypes=[("视频", "*.mp4 *.mov *.mkv *.avi"), ("所有文件", "*.*")],
        )
        if path:
            start(Path(path))

    def start_depth(video: Path):
        if state["busy"]:
            return
        if not check_video(video):
            return
        btn_depth_play.configure(state="disabled")
        status.configure(text="正在生成深度视频…")
        bar.configure(mode="indeterminate")
        bar.start()
        set_busy(True, "正在生成深度视频…")
        threading.Thread(target=depth_worker, args=(video,), daemon=True).start()

    def pick_depth(_event=None):
        if state["busy"]:
            return
        path = filedialog.askopenfilename(
            title="选择视频（做深度视频）",
            filetypes=[("视频", "*.mp4 *.mov *.mkv *.avi"), ("所有文件", "*.*")],
        )
        if path:
            start_depth(Path(path))

    def play_depth():
        out = state["depth_video"]
        if out:
            play_file(out)

    def open_scene():
        job = state["job"]
        if job:
            scene = scene_file(job)
            if scene.exists():
                play_file(scene)

    def open_folder():
        if state["job"]:
            play_file(state["job"])

    def open_preview():
        job = state["job"]
        if job and (job / "preview.html").exists():
            play_file(job / "preview.html")

    def print_markers():
        try:
            open_marker_sheet()
            log("已打开标记纸：打印时选实际大小（100%），0 号平贴地板，1-3 号贴墙。")
        except Exception:
            messagebox.showerror("打不开", "没找到标记纸文件（assets/markers_sheet.pdf）。")

    def selected_history_job():
        sel = hist_list.curselection()
        if sel and sel[0] < len(state["history"]):
            return state["history"][sel[0]]
        return None

    def on_history_select(_event=None):
        job = selected_history_job()
        btn_hist_open.configure(state="normal" if job else "disabled")
        btn_hist_folder.configure(state="normal" if job else "disabled")
        btn_hist_preview.configure(
            state="normal" if job and (job / "preview.html").exists()
            else "disabled")

    def open_history(_event=None):
        job = selected_history_job()
        if job:
            open_job(job)
            status.configure(text=f"已经打开。文件在这个文件夹里：{job}")

    def open_history_folder():
        job = selected_history_job()
        if job:
            play_file(job)

    def open_history_preview():
        job = selected_history_job()
        if job and (job / "preview.html").exists():
            play_file(job / "preview.html")

    btn_pick.configure(command=pick)
    btn_open.configure(command=open_scene)
    btn_folder.configure(command=open_folder)
    btn_preview.configure(command=open_preview)
    btn_hist_folder.configure(command=open_history_folder)
    btn_hist_preview.configure(command=open_history_preview)
    btn_markers.configure(command=print_markers)
    btn_depth.configure(command=pick_depth)
    btn_depth_play.configure(command=play_depth)
    btn_hist_open.configure(command=open_history)
    hist_list.bind("<<ListboxSelect>>", on_history_select)
    hist_list.bind("<Double-Button-1>", open_history)
    for w in [hero] + hero_parts:
        w.bind("<Button-1>", pick)

    # --- drag and drop (optional; click-to-pick always works) ------------
    def on_drop(event):
        hero_bg(C_SURFACE)
        if state["busy"]:
            return
        try:
            paths = root.tk.splitlist(event.data)
        except Exception:
            paths = event.data.split()
        paths = [p for p in paths if p]
        if not paths:
            return
        if len(paths) > 1:
            append_log("一次先做一段视频，先用第一段。")
        start(Path(paths[0]))

    if dnd_ready:
        for widget in [hero, root] + hero_parts:
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", on_drop)
        for widget in [hero] + hero_parts:
            widget.dnd_bind("<<DragEnter>>", lambda _e: hero_bg(C_DRAG))
            widget.dnd_bind("<<DragLeave>>", lambda _e: hero_bg(C_SURFACE))

    refresh_history()
    root.after(80, drain_queue)
    root.mainloop()


if __name__ == "__main__":
    main()
