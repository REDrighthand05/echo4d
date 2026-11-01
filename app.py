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
standalone scene page, the model files, the process preview, the
route replay -- and the window says where they are and opens them
directly (no server, no localhost page; that was the v0.1 design
the user rejected).

v0.4 shell behaviour (all of it edge code in echo_app, none of it in
the pipeline): the window lives in the system tray WeChat-style --
closing it hides it and a running job continues; a toast announces
completion while hidden; the app checks GitHub Releases for a newer
version and can download + launch the new installer (which upgrades
in place); a settings tile holds the jobs folder, 开机自启 and the
tray behaviour. Every button's enabled state derives from the one
explicit state in echo_app.appstate, and every optional subsystem is
failure-contained so it can never take a job down.
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
    "正在定真尺度": "正在找参照物，量出真实大小",
    "正在生成网格": "正在生成表面",
    "正在生成像素模型": "正在把场景变成一个个小方块",
    "正在收尾": "马上就好",
    "正在生成过程回放": "正在做一段过程回放",
    "正在生成路线回放": "正在做一段路线回放",
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
    from echo_scene.pipeline import stage_clean, stage_mesh, stage_poses, stage_preview, stage_replay, stage_scale, stage_voxel
    from echo_scene.markers import stage_markers
    from echo_scene.fuse import stage_fuse

    job = jobs_root / f"job_{video.stem[:24]}"
    n = 1
    base = job
    while job.exists():
        n += 1
        job = jobs_root / f"{base.name}_{n}"
    job.mkdir(parents=True)
    # Crash-recovery marker: from this moment the folder is recognisably
    # one of ours, so if the app dies mid-job the relaunch can list the
    # job as unfinished (each pipeline stage overwrites job.json with
    # its own record as it completes -- that stays untouched).
    try:
        (job / "job.json").write_text(json.dumps(
            {"stage": "started", "ok": False, "status": "running",
             "video": str(video)}, ensure_ascii=False, indent=2))
    except Exception:
        pass

    steps = [
        ("正在抽帧", lambda: stage_extract(video, job)),
        ("正在挑清晰的帧", lambda: stage_select(job)),
        ("正在算相机轨迹", lambda: stage_poses(job)),
        ("正在认标记纸", lambda: stage_markers(job)),
        ("正在算每帧的远近", lambda: stage_depth(job)),
        ("正在拼场景", lambda: stage_fuse(job)),
        ("正在清噪", lambda: stage_clean(job)),
        ("正在定真尺度", lambda: stage_scale(job)),
        ("正在生成网格", lambda: stage_mesh(job)),
        ("正在生成像素模型", lambda: stage_voxel(job)),
        ("正在收尾", lambda: stage_present(job)),
        ("正在生成过程回放", lambda: stage_preview(job)),
        ("正在生成路线回放", lambda: stage_replay(job)),
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


def replay_file(job: Path) -> Path:
    """The job's standalone route-replay page. Jobs finished before
    the replay existed only hold the scene data, so build replay.html
    on demand from points.bin + poses.json; a pure file either way."""
    replay = job / "replay.html"
    if not replay.exists() and (job / "points.bin").exists() \
            and (job / "poses.json").exists():
        try:
            from echo_scene.pipeline import stage_replay

            stage_replay(job)
        except Exception:
            pass
    return replay


def _replay_available(job: Path) -> bool:
    """True when the job can show a route replay: the page is already
    there, or the data needed to build it on demand is."""
    return (job / "replay.html").exists() or (
        (job / "points.bin").exists() and (job / "poses.json").exists())


def job_finished(job: Path) -> bool:
    """A job is finished once it holds a scene page (scene.html, or
    viewer.html for jobs finished by older versions)."""
    return (job / "scene.html").exists() or (job / "viewer.html").exists()


def list_previous_jobs(jobs_root: Path):
    """Jobs under jobs_root, newest first: finished ones AND
    unfinished ones -- a folder is ours (and therefore listable) once
    it holds a scene page or the job.json marker run_pipeline writes
    at job start, so a job the app died in the middle of still shows
    up after relaunch, marked 未完成， with whatever outputs it has."""
    if not jobs_root.exists():
        return []
    jobs = [
        d for d in jobs_root.iterdir()
        if d.is_dir() and (job_finished(d) or (d / "job.json").exists())
    ]
    jobs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return jobs


def scale_status_line(job):
    """The job's true-scale status as one calm line, read from the
    scale.json the scale stage writes. None for jobs that predate the
    stage (no record -- nothing honest to show)."""
    if job is None:
        return None
    path = job / "scale.json"
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if rec.get("status") == "set":
        anchor = (rec.get("anchor") or {}).get("label") or "参照物"
        if rec.get("evidence") == "实测":
            return f"真尺度：已定（{anchor}实测）"
        err = rec.get("error_pct")
        err_txt = f"±{err}%" if err else "估计"
        return f"真尺度：已定（估计 {err_txt}，来自{anchor}）"
    return "真尺度：未定（无参照物）"


def _job_display_name(job: Path) -> str:
    name = job.name
    if name.startswith("job_"):
        name = name[len("job_"):]
    stamp = datetime.fromtimestamp(job.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    suffix = "" if job_finished(job) else "（未完成）"
    line = scale_status_line(job)
    if line:
        suffix += f" · {line}"
    return f"{name}（{stamp}）{suffix}"


def job_file_flags(job):
    """echo_app.appstate.FileFlags for one job folder: what each
    output action can actually do right now. Scene and replay count
    as available when they can be built on demand from the job's
    data, not only when the page already exists."""
    from echo_app.appstate import FileFlags

    if job is None:
        return FileFlags()
    scene = (job / "scene.html").exists() or (job / "viewer.html").exists() \
        or (job / "points.bin").exists()
    return FileFlags(
        scene=scene,
        folder=job.exists(),
        preview=(job / "preview.html").exists(),
        replay=_replay_available(job),
    )


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
    root.geometry("860x740")
    root.minsize(700, 660)
    root.configure(fg_color=C_BG)

    # --- settings + the one explicit app state (echo_app.appstate) ---
    from echo_app import settings as settings_mod
    from echo_app.appstate import ActionContext, JobState, compute_actions
    from echo_app.version import APP_VERSION

    global JOBS_ROOT
    settings = settings_mod.load_settings()
    JOBS_ROOT = settings.jobs_root_path()

    state = {"job": None, "busy": False, "history": [],
             "depth_video": None, "panel_open": False,
             "job_state": JobState.IDLE, "busy_kind": None,
             "hidden": False, "progress_text": "空闲",
             "update_info": None, "update_checking": False,
             "update_downloading": False, "settings_win": None}
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
    # Update banner: stays packed away until a check actually finds a
    # newer version -- one calm line + one button, no nagging.
    update_box = ctk.CTkFrame(header, fg_color="transparent")
    update_label = ctk.CTkLabel(update_box, text="", font=F(12),
                                text_color=C_INK)
    update_label.pack(anchor="e", padx=(0, 2))
    btn_update = pill(update_box, "下载并安装", primary=True, font_size=12)
    btn_update.pack(anchor="e", pady=(4, 0))
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
    btn_replay = pill(hero_btn_row, "路线回放", font_size=12, state="disabled")
    btn_replay.pack(side="left", padx=(8, 0))

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
    btn_hist_replay = pill(hist_btn_row, "路线回放", font_size=11, state="disabled")
    btn_hist_replay.pack(side="left", padx=(6, 0))

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

    # Settings tile: a slim strip under the grid. The tile itself only
    # opens the settings view; the desc line mirrors where jobs go and
    # the running version so both are visible without opening anything.
    settings_tile = make_tile(3, 0, columnspan=2, pady=(8, 0))
    settings_parts = tile_header(
        settings_tile, "⚙", "设置",
        f"任务文件夹：{JOBS_ROOT} · 当前版本 v{APP_VERSION}")
    settings_desc = settings_parts[2]
    btn_settings = pill(settings_tile, "打开设置", font_size=12)
    btn_settings.pack(anchor="w", padx=14, pady=(0, 12))

    # --- UI helpers (Tk thread only) ------------------------------------
    def append_log(text):
        logbox.configure(state="normal")
        logbox.insert("end", text + "\n")
        logbox.see("end")
        logbox.configure(state="disabled")

    def log(text):
        """Thread-safe progress sink handed to run_pipeline."""
        ui_queue.put(("log", text))

    # --- tray + window lifecycle --------------------------------------
    # The tray is an optional subsystem: constructed and started under
    # guard, and if it never comes up the window simply behaves like
    # a plain app (close = quit) -- never an invisible window.
    from echo_app.tray import TrayController

    tray = TrayController(ui_queue,
                          icon_path=_resource_path("installer/echo.ico"))

    def set_progress_text(text):
        state["progress_text"] = text
        tray.set_status(text)

    def toast(title, message):
        """Completion toast, only while the window is hidden, and only
        ever from a background thread -- notify() itself never raises,
        and even a thread failure here is contained."""
        if not state["hidden"]:
            return

        def _go():
            try:
                from echo_app.notify import notify

                notify(title, message)
            except Exception:
                pass

        threading.Thread(target=_go, daemon=True).start()

    def show_window():
        state["hidden"] = False
        try:
            root.deiconify()
            root.lift()
            root.focus_force()
        except Exception:
            pass

    def hide_to_tray():
        if not tray.running:
            return False
        state["hidden"] = True
        try:
            root.withdraw()
        except Exception:
            state["hidden"] = False
            return False
        return True

    def quit_app(confirm=True):
        if confirm and state["job_state"] is JobState.RUNNING:
            if not messagebox.askyesno(
                    "任务还在进行",
                    "场景还在生成中，退出会中断它。要退出吗？"):
                return
        tray.stop()
        try:
            root.destroy()
        except Exception:
            pass

    def on_close():
        # WeChat behaviour: close hides to the tray (job continues)
        # when the user kept that setting AND the tray actually came
        # up; otherwise closing quits (with the same confirm).
        if settings.close_to_tray and hide_to_tray():
            return
        quit_app()

    def refresh_actions():
        """THE one place button states are set: everything derives
        from the explicit state + which files exist (echo_app.
        appstate.compute_actions). Handlers still guard themselves;
        this only keeps the visible state honest."""
        ctx = ActionContext(
            job_state=state["job_state"],
            current=job_file_flags(state["job"]),
            history=job_file_flags(selected_history_job()),
            depth_ready=bool(state["depth_video"]),
            update_checking=state["update_checking"],
            update_downloading=state["update_downloading"])
        acts = compute_actions(ctx)

        def on(widget, enabled):
            widget.configure(state="normal" if enabled else "disabled")

        from echo_app.appstate import (
            CHANGE_JOBS_FOLDER, CHECK_UPDATE, DOWNLOAD_UPDATE,
            HIST_FOLDER, HIST_OPEN, HIST_PREVIEW, HIST_REPLAY,
            OPEN_FOLDER, OPEN_PREVIEW, OPEN_REPLAY, OPEN_SCENE,
            PICK_DEPTH, PICK_VIDEO, PLAY_DEPTH)
        on(btn_pick, acts[PICK_VIDEO])
        on(btn_depth, acts[PICK_DEPTH])
        on(btn_depth_play, acts[PLAY_DEPTH])
        on(btn_open, acts[OPEN_SCENE])
        on(btn_folder, acts[OPEN_FOLDER])
        on(btn_preview, acts[OPEN_PREVIEW])
        on(btn_replay, acts[OPEN_REPLAY])
        on(btn_hist_open, acts[HIST_OPEN])
        on(btn_hist_folder, acts[HIST_FOLDER])
        on(btn_hist_preview, acts[HIST_PREVIEW])
        on(btn_hist_replay, acts[HIST_REPLAY])
        on(btn_update, acts[DOWNLOAD_UPDATE])
        widgets = state.get("settings_widgets") or {}
        if widgets.get("folder_btn") is not None:
            on(widgets["folder_btn"], acts[CHANGE_JOBS_FOLDER])
        if widgets.get("check_btn") is not None:
            on(widgets["check_btn"], acts[CHECK_UPDATE])

    def set_busy(busy, busy_text="正在做场景…"):
        state["busy"] = busy
        if busy:
            if state["job_state"] is not JobState.RUNNING:
                state["job_state"] = JobState.RUNNING
        hero_desc.configure(text=busy_text if busy else zone_idle)
        refresh_actions()

    def refresh_history():
        jobs = list_previous_jobs(JOBS_ROOT)
        state["history"] = jobs
        hist_list.delete(0, "end")
        for job in jobs:
            hist_list.insert("end", _job_display_name(job))
        refresh_actions()

    def open_job(job):
        state["job"] = job
        if job_finished(job):
            state["job_state"] = JobState.DONE
        btn_open.configure(text="重新打开场景")
        refresh_actions()
        ensure_and_open("scene", job)

    def handle_log(text):
        line = text.rstrip("…").strip()
        if line in PLAIN_STATUS:
            plain = PLAIN_STATUS[line]
            status.configure(text=plain + "…")
            frac = (list(PLAIN_STATUS).index(line) + 1) / STAGE_COUNT
            bar.set(frac)
            set_progress_text(f"{plain} · {round(frac * 100)}%")
            append_log(plain)
        elif line == "完成。":
            status.configure(text="做好了。")
            bar.set(1)
        else:
            append_log(text)

    def handle_done(job):
        state["job_state"] = JobState.DONE
        state["busy_kind"] = None
        set_busy(False)
        set_progress_text("已完成")
        refresh_history()
        open_job(job)
        status.configure(text=f"做好了。文件都在这个文件夹里：{job}")
        append_log(f"做好了。文件都在这个文件夹里：{job}")
        scale_line = scale_status_line(job)
        if scale_line:
            append_log(scale_line)
        for name in ("scene.html", "voxel_model.glb", "voxel_model.ply",
                     "mesh.ply", "scene_points.ply", "preview.html",
                     "replay.html"):
            p = job / name
            if p.exists():
                append_log(f"  {p}")
        toast("ECHO 场景生成器", "场景做好了，打开主窗口就能看。")

    def handle_error(detail):
        state["job_state"] = JobState.FAILED
        state["busy_kind"] = None
        set_busy(False)
        set_progress_text("上次任务未成功")
        bar.set(0)
        status.configure(text="没成功，这段视频没能做成场景。")
        append_log(detail)
        append_log("没成功，这段视频没能做成场景。")
        toast("ECHO 场景生成器", "这次没能做成场景，打开主窗口看一下。")
        if not state["hidden"]:
            messagebox.showerror("没成功", "这段视频没能做成场景，详情看日志。")

    def handle_depth_done(out):
        state["job_state"] = JobState.DONE
        state["busy_kind"] = None
        set_busy(False)
        set_progress_text("已完成")
        bar.stop()
        bar.configure(mode="determinate")
        bar.set(1)
        state["depth_video"] = out
        refresh_actions()
        status.configure(text="深度视频做好了，点“播放深度视频”就能看。")
        append_log(f"深度视频做好了：{out}")
        toast("ECHO 场景生成器", "深度视频做好了，打开主窗口就能看。")

    def handle_depth_error(detail):
        state["job_state"] = JobState.FAILED
        state["busy_kind"] = None
        set_busy(False)
        set_progress_text("上次任务未成功")
        bar.stop()
        bar.configure(mode="determinate")
        bar.set(0)
        status.configure(text="没成功，深度视频没做成。")
        append_log(detail)
        append_log("没成功，深度视频没做成。")
        toast("ECHO 场景生成器", "深度视频没做成，打开主窗口看一下。")
        if not state["hidden"]:
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
                elif kind == "tray_open":
                    show_window()
                elif kind == "tray_quit":
                    quit_app()
                elif kind == "play":
                    if payload and Path(payload).exists():
                        play_file(Path(payload))
                elif kind == "update_available":
                    handle_update_available(payload)
                elif kind == "update_none":
                    handle_update_none(payload)
                elif kind == "update_downloaded":
                    handle_update_downloaded(payload)
                elif kind == "update_download_failed":
                    handle_update_download_failed()
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
            return  # reentrancy guard: a second job can never start
        if not check_video(video):
            return
        bar.set(0)
        status.configure(text="开始了。")
        set_progress_text("开始了")
        state["busy_kind"] = "scene"
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
            return  # same guard as scene jobs: one thing at a time
        if not check_video(video):
            return
        state["depth_video"] = None
        status.configure(text="正在生成深度视频…")
        set_progress_text("正在生成深度视频")
        bar.configure(mode="indeterminate")
        bar.start()
        state["busy_kind"] = "depth"
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
        if out and out.exists():
            play_file(out)

    def ensure_and_open(kind, job):
        """Open a job's scene/replay page. When an older (or crashed)
        job doesn't have the page yet, it is built on a background
        thread first -- stage_present/stage_replay read the whole
        point cloud, which is real work and never belongs on the UI
        thread -- and played via the UI queue once it exists. Every
        path is guarded: no file and no way to build it = calm no-op,
        never an exception."""
        if job is None:
            return
        target = job / ("scene.html" if kind == "scene" else "replay.html")
        if target.exists():
            play_file(target)
            return
        if kind == "scene" and (job / "viewer.html").exists():
            play_file(job / "viewer.html")  # oldest jobs: page as-is
            return
        buildable = (job / "points.bin").exists() if kind == "scene" \
            else _replay_available(job)
        if not buildable:
            return

        def _build():
            try:
                if kind == "scene":
                    from echo_scene.pipeline import stage_present

                    stage_present(job)
                else:
                    from echo_scene.pipeline import stage_replay

                    stage_replay(job)
            except Exception:
                pass
            if target.exists():
                ui_queue.put(("play", target))

        threading.Thread(target=_build, daemon=True).start()

    def open_scene():
        ensure_and_open("scene", state["job"])

    def open_folder():
        job = state["job"]
        if job and job.exists():
            play_file(job)

    def open_preview():
        job = state["job"]
        if job and (job / "preview.html").exists():
            play_file(job / "preview.html")

    def open_replay():
        ensure_and_open("replay", state["job"])

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
        refresh_actions()

    def open_history(_event=None):
        job = selected_history_job()
        if job:
            open_job(job)
            status.configure(text=f"已经打开。文件在这个文件夹里：{job}")

    def open_history_folder():
        job = selected_history_job()
        if job and job.exists():
            play_file(job)

    def open_history_preview():
        job = selected_history_job()
        if job and (job / "preview.html").exists():
            play_file(job / "preview.html")

    def open_history_replay():
        ensure_and_open("replay", selected_history_job())

    # --- update check + download (background threads only) -----------
    def set_settings_update_text(text):
        widgets = state.get("settings_widgets") or {}
        label = widgets.get("update_label")
        if label is not None:
            try:
                label.configure(text=text)
            except Exception:
                pass

    def start_update_check(manual=False):
        if state["update_checking"] or state["update_downloading"]:
            return  # one check at a time; a double-click is harmless
        state["update_checking"] = True
        refresh_actions()
        if manual:
            set_settings_update_text("正在检查更新…")

        def _check():
            try:
                from echo_app.updater import check_for_update_status

                result = check_for_update_status(APP_VERSION)
            except Exception:
                result = ("error", None)
            status_word, info = result
            if status_word == "update":
                ui_queue.put(("update_available", info))
            else:
                ui_queue.put(("update_none", (manual, status_word)))

        threading.Thread(target=_check, daemon=True).start()

    def handle_update_available(info):
        state["update_checking"] = False
        state["update_info"] = info
        update_label.configure(text=f"有新版本 v{info.version}")
        if not update_box.winfo_manager():
            update_box.pack(side="right", padx=(0, 18), pady=(10, 12))
        set_settings_update_text(f"有新版本 v{info.version}。")
        append_log(f"有新版本 v{info.version}，可以点主窗口右上角的“下载并安装”。")
        refresh_actions()

    def handle_update_none(payload):
        manual, status_word = payload
        state["update_checking"] = False
        if manual:
            set_settings_update_text(
                "已是最新版本。" if status_word == "latest"
                else "没检查成，网络好像不通，过会儿再试。")
        refresh_actions()

    def start_update_download():
        info = state["update_info"]
        if info is None or state["update_downloading"]:
            return  # the double-fire guard: one download per offer
        state["update_downloading"] = True
        refresh_actions()
        status.configure(text=f"正在下载新版本 v{info.version}…")
        set_settings_update_text(f"正在下载 v{info.version}…")

        def _download():
            try:
                import tempfile

                from echo_app.updater import download_installer

                dest = Path(tempfile.gettempdir()) / "ECHO Update"
                path = download_installer(info, dest)
                ui_queue.put(("update_downloaded", path))
            except Exception:
                ui_queue.put(("update_download_failed", None))

        threading.Thread(target=_download, daemon=True).start()

    def handle_update_downloaded(path):
        state["update_downloading"] = False
        status.configure(text="下载好了，正在打开安装程序…")
        append_log(f"新版本安装程序已下载：{path}")
        try:
            subprocess.Popen([str(path)])
        except Exception:
            status.configure(text="安装程序没能打开，过会儿再试一次。")
            set_settings_update_text("安装程序没能打开，过会儿再试一次。")
            refresh_actions()
            return
        # The installer upgrades in place and closes the running app;
        # step aside ourselves so no file stays locked.
        quit_app(confirm=False)

    def handle_update_download_failed():
        state["update_downloading"] = False
        status.configure(text="新版本没下载成，网络恢复后再试。")
        set_settings_update_text("下载没成功，网络恢复后再试。")
        refresh_actions()

    # --- settings window ------------------------------------------------
    def open_settings():
        win = state["settings_win"]
        if win is not None:
            try:
                if win.winfo_exists():
                    win.lift()
                    win.focus_force()
                    return
            except Exception:
                pass
            state["settings_win"] = None
        win = ctk.CTkToplevel(root)
        state["settings_win"] = win
        win.title("设置")
        win.geometry("540x470")
        win.minsize(480, 430)
        win.configure(fg_color=C_BG)
        try:
            win.transient(root)
        except Exception:
            pass

        def card():
            frame = ctk.CTkFrame(win, fg_color=C_SURFACE,
                                 corner_radius=CORNER, border_width=1,
                                 border_color=C_BORDER)
            frame.pack(fill="x", padx=12, pady=(12, 0))
            return frame

        def card_title(frame, glyph, title):
            ctk.CTkLabel(frame, text=glyph, font=F(18),
                         text_color=C_ACCENT).pack(anchor="w", padx=14,
                                                    pady=(10, 0))
            ctk.CTkLabel(frame, text=title, font=F(14, True),
                         text_color=C_INK).pack(anchor="w", padx=14)

        # Jobs folder: view + change. The change applies to the NEXT
        # job (the button is disabled while one runs, and the note
        # says why); existing scenes are never moved or deleted.
        folder_card = card()
        card_title(folder_card, "▤", "任务文件夹")
        folder_label = ctk.CTkLabel(
            folder_card, text=str(JOBS_ROOT), font=F(11),
            text_color=C_SUBINK, justify="left", wraplength=480)
        folder_label.pack(anchor="w", padx=14, pady=(2, 0))
        folder_btn = pill(folder_card, "更改…", font_size=12)
        folder_btn.pack(anchor="w", padx=14, pady=(6, 2))
        ctk.CTkLabel(
            folder_card,
            text="新任务会存到新文件夹；已经做好的场景不会被移动，也不会被删除。"
                 "任务进行中时不能改，从下一次任务开始生效。",
            font=F(11), text_color=C_SUBINK, justify="left",
            wraplength=480).pack(anchor="w", padx=14, pady=(0, 12))

        def choose_jobs_folder():
            global JOBS_ROOT
            if state["job_state"] is JobState.RUNNING:
                return  # guarded here too, not only by the disabled state
            path = filedialog.askdirectory(
                title="选择任务文件夹", initialdir=str(JOBS_ROOT))
            if not path:
                return
            settings.jobs_root = "" if Path(path) == settings_mod.default_jobs_root() else path
            settings_mod.save_settings(settings)
            JOBS_ROOT = Path(path)
            folder_label.configure(text=str(JOBS_ROOT))
            settings_desc.configure(
                text=f"任务文件夹：{JOBS_ROOT} · 当前版本 v{APP_VERSION}")
            refresh_history()

        folder_btn.configure(command=choose_jobs_folder)

        # Behaviour switches: applied (and saved) the moment they flip.
        behaviour_card = card()
        card_title(behaviour_card, "✦", "使用习惯")
        autostart_var = tk.BooleanVar(value=settings.autostart)
        tray_var = tk.BooleanVar(value=settings.close_to_tray)

        def on_autostart_toggle():
            settings.autostart = bool(autostart_var.get())
            settings_mod.save_settings(settings)
            try:
                from echo_app.autostart import set_autostart

                set_autostart(settings.autostart)
            except Exception:
                pass

        def on_tray_toggle():
            settings.close_to_tray = bool(tray_var.get())
            settings_mod.save_settings(settings)

        ctk.CTkSwitch(
            behaviour_card, text="开机自启（启动后藏在系统托盘）",
            font=F(12), variable=autostart_var,
            command=on_autostart_toggle,
            progress_color=C_ACCENT).pack(anchor="w", padx=14, pady=(8, 2))
        ctk.CTkSwitch(
            behaviour_card, text="关闭窗口时最小化到系统托盘（任务继续）",
            font=F(12), variable=tray_var, command=on_tray_toggle,
            progress_color=C_ACCENT).pack(anchor="w", padx=14, pady=(2, 12))

        # Version + manual update check.
        update_card = card()
        card_title(update_card, "↻", "版本与更新")
        ctk.CTkLabel(update_card, text=f"当前版本 v{APP_VERSION}",
                     font=F(12), text_color=C_INK).pack(
                         anchor="w", padx=14, pady=(2, 0))
        check_btn = pill(update_card, "检查更新", font_size=12)
        check_btn.pack(anchor="w", padx=14, pady=(6, 2))
        update_result = ctk.CTkLabel(
            update_card, text="", font=F(11), text_color=C_SUBINK,
            justify="left", wraplength=480)
        update_result.pack(anchor="w", padx=14, pady=(0, 12))
        check_btn.configure(command=lambda: start_update_check(manual=True))

        state["settings_widgets"] = {
            "folder_btn": folder_btn, "check_btn": check_btn,
            "update_label": update_result}
        refresh_actions()

        def on_settings_destroy(_event=None):
            if state["settings_win"] is win:
                state["settings_win"] = None
                state["settings_widgets"] = {}

        win.bind("<Destroy>", on_settings_destroy)

    btn_pick.configure(command=pick)
    btn_open.configure(command=open_scene)
    btn_folder.configure(command=open_folder)
    btn_preview.configure(command=open_preview)
    btn_replay.configure(command=open_replay)
    btn_hist_folder.configure(command=open_history_folder)
    btn_hist_preview.configure(command=open_history_preview)
    btn_hist_replay.configure(command=open_history_replay)
    btn_markers.configure(command=print_markers)
    btn_depth.configure(command=pick_depth)
    btn_depth_play.configure(command=play_depth)
    btn_settings.configure(command=open_settings)
    btn_update.configure(command=start_update_download)
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

    root.protocol("WM_DELETE_WINDOW", on_close)
    refresh_history()
    refresh_actions()
    # The tray goes up before anything can hide to it; both the
    # close-to-tray behaviour and a --tray (autostart) launch check
    # tray.running first, so a tray that failed to start can never
    # strand an invisible window.
    tray.start()
    if settings.autostart:
        # Reconcile the sign-in entry with the saved preference -- an
        # upgrade or reinstall must not silently lose it. A tiny local
        # write, failure-contained inside set_autostart.
        try:
            from echo_app.autostart import set_autostart

            set_autostart(True)
        except Exception:
            pass
    if "--tray" in sys.argv:
        hide_to_tray()
    root.after(80, drain_queue)
    # Quiet startup update check: async, fail-silent offline; only a
    # found update ever becomes visible.
    root.after(1500, lambda: start_update_check(manual=False))
    root.mainloop()


if __name__ == "__main__":
    main()
