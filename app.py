#!/usr/bin/env python3
"""ECHO 场景生成器 desktop shell.

Layout (2026-11-01 redesign, on the user's direction): an identity
header band on top, the plain-language progress band under it, then
the working area — the 生成场景 drop tile and 历史场景 on the left,
and the big LIVE PREVIEW card on the right: while a job runs, the
scene grows in that card from the job's own intermediate products
(echo_app.live_preview drives it; pioneer C proved the cadence).
深度视频 / 打印标记纸 tiles sit under the columns and 设置 is a slim
strip at the bottom. The visual language follows the Muse app:
large-radius soft cards (24 px) on a pale grey window, pill buttons,
generous padding, and small eased motions (the progress bar glides,
preview frames crossfade, an idle breathing dot waits in the preview
card) — all animated with plain after() loops that never block the
UI thread. The shell is built with CustomTkinter; the 生成场景 tile
is the drop zone: drag a video onto it (or click to pick one) --
dropping only selects the file; 生成场景 or 深度视频 then starts
from the same selection -- watch the progress bar under the header,
and the finished scene
opens by itself. Detailed progress text goes to a collapsible
terminal panel docked on the right (the 日志 edge tab); the main
area only ever shows the plain-language status line. All heavy
lifting lives in echo_scene.pipeline / echo_scene.fuse; this file
only wires the stages together. A finished scene is plain files in
its job folder -- the standalone scene page, the model files, the
process preview, the route replay -- and the window says where they
are and opens them directly (no server, no localhost page; that was
the v0.1 design the user rejected).

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
import time
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


def run_pipeline(video: Path, jobs_root: Path, log, job_sink=None) -> Path:
    """Run every stage in order. log(str) receives human progress
    lines; job_sink(job_dir), when given, is called once the job
    folder exists (the live preview watches it from there)."""
    from echo_scene.pipeline import stage_depth, stage_extract, stage_present, stage_select
    from echo_scene.pipeline import stage_clean, stage_mesh, stage_poses, stage_preview, stage_replay, stage_scale, stage_voxel
    from echo_scene.fuse import stage_fuse

    job = jobs_root / f"job_{video.stem[:24]}"
    n = 1
    base = job
    while job.exists():
        n += 1
        job = jobs_root / f"{base.name}_{n}"
    job.mkdir(parents=True)
    if job_sink is not None:
        try:
            job_sink(job)
        except Exception:
            pass
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
    # Phone-capture imports (inbox selection or a drag-in of the
    # capture's own video.mp4) land the capture's sensor record inside
    # the job -- the same helper the CLI import uses -- so the fuse
    # stage can orient by IMU gravity. A plain video in an ordinary
    # folder is not a capture folder and imports exactly as before.
    try:
        from echo_scene import sensor_record as sr
        sr.import_capture_record(video.parent, job)
    except Exception:
        pass

    steps = [
        ("正在抽帧", lambda: stage_extract(video, job)),
        ("正在挑清晰的帧", lambda: stage_select(job)),
        ("正在算相机轨迹", lambda: stage_poses(job)),
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


def run_depth_video(video: Path, progress=None) -> Path:
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
    info = make_depth_video(video, out, progress=progress)
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
C_BG = "#E9EDF2"         # window surface (sits inside the frost tone)
C_SURFACE = "#FFFFFF"    # cards (header / tiles / log panel)
C_BORDER = "#DADDE1"     # subtle card borders
C_INK = "#1C1E21"        # primary text
C_SUBINK = "#606770"     # secondary text
C_ACCENT = "#0064D1"     # Meta blue
C_ACCENT_HOVER = "#0053AD"
C_BTN2 = "#E9EDF3"       # secondary pill buttons (pale glass)
C_BTN2_HOVER = "#DCE3EC"
C_TRACK = "#E4E6EB"      # progress bar track
C_LOG_BG = "#F4F6FA"     # log textbox surface
C_DRAG = "#E7F3FF"       # drag-over highlight
C_PREVIEW_BG = "#05070E"  # live preview card surface (Detroit dark)
CORNER = 24              # card corner radius (Muse-app scale)
CORNER_SM = 16           # inner surfaces

# Frontend skin variants (user review, 2026-11-02): how far every
# element melts into the frosted-glass wall. "now" = the look as
# shipped in the working tree; a/b/c are the review candidates.
SKINS = {
    "now": dict(corner=24, pane_milk=100, pane_blur=19, p_fg=C_ACCENT, p_hover=C_ACCENT_HOVER,
                p_text="#FFFFFF", s_fg=C_BTN2, s_hover=C_BTN2_HOVER,
                bar_h=10, track=C_TRACK),
    # A 轻透: glass pills -- pale blue primary, near-clear secondaries,
    # clearer panes, softer corners, hairline progress.
    "a": dict(corner=28, pane_milk=58, pane_blur=14, p_fg="#AAC7FA", p_hover="#97B8F7",
              p_text="#1E4FD6", s_fg="#EFF4FB", s_hover="#E1EBF9",
              bar_h=6, track="#DFE8F3"),
    # B 浮卡: tidy floating cards -- panes more present, buttons crisp
    # white glass, primary stays a confident solid blue.
    "b": dict(corner=20, pane_milk=118, pane_blur=16, p_fg=C_ACCENT, p_hover=C_ACCENT_HOVER,
              p_text="#FFFFFF", s_fg="#FAFCFF", s_hover="#EBF1FA",
              bar_h=8, track="#E3E9F2"),
    # C 极简: buttons unloaded entirely -- text resting on the glass,
    # panes barely there, the wall and its beads carry the window.
    "c": dict(corner=32, pane_milk=26, pane_blur=10, p_fg="transparent", p_hover="#E1EBF9",
              p_text=C_ACCENT, s_fg="transparent", s_hover="#E7EFFA",
              bar_h=3, track="#E6EDF6"),
}
SKIN = SKINS.get(os.environ.get("ECHO_SKIN", "a"), SKINS["now"])

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
    root.geometry("1020x800")
    root.minsize(860, 700)
    root.configure(fg_color=C_BG)

    # Frosted-glass skin (echo_app.glass): graffiti behind glass,
    # every card a translucent pane. Decoration only -- if it fails,
    # the window keeps its flat colours and nothing else changes.
    wall = None
    try:
        from echo_app.glass import GlassWall

        wall = GlassWall(root)
    except Exception:
        wall = None

    # --- settings + the one explicit app state (echo_app.appstate) ---
    from echo_app import settings as settings_mod
    from echo_app.appstate import ActionContext, JobState, compute_actions
    from echo_app.version import APP_VERSION

    global JOBS_ROOT
    settings = settings_mod.load_settings()
    JOBS_ROOT = settings.jobs_root_path()

    state = {"job": None, "busy": False, "history": [], "video": None,
             "depth_video": None, "panel_open": False,
             "job_state": JobState.IDLE, "busy_kind": None,
             "hidden": False, "progress_text": "空闲",
             "update_info": None, "update_checking": False,
             "update_downloading": False, "settings_win": None,
             "preview_stop": None, "preview_img": None,
             "preview_full_win": None, "preview_full_label": None,
             "preview_full_img": None, "preview_last_pil": None,
             "preview_render_size": (880, 520),
             "bar_target": 0.0, "bar_shown": 0.0}
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
                corner_radius=18, fg_color=SKIN["p_fg"],
                hover_color=SKIN["p_hover"], text_color=SKIN["p_text"],
                state=state)
        return ctk.CTkButton(
            master, text=text, font=F(font_size), height=36,
            corner_radius=18, fg_color=SKIN["s_fg"],
            hover_color=SKIN["s_hover"],
            border_width=0,
            text_color=C_INK, state=state)

    outer = ctk.CTkFrame(root, fg_color=C_BG, corner_radius=0)
    outer.pack(fill="both", expand=True)

    center = ctk.CTkFrame(outer, fg_color=C_BG, corner_radius=0)
    center.pack(side="left", fill="both", expand=True)

    # --- collapsible terminal panel (right edge, hidden by default) -----
    panel = ctk.CTkFrame(outer, fg_color=C_SURFACE, corner_radius=SKIN["corner"],
                         border_width=0, border_color=C_BORDER, width=340)
    panel.pack_propagate(False)
    if wall is not None:
        wall.attach(panel, name="log", milk=SKIN["pane_milk"], blur=SKIN["pane_blur"], radius=SKIN["corner"])
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
        if wall is not None:
            root.after(60, wall.refresh)

    btn_log.configure(command=toggle_panel)

    # --- header band ------------------------------------------------------
    header = ctk.CTkFrame(center, fg_color=C_SURFACE, corner_radius=SKIN["corner"],
                          border_width=0, border_color=C_BORDER)
    header.pack(fill="x", padx=8, pady=(10, 6))
    if wall is not None:
        wall.attach(header, name="header", milk=SKIN["pane_milk"], blur=SKIN["pane_blur"], radius=SKIN["corner"])
    # Update banner: stays packed away until a check actually finds a
    # newer version -- one calm line + one button, no nagging.
    update_box = ctk.CTkFrame(header, fg_color="transparent")
    update_label = ctk.CTkLabel(update_box, text="", font=F(12),
                                text_color=C_INK)
    update_label.pack(anchor="e", padx=(0, 2))
    btn_update = pill(update_box, "下载并安装", primary=True, font_size=12)
    btn_update.pack(anchor="e", pady=(4, 0))
    _hdr_title = ctk.CTkLabel(header, text="ECHO 回声", font=F(22, True),
                              text_color=C_INK)
    _hdr_title.pack(anchor="w", padx=18, pady=(12, 0))
    _hdr_sub = ctk.CTkLabel(header, text="视频变 3D 场景", font=F(12),
                            text_color=C_SUBINK)
    _hdr_sub.pack(anchor="w", padx=18, pady=(0, 12))
    if wall is not None:
        wall.textify(_hdr_title)
        wall.textify(_hdr_sub)
        wall.textify(update_label)

    # --- progress band (plain-language status only) -----------------------
    band = ctk.CTkFrame(center, fg_color=C_BG, corner_radius=0)
    band.pack(fill="x")
    if wall is not None:
        wall.attach(band, name="band", milk=SKIN["pane_milk"], blur=SKIN["pane_blur"], radius=14)
    status = ctk.CTkLabel(band, text="还没开始。", font=F(12),
                          text_color=C_INK)
    status.pack(anchor="w", padx=16, pady=(2, 0))
    if wall is not None:
        wall.textify(status)
    bar = ctk.CTkProgressBar(band, mode="determinate", height=SKIN["bar_h"],
                             corner_radius=5, fg_color=SKIN["track"],
                             progress_color=C_ACCENT)
    bar.set(0)
    bar.pack(fill="x", padx=16, pady=(4, 8))

    # --- tile grid ----------------------------------------------------------
    # Left column: drop tile + history. Right column: the live preview
    # card (the star of the redesign) spanning both rows; small tiles
    # under the columns; settings strip across the bottom.
    tiles = ctk.CTkFrame(center, fg_color=C_BG, corner_radius=0)
    tiles.pack(fill="both", expand=True, padx=8, pady=(2, 8))
    tiles.columnconfigure(0, weight=5)
    tiles.columnconfigure(1, weight=7)
    tiles.rowconfigure(1, weight=1)

    def make_tile(row, column, columnspan=1, rowspan=1,
                  padx=(0, 0), pady=(0, 0), glass=True, glass_name=None):
        tile = ctk.CTkFrame(tiles, fg_color=C_SURFACE, corner_radius=SKIN["corner"],
                            border_width=0, border_color=C_BORDER)
        tile.grid(row=row, column=column, columnspan=columnspan,
                  rowspan=rowspan, sticky="nsew", padx=padx, pady=pady)
        if wall is not None and glass:
            wall.attach(tile, name=glass_name, milk=SKIN["pane_milk"], blur=SKIN["pane_blur"], radius=SKIN["corner"])
        return tile

    _ICON_GLYPHS = {"spark": "✦", "list": "☰", "depth": "◐",
                    "grid": "▦", "gear": "⚙", "folder": "▤",
                    "refresh": "↻"}

    def icon_label(parent, key, size=26, dark=False, font_size=22):
        """A section icon drawn in the glass-bead style of the skin
        (echo_app.icons); falls back to the old text glyph if the
        drawn icon cannot be built."""
        try:
            from echo_app.icons import icon_pil

            img = ctk.CTkImage(icon_pil(key, size * 3, dark=dark),
                               size=(size, size))
            return ctk.CTkLabel(parent, text="", image=img)
        except Exception:
            return ctk.CTkLabel(
                parent, text=_ICON_GLYPHS.get(key, "✦"), font=F(font_size),
                text_color="#7FB2FF" if dark else C_ACCENT)

    def tile_header(tile, glyph, title, desc):
        parts = [
            icon_label(tile, glyph),
            ctk.CTkLabel(tile, text=title, font=F(14, True),
                         text_color=C_INK),
            ctk.CTkLabel(tile, text=desc, font=F(12), text_color=C_SUBINK,
                         justify="left"),
        ]
        if wall is not None:
            wall.textify(parts[1])
            wall.textify(parts[2])
        parts[0].pack(anchor="w", padx=14, pady=(12, 0))
        parts[1].pack(anchor="w", padx=14)
        parts[2].pack(anchor="w", padx=14, pady=(0, 6))
        return parts

    # Hero tile: the input zone. It IS the drop zone. Dropping or
    # picking a video only SELECTS it -- starting is a separate,
    # explicit choice between 生成场景 and 深度视频, so neither job
    # can swallow a file the user meant for the other.
    zone_idle = "把视频拖到这里，或者点「选择视频」。\n选好后点「开始生成」建模，或者先做一段深度视频。"
    hero = make_tile(0, 0, padx=(0, 4), pady=(2, 8), glass_name="hero")
    try:
        hero.configure(cursor="hand2")
    except Exception:
        pass
    hero_parts = tile_header(hero, "spark", "生成场景", zone_idle)
    hero_desc = hero_parts[2]
    hero_btn_row = ctk.CTkFrame(hero, fg_color="transparent")
    hero_btn_row.pack(anchor="w", padx=14, pady=(0, 4))
    btn_pick = pill(hero_btn_row, "选择视频", font_size=12)
    btn_pick.pack(side="left")
    btn_start_scene = pill(hero_btn_row, "开始生成", primary=True,
                           font_size=12, state="disabled")
    btn_start_scene.pack(side="left", padx=(8, 0))
    btn_hero_depth = pill(hero_btn_row, "深度视频", font_size=12,
                          state="disabled")
    btn_hero_depth.pack(side="left", padx=(8, 0))
    hero_out_row = ctk.CTkFrame(hero, fg_color="transparent")
    hero_out_row.pack(anchor="w", padx=14, pady=(0, 12))
    btn_open = pill(hero_out_row, "打开场景", font_size=12, state="disabled")
    btn_open.pack(side="left")
    btn_folder = pill(hero_out_row, "打开文件夹", font_size=12, state="disabled")
    btn_folder.pack(side="left", padx=(8, 0))
    btn_preview = pill(hero_out_row, "过程回放", font_size=12, state="disabled")
    btn_preview.pack(side="left", padx=(8, 0))
    btn_replay = pill(hero_out_row, "路线回放", font_size=12, state="disabled")
    btn_replay.pack(side="left", padx=(8, 0))

    def hero_bg(color):
        # CTk labels/rows are transparent, so tinting the card is
        # enough -- and the hero's glass pane tints along with it.
        hero.configure(fg_color=color)
        if wall is not None:
            try:
                wall.tint("hero", None if color == C_SURFACE else color)
            except Exception:
                pass

    # Live preview card: while a job runs, the scene grows here from
    # the job's own intermediate products (echo_app.live_preview).
    # Idle, it waits with one calm line and a breathing dot.
    preview_tile = make_tile(0, 1, rowspan=2, padx=(4, 0), pady=(2, 8),
                             glass=False)
    preview_tile.configure(fg_color=C_PREVIEW_BG, border_width=0)
    preview_head = ctk.CTkFrame(preview_tile, fg_color="transparent")
    preview_head.pack(fill="x", padx=16, pady=(12, 0))
    icon_label(preview_head, "spark", size=20, dark=True).pack(side="left")
    ctk.CTkLabel(preview_head, text="实时预览", font=F(14, True),
                 text_color="#E8EDF5").pack(side="left", padx=(6, 0))
    # The breathing dot is a little pearl in the same bead style:
    # a dim bead and a bright bead, swapped by the breath tick.
    dot_imgs = {}
    try:
        from echo_app.icons import icon_pil as _icon_pil

        dot_imgs = {
            "dim": ctk.CTkImage(_icon_pil("dot_dim", 36), size=(12, 12)),
            "bright": ctk.CTkImage(_icon_pil("dot_bright", 36),
                                   size=(12, 12)),
        }
        preview_dot = ctk.CTkLabel(preview_head, text="",
                                   image=dot_imgs["dim"])
    except Exception:
        dot_imgs = {}
        preview_dot = ctk.CTkLabel(preview_head, text="●", font=F(10),
                                   text_color="#3D5A80")
    dot_state = {"bright": False}
    preview_dot.pack(side="left", padx=(8, 0))
    btn_preview_full = ctk.CTkButton(
        preview_head, text="全屏", font=F(11), height=28, width=64,
        corner_radius=14, fg_color="#1B2436", hover_color="#26334D",
        text_color="#E8EDF5")
    btn_preview_full.pack(side="right")
    preview_img_label = ctk.CTkLabel(preview_tile, text="", fg_color="transparent")
    preview_img_label.pack(fill="both", expand=True, padx=10, pady=(6, 4))
    preview_caption = ctk.CTkLabel(
        preview_tile, text="场景会在这里长出来", font=F(11),
        text_color="#8A94A6")
    preview_caption.pack(anchor="w", padx=16, pady=(0, 12))

    # History tile: previous scenes, each reopenable.
    hist_tile = make_tile(1, 0, padx=(0, 4))
    tile_header(hist_tile, "list", "历史场景", "以前做好的场景，选一个点“打开”。")
    hist_list = tk.Listbox(hist_tile, height=4, font=FT(11), relief="flat",
                           borderwidth=0, highlightthickness=0,
                           bg="#EDF1F6", fg=C_INK,
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
    depth_tile = make_tile(2, 0, padx=(0, 4))
    tile_header(depth_tile, "depth", "深度视频", "把选好的视频做成黑白的深度画面；还没选就先挑一段。")
    depth_btn_row = ctk.CTkFrame(depth_tile, fg_color="transparent")
    depth_btn_row.pack(anchor="w", padx=14, pady=(2, 12))
    btn_depth = pill(depth_btn_row, "深度视频", primary=True, font_size=12)
    btn_depth.pack(side="left")
    btn_depth_play = pill(depth_btn_row, "播放深度视频", font_size=12,
                          state="disabled")
    btn_depth_play.pack(side="left", padx=(8, 0))

    # Settings tile: a slim strip under the grid. The tile itself only
    # opens the settings view; the desc line mirrors where jobs go and
    # the running version so both are visible without opening anything.
    settings_tile = make_tile(3, 0, columnspan=2, pady=(8, 0))
    settings_parts = tile_header(
        settings_tile, "gear", "设置",
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

    # --- motion: eased progress bar + breathing dot (UI thread) -------
    def _bar_tick():
        if state["busy_kind"] != "depth":
            shown, target = state["bar_shown"], state["bar_target"]
            if abs(target - shown) > 0.0015:
                shown += (target - shown) * 0.22
                state["bar_shown"] = shown
                try:
                    bar.set(shown)
                except Exception:
                    pass
        root.after(40, _bar_tick)

    def _breath_tick():
        try:
            if dot_imgs:
                bright = True if state["busy"] else not dot_state["bright"]
                dot_state["bright"] = bright
                preview_dot.configure(
                    image=dot_imgs["bright" if bright else "dim"])
            elif state["busy"]:
                preview_dot.configure(text_color="#7FB2FF")
            else:
                cur = preview_dot.cget("text_color")
                preview_dot.configure(
                    text_color="#7FB2FF" if cur == "#3D5A80" else "#3D5A80")
        except Exception:
            pass
        root.after(800, _breath_tick)

    def set_bar(frac):
        """Move the progress bar, gliding (scene jobs) — the tick loop
        does the easing; instant jumps set both ends of the glide."""
        state["bar_target"] = frac
        if frac in (0.0, 1.0):
            state["bar_shown"] = frac
            try:
                bar.set(frac)
            except Exception:
                pass

    # --- live preview engine (background thread, failure-contained) ---
    def preview_view():
        """The one shared camera state (echo_app.live_preview.
        ViewState): the fullscreen view manipulates it, the card
        inherits it on exit, the next job starts from it."""
        view = state.get("preview_view")
        if view is None:
            try:
                from echo_app.live_preview import ViewState

                view = ViewState()
                state["preview_view"] = view
            except Exception:
                return None
        return view

    def start_preview(job):
        stop_preview()
        stop_flag = threading.Event()
        state["preview_stop"] = stop_flag
        try:
            preview_caption.configure(text="正在看着它长出来…")
        except Exception:
            pass

        def _run():
            try:
                from PIL import Image

                from echo_app.live_preview import (
                    JobWatcher, LiveSequencer, render_snapshot)
                watcher = JobWatcher(job, stable_polls=2)
                seq = LiveSequencer(job, view=preview_view())
                prev_img = None
                last_render = 0.0
                last_signature = None
                while not stop_flag.is_set():
                    events = watcher.poll()
                    if events:
                        seq.consume(events)
                    view = state.get("preview_view")
                    signature = (seq.state.phase, seq.state.frames_seen,
                                 len(seq.state.reliefs),
                                 seq.state.settled is not None,
                                 view.version if view is not None else 0)
                    now = time.monotonic()
                    if signature != last_signature or now - last_render > 3.0:
                        rw, rh = state.get("preview_render_size",
                                           (880, 520))
                        if state.get("preview_dragging") \
                                and state.get("preview_full_win") is not None:
                            rw, rh = 1066, 600  # lighter while dragging
                        img, _dt = render_snapshot(seq, width=rw,
                                                   height=rh)
                        frame = Image.fromarray(img)
                        if prev_img is not None \
                                and prev_img.size == frame.size:
                            for alpha in (0.35, 0.7):
                                ui_queue.put(("preview_frame",
                                              Image.blend(prev_img, frame,
                                                          alpha)))
                        ui_queue.put(("preview_frame", frame))
                        prev_img = frame
                        last_render = now
                        last_signature = signature
                    active = state.get("preview_dragging") \
                        or state.get("preview_full_win") is not None
                    stop_flag.wait(0.15 if active else 0.7)
            except Exception:
                pass  # optional subsystem: it can never take a job down

        threading.Thread(target=_run, daemon=True).start()

    def stop_preview():
        flag = state.get("preview_stop")
        if flag is not None:
            try:
                flag.set()
            except Exception:
                pass
            state["preview_stop"] = None

    def handle_job_started(job):
        if state["busy_kind"] == "scene":
            start_preview(job)

    def handle_preview_frame(frame):
        state["preview_last_pil"] = frame
        try:
            w = max(preview_img_label.winfo_width() - 4, 200)
            h = max(preview_img_label.winfo_height() - 4, 120)
            cimg = ctk.CTkImage(light_image=frame, dark_image=frame,
                                size=(w, h))
            state["preview_img"] = cimg
            preview_img_label.configure(image=cimg, text="")
        except Exception:
            pass
        full_label = state.get("preview_full_label")
        if full_label is not None:
            try:
                sw = full_label.winfo_screenwidth()
                sh = full_label.winfo_screenheight()
                fimg = ctk.CTkImage(light_image=frame, dark_image=frame,
                                    size=(sw, sh))
                state["preview_full_img"] = fimg
                full_label.configure(image=fimg, text="")
            except Exception:
                pass

    # --- live preview fullscreen (pure view; Esc / double-click out) ---
    def open_preview_fullscreen():
        if state.get("preview_full_win") is not None:
            return
        try:
            win = ctk.CTkToplevel(root)
            win.configure(fg_color=C_PREVIEW_BG)
            try:
                win.attributes("-fullscreen", True)
            except Exception:
                try:
                    win.state("zoomed")
                except Exception:
                    pass
            try:
                # Explicit full-screen geometry too: the attribute is a
                # window-manager request, and where no manager honours
                # it the window would otherwise open small.
                win.geometry("%dx%d+0+0" % (win.winfo_screenwidth(),
                                            win.winfo_screenheight()))
            except Exception:
                pass
            lbl = ctk.CTkLabel(win, text="场景会在这里长出来",
                               font=F(14), text_color="#8A94A6",
                               fg_color="transparent")
            lbl.pack(fill="both", expand=True)
            try:
                lbl.configure(cursor="fleur")
            except Exception:
                pass
            hint = ctk.CTkLabel(
                win, text="拖动转视角 · 滚轮缩放 · 右键拖动平移 · 按 0 回正 · Esc 退出全屏",
                font=F(11), text_color="#5A6B84", fg_color="transparent")
            hint.place(relx=1.0, rely=1.0, x=-16, y=-12, anchor="se")
            lbl.bind("<Double-Button-1>",
                     lambda _e: close_preview_fullscreen())
            win.bind("<Escape>", lambda _e: close_preview_fullscreen())

            # Mouse control over the shared ViewState: whatever the
            # user does here is still the camera when fullscreen
            # closes — the card (and the next entry) inherit it.
            drag = {"x": 0, "y": 0, "mode": None}

            def _press(mode):
                def handler(e):
                    drag["x"], drag["y"], drag["mode"] = e.x, e.y, mode
                    state["preview_dragging"] = True
                return handler

            def _motion(e):
                if drag["mode"] is None:
                    return
                dx, dy = e.x - drag["x"], e.y - drag["y"]
                drag["x"], drag["y"] = e.x, e.y
                view = preview_view()
                if view is None:
                    return
                if drag["mode"] == "orbit":
                    view.drag(dx, dy)
                else:
                    view.pan_drag(dx, dy)

            def _release(_e=None):
                drag["mode"] = None
                state["preview_dragging"] = False

            def _wheel(e):
                view = preview_view()
                if view is not None:
                    view.zoom(1 if (getattr(e, "delta", 0) > 0
                                    or getattr(e, "num", 0) == 4) else -1)

            def _reset(_e=None):
                view = preview_view()
                if view is not None:
                    view.reset()

            lbl.bind("<ButtonPress-1>", _press("orbit"))
            lbl.bind("<B1-Motion>", _motion)
            lbl.bind("<ButtonRelease-1>", _release)
            lbl.bind("<ButtonPress-3>", _press("pan"))
            lbl.bind("<B3-Motion>", _motion)
            lbl.bind("<ButtonRelease-3>", _release)
            lbl.bind("<MouseWheel>", _wheel)
            lbl.bind("<Button-4>", _wheel)
            lbl.bind("<Button-5>", _wheel)
            win.bind("<KeyPress-0>", _reset)
            state["preview_full_win"] = win
            state["preview_full_label"] = lbl
            state["preview_render_size"] = (1600, 900)
            win.focus_force()
            last = state.get("preview_last_pil")
            if last is not None:
                handle_preview_frame(last)
            else:
                try:
                    preview_caption.configure(text="场景会在这里长出来")
                except Exception:
                    pass

            def _on_destroy(_e=None, _win=win):
                if state.get("preview_full_win") is _win:
                    state["preview_full_win"] = None
                    state["preview_full_label"] = None
                    state["preview_render_size"] = (880, 520)

            win.bind("<Destroy>", _on_destroy)
        except Exception:
            state["preview_full_win"] = None
            state["preview_full_label"] = None

    def close_preview_fullscreen():
        win = state.get("preview_full_win")
        state["preview_dragging"] = False
        state["preview_full_win"] = None
        state["preview_full_label"] = None
        state["preview_render_size"] = (880, 520)
        if win is not None:
            try:
                win.destroy()
            except Exception:
                pass

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
            video_selected=bool(state["video"]),
            update_checking=state["update_checking"],
            update_downloading=state["update_downloading"])
        acts = compute_actions(ctx)

        def on(widget, enabled):
            widget.configure(state="normal" if enabled else "disabled")

        from echo_app.appstate import (
            CHANGE_JOBS_FOLDER, CHECK_UPDATE, DOWNLOAD_UPDATE,
            HIST_FOLDER, HIST_OPEN, HIST_PREVIEW, HIST_REPLAY,
            OPEN_FOLDER, OPEN_PREVIEW, OPEN_REPLAY, OPEN_SCENE,
            PICK_DEPTH, PICK_VIDEO, PLAY_DEPTH, START_DEPTH, START_SCENE)
        on(btn_pick, acts[PICK_VIDEO])
        on(btn_start_scene, acts[START_SCENE])
        on(btn_hero_depth, acts[START_DEPTH])
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
            set_bar(frac)
            set_progress_text(f"{plain} · {round(frac * 100)}%")
            append_log(plain)
        elif line == "完成。":
            status.configure(text="做好了。")
            set_bar(1)
        else:
            append_log(text)

    def handle_done(job):
        state["job_state"] = JobState.DONE
        state["busy_kind"] = None
        set_busy(False)
        set_progress_text("已完成")
        try:
            preview_caption.configure(text="做好了，这是它最后的样子。")
        except Exception:
            pass
        root.after(3000, stop_preview)
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
        stop_preview()
        set_bar(0)
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
                elif kind == "job_started":
                    handle_job_started(payload)
                elif kind == "preview_frame":
                    handle_preview_frame(payload)
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
            job = run_pipeline(
                video, JOBS_ROOT, log,
                job_sink=lambda j: ui_queue.put(("job_started", j)))
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

    def select_video(path: Path):
        """Choosing a video only selects it. Nothing starts here:
        the user then picks 生成场景 or 深度视频 for the same file."""
        if state["busy"]:
            return
        if not check_video(path):
            return
        state["video"] = path
        hero_desc.configure(
            text=f"已选：{path.name}\n点「开始生成」建模；也可以先做一段深度视频。")
        status.configure(text="视频选好了。点「开始生成」，或者做深度视频。")
        append_log(f"已选视频：{path.name}")
        refresh_actions()

    def start(video: Path = None):
        video = video or state["video"]
        if video is None or state["busy"]:
            return  # reentrancy guard: a second job can never start
        if not check_video(video):
            return
        set_bar(0)
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
            select_video(Path(path))

    def start_depth(video: Path = None):
        video = video or state["video"]
        if video is None or state["busy"]:
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
        if state["video"] is not None:
            start_depth(state["video"])
            return
        path = filedialog.askopenfilename(
            title="选择视频（做深度视频）",
            filetypes=[("视频", "*.mp4 *.mov *.mkv *.avi"), ("所有文件", "*.*")],
        )
        if path:
            select_video(Path(path))
            start_depth(state["video"])

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
                                 corner_radius=SKIN["corner"], border_width=0,
                                 border_color=C_BORDER)
            frame.pack(fill="x", padx=12, pady=(12, 0))
            return frame

        def card_title(frame, glyph, title):
            icon_label(frame, glyph, size=22,
                       font_size=18).pack(anchor="w", padx=14, pady=(10, 0))
            ctk.CTkLabel(frame, text=title, font=F(14, True),
                         text_color=C_INK).pack(anchor="w", padx=14)

        # Jobs folder: view + change. The change applies to the NEXT
        # job (the button is disabled while one runs, and the note
        # says why); existing scenes are never moved or deleted.
        folder_card = card()
        card_title(folder_card, "folder", "任务文件夹")
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
        card_title(behaviour_card, "spark", "使用习惯")
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
        card_title(update_card, "refresh", "版本与更新")
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
    btn_start_scene.configure(command=lambda: start())
    btn_hero_depth.configure(command=lambda: start_depth())
    btn_open.configure(command=open_scene)
    btn_folder.configure(command=open_folder)
    btn_preview.configure(command=open_preview)
    btn_replay.configure(command=open_replay)
    btn_hist_folder.configure(command=open_history_folder)
    btn_hist_preview.configure(command=open_history_preview)
    btn_hist_replay.configure(command=open_history_replay)
    btn_depth.configure(command=pick_depth)
    btn_depth_play.configure(command=play_depth)
    btn_settings.configure(command=open_settings)
    btn_update.configure(command=start_update_download)
    btn_hist_open.configure(command=open_history)
    btn_preview_full.configure(command=open_preview_fullscreen)
    preview_img_label.bind("<Double-Button-1>",
                           lambda _e: open_preview_fullscreen())
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
        select_video(Path(paths[0]))

    if dnd_ready:
        # The hero's glass pane covers the tile, so drops land on it:
        # it registers as a drop target alongside the tile itself.
        hero_glass = []
        if wall is not None and "hero" in wall.labels:
            hero_glass = [wall.labels["hero"]]
        for widget in [hero, root] + hero_parts + hero_glass:
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", on_drop)
        for widget in [hero] + hero_parts + hero_glass:
            widget.dnd_bind("<<DragEnter>>", lambda _e: hero_bg(C_DRAG))
            widget.dnd_bind("<<DragLeave>>", lambda _e: hero_bg(C_SURFACE))

    root.protocol("WM_DELETE_WINDOW", on_close)
    refresh_history()
    refresh_actions()
    if wall is not None:
        # First panes once the layout settles, then re-hang them
        # (debounced) whenever the window changes size.
        root.after(90, wall.refresh)
        _glass_pending = {"id": None}

        def _glass_on_configure(event):
            if event.widget is not root:
                return
            if _glass_pending["id"] is not None:
                try:
                    root.after_cancel(_glass_pending["id"])
                except Exception:
                    pass
            _glass_pending["id"] = root.after(320, wall.refresh)

        root.bind("<Configure>", _glass_on_configure, add="+")
        # The graffiti comes alive behind the glass: scattered into
        # 3D beads that dodge the mouse while the app sits idle,
        # frozen flat while a job owns the machine. Decoration only.
        try:
            wall.start_animation(lambda: bool(state["busy"]))
        except Exception:
            pass
        # Text on glass must not sit on a box: hand the wall every
        # label/textbox living on a glass frame, so each one's
        # background is matched to the glass directly beneath it.
        try:
            def _collect(frame, out):
                for child in frame.winfo_children():
                    if isinstance(child, ctk.CTkTextbox):
                        out.append(child)
                    elif isinstance(child, ctk.CTkLabel):
                        try:
                            if child.cget("text"):
                                out.append(child)
                        except Exception:
                            pass
                    _collect(child, out)

            tracked = [hist_list]
            for fr in (header, band, hero, hist_tile, depth_tile,
                       settings_tile, panel):
                _collect(fr, tracked)
            wall.track_labels(tracked)
            wall.track_grout([outer, center, tiles])
        except Exception:
            pass
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
    root.after(40, _bar_tick)
    root.after(800, _breath_tick)
    # Quiet startup update check: async, fail-silent offline; only a
    # found update ever becomes visible.
    root.after(1500, lambda: start_update_check(manual=False))
    root.mainloop()


if __name__ == "__main__":
    main()
