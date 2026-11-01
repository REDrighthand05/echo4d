"""Render Studio — window.

A Toplevel that directs one finished job's build process like a tiny
video editor: a dark preview (the same Detroit-dark surface as the
live preview card), a draggable timeline with attachment bands and
slow segments, element switches, a crush-amplitude keyframer, and an
export that renders the directed timeline to an mp4 in the job
folder. Playback renders on a worker thread (latest request wins)
and crossfades into the preview, matching the main window's motion
language; seeks glide instead of jumping.

The window owns only its own small state; the project itself lives
in echo_app.studio.StudioProject and persists as studio.json in the
job folder.
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import customtkinter as ctk
import numpy as np
from PIL import Image

from .studio import ATTACH_MODES, StudioProject
from .studio_render import StudioRenderer, export_mp4

PREVIEW_W, PREVIEW_H = 768, 432
MODE_LABEL = {"route": "跟随路线", "photo": "跟随照片", "orbit": "环视"}
MODE_COLOR = {"route": "#0064D1", "photo": "#0E7C86", "orbit": "#B0B3B8"}


class StudioWindow(ctk.CTkToplevel):
    def __init__(self, master, job_dir: Path, style: dict):
        super().__init__(master)
        self.job_dir = Path(job_dir)
        self.style = style
        F = style["font"]
        self.title(f"渲染 Studio — {self.job_dir.name}")
        self.geometry("1020x760")
        self.minsize(880, 640)
        self.configure(fg_color=style["bg"])

        self.project = StudioProject.load(self.job_dir)
        self.renderer = StudioRenderer(self.job_dir, self.project)
        self.t_out = 0.0            # playhead, output seconds
        self.t_shown = 0.0          # gliding display position
        self.playing = False
        self._sel = None            # (t0, t1) source seconds or None
        self._drag_x = None
        self._last_tick = time.monotonic()
        self._render_busy = False
        self._render_want = None
        self._ui_queue = queue.Queue()   # worker threads never touch tk
        self._current_img = None
        self._exporting = False

        # -- layout ------------------------------------------------------
        card = ctk.CTkFrame(self, fg_color=style["surface"],
                            corner_radius=style["corner"])
        card.pack(fill="x", padx=14, pady=(14, 8))
        ctk.CTkLabel(card, text="渲染 Studio", font=F(16, True),
                     text_color=style["ink"]).pack(
                         side="left", padx=16, pady=12)
        ctk.CTkLabel(card, text="把这次重建的过程，导成你想要的样子",
                     font=F(12), text_color=style["subink"]).pack(
                         side="left")
        self.btn_save = self._pill(card, "保存项目", primary=False)
        self.btn_save.pack(side="right", padx=16, pady=12)
        self.btn_save.configure(command=self._save)

        prev = ctk.CTkFrame(self, fg_color=style["preview_bg"],
                            corner_radius=style["corner"])
        prev.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        self.img_label = ctk.CTkLabel(prev, text="")
        self.img_label.pack(fill="both", expand=True, padx=10, pady=10)
        self.caption = ctk.CTkLabel(prev, text="…", font=F(11),
                                    text_color="#8A94A6")
        self.caption.place(relx=0.0, rely=1.0, x=18, y=-14, anchor="sw")

        tl = ctk.CTkFrame(self, fg_color=style["surface"],
                          corner_radius=style["corner"])
        tl.pack(fill="x", padx=14, pady=(0, 8))
        import tkinter as tk
        self.canvas = tk.Canvas(tl, height=66, bg=style["surface"],
                                highlightthickness=0, relief="flat")
        self.canvas.pack(fill="x", padx=14, pady=(12, 4))
        self.canvas.bind("<Configure>", lambda _e: self._draw_timeline())
        self.canvas.bind("<Button-1>", self._tl_down)
        self.canvas.bind("<B1-Motion>", self._tl_move)
        self.canvas.bind("<ButtonRelease-1>", self._tl_up)
        row = ctk.CTkFrame(tl, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=(0, 12))
        self.btn_play = self._pill(row, "播放", primary=True)
        self.btn_play.pack(side="left")
        self.btn_play.configure(command=self._toggle_play)
        for mode in ATTACH_MODES:
            b = self._pill(row, MODE_LABEL[mode])
            b.pack(side="left", padx=(8, 0))
            b.configure(command=lambda m=mode: self._apply_attach(m))
        self.btn_slow = self._pill(row, "选段慢动作 ×2")
        self.btn_slow.pack(side="left", padx=(8, 0))
        self.btn_slow.configure(command=self._apply_slow)
        self.btn_unslow = self._pill(row, "取消慢动作")
        self.btn_unslow.pack(side="left", padx=(8, 0))
        self.btn_unslow.configure(command=lambda: self._apply_slow(clear=True))

        ctl = ctk.CTkFrame(self, fg_color=style["surface"],
                           corner_radius=style["corner"])
        ctl.pack(fill="x", padx=14, pady=(0, 14))
        self.switches = {}
        for key, label in (("grid", "场景网格"), ("route", "路线"),
                           ("photos", "照片")):
            sw = ctk.CTkSwitch(ctl, text=label, font=F(12),
                               progress_color=style["accent"])
            sw.pack(side="left", padx=(16, 4), pady=12)
            if self.project.elements.get(key, True):
                sw.select()
            sw.configure(command=lambda k=key: self._toggle_element(k))
            self.switches[key] = sw
        ctk.CTkLabel(ctl, text="爆碎幅度", font=F(12),
                     text_color=style["ink"]).pack(side="left",
                                                   padx=(18, 4))
        self.crush = ctk.CTkSlider(ctl, from_=0.0, to=1.5, width=150,
                                   progress_color=style["accent"])
        self.crush.set(self.project.crush_at(0.0))
        self.crush.pack(side="left")
        self.crush.configure(command=self._crush_changed)
        self.btn_export = self._pill(ctl, "导出视频", primary=True)
        self.btn_export.pack(side="right", padx=16)
        self.btn_export.configure(command=self._export)
        self.status = ctk.CTkLabel(ctl, text="", font=F(11),
                                   text_color=style["subink"])
        self.status.pack(side="right", padx=(0, 6))

        self._draw_timeline()
        self._request_render()
        self.after(50, self._tick)

    # -- helpers ---------------------------------------------------------
    def _pill(self, master, text, primary=False):
        st = self.style
        if primary:
            return ctk.CTkButton(master, text=text,
                                 font=st["font"](12), height=36,
                                 corner_radius=18, fg_color=st["accent"],
                                 hover_color=st["accent_hover"],
                                 text_color="#FFFFFF")
        return ctk.CTkButton(master, text=text, font=st["font"](12),
                             height=36, corner_radius=18,
                             fg_color=st["btn2"],
                             hover_color=st["btn2_hover"],
                             text_color=st["ink"])

    def _save(self):
        path = self.project.save(self.job_dir)
        self.status.configure(text=f"已保存 {path.name}")

    # -- timeline ----------------------------------------------------------
    def _tl_geom(self):
        w = max(self.canvas.winfo_width(), 100)
        return 8, w - 8

    def _x_to_source(self, x):
        a, b = self._tl_geom()
        frac = min(max((x - a) / max(b - a, 1), 0.0), 1.0)
        return frac * self.project.duration

    def _source_to_x(self, s):
        a, b = self._tl_geom()
        return a + (s / max(self.project.duration, 1e-6)) * (b - a)

    def _draw_timeline(self):
        c = self.canvas
        c.delete("all")
        a, b = self._tl_geom()
        dur = self.project.duration
        # attachment bands (top row)
        c.create_rectangle(a, 8, b, 30, fill="#EDEFF2", outline="")
        cursor = 0.0
        segs = sorted(self.project.attach, key=lambda s: s.t0)
        spans = []
        for seg in segs:
            if seg.t0 > cursor:
                spans.append((cursor, seg.t0, "orbit"))
            spans.append((seg.t0, seg.t1, seg.mode))
            cursor = max(cursor, seg.t1)
        if cursor < dur:
            spans.append((cursor, dur, "orbit"))
        for t0, t1, mode in spans:
            x0, x1 = self._source_to_x(t0), self._source_to_x(t1)
            c.create_rectangle(x0, 8, x1, 30, fill=MODE_COLOR[mode],
                               outline="")
            if x1 - x0 > 58:
                c.create_text((x0 + x1) / 2, 19, text=MODE_LABEL[mode],
                              fill="#FFFFFF" if mode != "orbit" else "#4B4F56",
                              font=("TkDefaultFont", 9))
        # slow segments (bottom row)
        c.create_rectangle(a, 38, b, 56, fill="#F5F6F8", outline="")
        for seg in self.project.slow:
            x0, x1 = self._source_to_x(seg.t0), self._source_to_x(seg.t1)
            c.create_rectangle(x0, 38, x1, 56, fill="#F2C94C", outline="")
            if x1 - x0 > 40:
                c.create_text((x0 + x1) / 2, 47, text=f"慢 ×{seg.factor:g}",
                              fill="#5A4A00", font=("TkDefaultFont", 9))
        # event ticks
        for i in range(len(self.renderer.events)):
            x = self._source_to_x(self.renderer.event_time(i))
            c.create_line(x, 60, x, 65, fill="#C4C9D1")
        # selection
        if self._sel:
            x0, x1 = (self._source_to_x(self._sel[0]),
                      self._source_to_x(self._sel[1]))
            c.create_rectangle(x0, 4, x1, 60, outline=self.style["accent"],
                               width=2)
        # playhead (shown position, glides)
        x = self._source_to_x(self.project.to_source(self.t_shown))
        c.create_line(x, 4, x, 62, fill=self.style["accent"], width=2)
        c.create_polygon(x - 5, 0, x + 5, 0, x, 7,
                         fill=self.style["accent"], outline="")

    def _tl_down(self, event):
        self._drag_x = event.x
        self._sel = None

    def _tl_move(self, event):
        if self._drag_x is None:
            return
        if abs(event.x - self._drag_x) > 6:
            s0 = self._x_to_source(self._drag_x)
            s1 = self._x_to_source(event.x)
            self._sel = (min(s0, s1), max(s0, s1))
            self._draw_timeline()

    def _tl_up(self, event):
        if self._drag_x is not None and self._sel is None:
            s = self._x_to_source(event.x)
            self.t_out = self.project.to_output(s)
            self.playing = False
            self.btn_play.configure(text="播放")
        self._drag_x = None
        self._draw_timeline()
        self._request_render()

    def _range(self):
        if self._sel and self._sel[1] - self._sel[0] > 0.05:
            return self._sel
        return (0.0, self.project.duration)

    def _apply_attach(self, mode):
        t0, t1 = self._range()
        self.project.set_attach(t0, t1, mode)
        self._draw_timeline()
        self._request_render()

    def _apply_slow(self, clear=False):
        t0, t1 = self._range()
        self.project.set_slow(t0, t1, None if clear else 2.0)
        self._draw_timeline()

    # -- controls ------------------------------------------------------------
    def _toggle_element(self, key):
        self.project.elements[key] = bool(self.switches[key].get())
        self._request_render()

    def _crush_changed(self, value):
        s = self.project.to_source(self.t_shown)
        self.project.set_crush(s, float(value))

    def _toggle_play(self):
        self.playing = not self.playing
        self.btn_play.configure(text="暂停" if self.playing else "播放")

    # -- playback / rendering --------------------------------------------------
    def _request_render(self):
        self._render_want = self.project.to_source(self.t_shown)
        if self._render_busy:
            return
        self._render_busy = True
        threading.Thread(target=self._render_worker, daemon=True).start()

    def _render_worker(self):
        while self._render_want is not None:
            s = self._render_want
            self._render_want = None
            try:
                img = self.renderer.render(s, PREVIEW_W, PREVIEW_H)
            except Exception:
                img = None
            if img is not None:
                self._ui_queue.put(("frame", img, s))
        self._render_busy = False

    def _show(self, img: np.ndarray, source_t: float):
        pil = Image.fromarray(img)
        if self._current_img is not None and \
                self._current_img.size == pil.size:
            # two-step crossfade, the main window's motion language
            step1 = Image.blend(self._current_img, pil, 0.45)
            self._set_image(step1)
            self.after(60, lambda: self._set_image(pil))
        else:
            self._set_image(pil)
        self._current_img = pil
        mode = MODE_LABEL[self.project.attach_at(source_t)]
        self.caption.configure(
            text=f"{mode} · {source_t:.1f}s / {self.project.duration:.0f}s")

    def _set_image(self, pil: Image.Image):
        cimg = ctk.CTkImage(light_image=pil, dark_image=pil,
                            size=(PREVIEW_W, PREVIEW_H))
        self.img_label.configure(image=cimg)
        self.img_label.image = cimg

    def _tick(self):
        while True:
            try:
                item = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            if item[0] == "frame":
                self._show(item[1], item[2])
            elif item[0] == "status":
                self.status.configure(text=item[1])
            elif item[0] == "export_done":
                self._exporting = False
                self.btn_export.configure(state="normal")
        now = time.monotonic()
        dt = min(now - self._last_tick, 0.1)
        self._last_tick = now
        if self.playing:
            self.t_out += dt
            total = self.project.output_duration()
            if self.t_out >= total:
                self.t_out = total
                self.playing = False
                self.btn_play.configure(text="播放")
        # glide the shown position toward the playhead
        delta = self.t_out - self.t_shown
        if abs(delta) > 0.005:
            self.t_shown += delta * min(1.0, dt * 10)
            self._draw_timeline()
            self._request_render()
        self.after(50, self._tick)

    # -- export ------------------------------------------------------------
    def _export(self):
        if self._exporting:
            return
        self._exporting = True
        self.btn_export.configure(state="disabled")
        self.status.configure(text="正在导出…")
        out = self.job_dir / "studio_export.mp4"

        def work():
            try:
                export_mp4(self.job_dir, self.project, out,
                           progress=lambda i, n: self._ui_queue.put(
                               ("status", f"正在导出… {i}/{n}")))
                self._ui_queue.put(("status", f"已导出 {out.name}"))
            except Exception as exc:  # surface, never crash the app
                self._ui_queue.put(("status", f"导出失败：{exc}"))
            self._ui_queue.put(("export_done",))

        threading.Thread(target=work, daemon=True).start()
