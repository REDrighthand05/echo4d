"""ECHO web shell -- the Meta-style glass frontend (web_ui/) in a
pywebview window, bridged to the SAME backend the desktop shell
uses: app.py's module-level run_pipeline / run_depth_video /
list_previous_jobs / scene_file / replay_file, echo_app.settings
for persistence, echo_app.updater for updates, and the
echo_app.live_preview sequencer for the growing-scene preview.

Nothing here re-implements the pipeline. The bridge only moves
state between the page and those functions, with the same guards
as the desktop shell (one job at a time, validated video, outputs
only for jobs that have them).

Run:  python web_app.py           (needs pywebview; on Windows it
                                   rides the system WebView2)
"""
import base64
import io
import json
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import app as backend  # the desktop shell's module-level backend
from echo_app.settings import load_settings, save_settings
from echo_app.version import APP_VERSION

PLAIN_STATUS = backend.PLAIN_STATUS
STAGE_COUNT = backend.STAGE_COUNT
STAGE_ORDER = list(PLAIN_STATUS.keys())


def _open_folder(path: Path):
    if sys.platform.startswith("win"):
        subprocess.Popen(["explorer", str(path)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class Bridge:
    """State + actions exposed to the page as window.pywebview.api."""

    def __init__(self):
        self.settings = load_settings()
        try:
            from echo_app.updater import set_proxy
            set_proxy(self.settings.update_proxy)
        except Exception:
            pass
        self.jobs_root = self.settings.jobs_root_path()
        self.video = None            # Path of the selected video
        self.job = None              # Path of the current job
        self.busy = False
        self.busy_kind = None        # "scene" | "depth" | None
        self.status = "还没开始。"
        self.frac = 0.0
        self._stage_idx = 0
        self.depth_out = None        # Path of the last depth video
        self._pending_log = []
        self._all_log = []
        self._hist_sig = None
        self._preview_seq = 0
        self._preview_data = None
        self._preview_stop = threading.Event()
        self._depth_stop = threading.Event()
        self._depth_stop.set()
        self._view = None
        self.update = None           # {"version": str} when available
        self._window = None
        self._log("ECHO 已就绪。")
        # Hot-update housekeeping: report the last apply's outcome
        # once, and repair an interrupted one if it is still pending.
        try:
            from echo_app import hotupdate
            _root = hotupdate.install_root()
            hotupdate.recover(_root)
            _result = hotupdate.last_result(_root)
            if _result is not None:
                if _result.get("ok"):
                    self._log(f"热更新完成：已升级到 v{_result.get('version')}。")
                else:
                    self._log("上次热更新没成功，已经自动退回原来的版本。")
                (_root / ".echo" / "result.json").unlink(missing_ok=True)
        except Exception:
            pass

    # -- internal helpers -------------------------------------------
    def _log(self, line):
        self._all_log.append(line)
        del self._all_log[:-400]
        self._pending_log.append(line)

    def _pipeline_log(self, line):
        """run_pipeline's log callback: stage lines become the
        plain-language status + progress; everything is logged."""
        text = str(line)
        if text in PLAIN_STATUS:
            if text in STAGE_ORDER:
                self._stage_idx = max(self._stage_idx,
                                      STAGE_ORDER.index(text) + 1)
            self.status = PLAIN_STATUS[text] + "…"
            self.frac = self._stage_idx / STAGE_COUNT
        self._log(text)

    def _check_video(self, path: Path):
        if path.suffix.lower() not in backend.VIDEO_SUFFIXES:
            return "这个文件不是视频，换个视频文件试试。"
        if not path.exists():
            return "没找到这个视频文件。"
        return None

    def _history(self):
        out = []
        for job in backend.list_previous_jobs(self.jobs_root):
            try:
                when = time.strftime(
                    "%m-%d %H:%M", time.localtime(job.stat().st_mtime))
            except OSError:
                when = ""
            out.append({
                "name": backend._job_display_name(job),
                "dir": str(job),
                "finished": backend.job_finished(job),
                "when": when,
                "scale": backend.scale_status_line(job),
                "preview": (job / "preview.html").exists(),
                "replay": backend._replay_available(job),
            })
        return out

    # -- snapshot + polling ------------------------------------------
    def get_snapshot(self):
        hist = self._history()
        self._hist_sig = json.dumps(hist, ensure_ascii=False)
        return {
            "version": APP_VERSION,
            "jobsRoot": str(self.jobs_root),
            "settings": {
                "autostart": self.settings.autostart,
                "closeToTray": self.settings.close_to_tray,
                "updateProxy": self.settings.update_proxy,
            },
            "history": hist,
            "status": self.status,
            "frac": self.frac,
            "busy": self.busy,
            "log": list(self._all_log),
        }

    def poll(self):
        out = {
            "status": self.status,
            "frac": self.frac,
            "busy": self.busy,
            "busyKind": self.busy_kind,
            "hasJob": self.job is not None,
            "depthReady": self.depth_out is not None,
            "logs": self._pending_log[:],
            "previewSeq": self._preview_seq,
            "update": self.update,
        }
        self._pending_log = []
        hist = self._history()
        sig = json.dumps(hist, ensure_ascii=False)
        if sig != self._hist_sig:
            self._hist_sig = sig
            out["history"] = hist
        if self._preview_data is not None:
            out["preview"] = self._preview_data
        if self.depth_out is not None:
            out["depthUrl"] = Path(self.depth_out).as_uri()
        return out

    def get_preview_frame(self):
        """The latest preview frame (data URL) + its sequence."""
        return {"seq": self._preview_seq, "data": self._preview_data}

    # -- selection ----------------------------------------------------
    def pick_video(self):
        if self.busy or self._window is None:
            return None
        import webview

        paths = self._window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=False,
            file_types=("视频 (*.mp4;*.mov;*.mkv;*.avi)",
                        "所有文件 (*.*)"))
        if paths:
            return self.select_video_path(paths[0])
        return None

    def select_video_path(self, path):
        if self.busy:
            return {"ok": False, "message": "正在忙，先等这一件事做完。"}
        p = Path(str(path))
        err = self._check_video(p)
        if err:
            self.status = err
            return {"ok": False, "message": err}
        self.video = p
        self.status = "视频选好了。点「开始生成」，或者做深度视频。"
        self._log(f"已选视频：{p.name}")
        return {"ok": True, "name": p.name}

    # -- jobs -----------------------------------------------------------
    def start_scene(self):
        if self.busy or self.video is None:
            return {"ok": False}
        err = self._check_video(self.video)
        if err:
            self.status = err
            return {"ok": False, "message": err}
        self.busy, self.busy_kind = True, "scene"
        self._stage_idx, self.frac = 0, 0.0
        self.status = "开始了。"
        threading.Thread(target=self._scene_worker, daemon=True).start()
        return {"ok": True}

    def _scene_worker(self):
        try:
            job = backend.run_pipeline(
                self.video, self.jobs_root, self._pipeline_log,
                job_sink=self._on_job_started)
            self.job = job
            self.status = "场景做好了。点「打开场景」就能看。"
            self.frac = 1.0
            scale = backend.scale_status_line(job)
            if scale:
                self._log(scale)
            self._log("场景生成完成。")
        except Exception:
            self.status = "生成失败，看一下日志。"
            self._log(traceback.format_exc(limit=2))
        finally:
            self.busy, self.busy_kind = False, None

    def start_depth(self):
        if self.busy or self.video is None:
            return {"ok": False}
        err = self._check_video(self.video)
        if err:
            self.status = err
            return {"ok": False, "message": err}
        self.busy, self.busy_kind = True, "depth"
        self.depth_out = None
        self.status = "正在生成深度视频…"
        # The depth module lands its stills progressively in
        # <stem>_depth/ next to the output; watch them so the
        # preview screen (and the echo behind the glass) can show
        # the depth picture while it is still being made. The
        # output name mirrors run_depth_video's own choice -- one
        # job at a time, so the name cannot race.
        out = self.video.with_name(f"{self.video.stem}_depth.mp4")
        base, n = out, 1
        while out.exists():
            n += 1
            out = base.with_name(f"{base.stem}_{n}.mp4")
        self._depth_stop = threading.Event()
        threading.Thread(target=self._depth_stills_loop,
                         args=(out.parent / (out.stem + "_depth"),
                               self._depth_stop), daemon=True).start()
        threading.Thread(target=self._depth_worker, daemon=True).start()
        return {"ok": True}

    def _depth_stills_loop(self, stills_dir, stop):
        seen = None
        while not stop.is_set():
            try:
                stills = sorted(stills_dir.glob("d_*.png"))
                if stills:
                    newest = stills[-1]
                    size = newest.stat().st_size
                    if size and (newest.name, size) != seen:
                        seen = (newest.name, size)
                        data = newest.read_bytes()
                        self._preview_data = (
                            "data:image/png;base64,"
                            + base64.b64encode(data).decode())
                        self._preview_seq += 1
            except Exception:
                pass
            stop.wait(0.8)

    def _depth_worker(self):
        try:
            self.depth_out = backend.run_depth_video(self.video)
            self.status = "深度视频做好了。"
            self._log("深度视频完成。")
        except Exception:
            self.status = "深度视频生成失败，看一下日志。"
            self._log(traceback.format_exc(limit=2))
        finally:
            self._depth_stop.set()
            self.busy, self.busy_kind = False, None

    # -- outputs --------------------------------------------------------
    def _job_for(self, job_dir):
        if job_dir:
            p = Path(str(job_dir))
            if p.exists():
                return p
        return self.job

    def select_job(self, job_dir):
        job = self._job_for(job_dir)
        if job is not None:
            self.job = job
            self._start_preview(job)
        return {"ok": job is not None}

    def open_scene(self, job_dir=None):
        job = self._job_for(job_dir)
        if job is None:
            return {"ok": False}
        target = backend.scene_file(job)
        if target.exists():
            backend.play_file(target)
            return {"ok": True}
        return {"ok": False}

    def open_folder(self, job_dir=None):
        job = self._job_for(job_dir)
        if job is None:
            return {"ok": False}
        _open_folder(job)
        return {"ok": True}

    def open_preview_replay(self, job_dir=None):
        job = self._job_for(job_dir)
        if job is None:
            return {"ok": False}
        target = job / "preview.html"
        if target.exists():
            backend.play_file(target)
            return {"ok": True}
        return {"ok": False}

    def open_route_replay(self, job_dir=None):
        job = self._job_for(job_dir)
        if job is None:
            return {"ok": False}
        target = backend.replay_file(job)
        if target.exists():
            backend.play_file(target)
            return {"ok": True}
        return {"ok": False}

    def play_depth(self):
        if self.depth_out is not None and Path(self.depth_out).exists():
            backend.play_file(Path(self.depth_out))
            return {"ok": True}
        return {"ok": False}

    def print_markers(self):
        try:
            backend.open_marker_sheet()
            return {"ok": True}
        except Exception as exc:
            self._log(f"标记纸打开失败：{exc}")
            return {"ok": False}

    # -- live preview -----------------------------------------------------
    def _on_job_started(self, job):
        self.job = job
        self._start_preview(job)

    def _start_preview(self, job):
        self._preview_stop.set()
        self._preview_stop = threading.Event()
        stop = self._preview_stop
        threading.Thread(target=self._preview_loop, args=(job, stop),
                         daemon=True).start()

    def _preview_loop(self, job, stop):
        """Same watcher/sequencer loop as the desktop shell; frames
        leave as PNG data URLs for the page's preview canvas.
        Optional subsystem: it can never take a job down."""
        try:
            from PIL import Image

            from echo_app.live_preview import (JobWatcher, LiveSequencer,
                                               ViewState, render_snapshot)
            if self._view is None:
                self._view = ViewState()
            watcher = JobWatcher(job, stable_polls=2)
            seq = LiveSequencer(job, view=self._view)
            last_signature, last_render = None, 0.0
            while not stop.is_set():
                events = watcher.poll()
                if events:
                    seq.consume(events)
                view = self._view
                signature = (seq.state.phase, seq.state.frames_seen,
                             len(seq.state.reliefs),
                             seq.state.settled is not None,
                             view.version if view is not None else 0)
                now = time.monotonic()
                if signature != last_signature or now - last_render > 3.0:
                    img, _dt = render_snapshot(seq, width=880, height=520)
                    frame = Image.fromarray(img)
                    buf = io.BytesIO()
                    frame.save(buf, format="PNG")
                    self._preview_data = (
                        "data:image/png;base64,"
                        + base64.b64encode(buf.getvalue()).decode())
                    self._preview_seq += 1
                    last_render, last_signature = now, signature
                stop.wait(0.7)
        except Exception:
            pass

    def preview_drag(self, dx, dy, pan=False):
        if self._view is not None:
            if pan:
                self._view.pan_drag(float(dx), float(dy))
            else:
                self._view.drag(float(dx), float(dy))
        return {"ok": True}

    def preview_zoom(self, steps):
        if self._view is not None:
            self._view.zoom(float(steps))
        return {"ok": True}

    def preview_reset(self):
        if self._view is not None:
            self._view.reset()
        return {"ok": True}

    # -- settings + updates ----------------------------------------------
    def save_settings(self, data):
        try:
            if "jobsRoot" in data and data["jobsRoot"]:
                self.settings.jobs_root = str(data["jobsRoot"])
                self.jobs_root = self.settings.jobs_root_path()
                self._hist_sig = None
            if "autostart" in data:
                self.settings.autostart = bool(data["autostart"])
                try:
                    from echo_app.autostart import set_autostart
                    set_autostart(self.settings.autostart)
                except Exception:
                    pass
            if "closeToTray" in data:
                self.settings.close_to_tray = bool(data["closeToTray"])
            if "updateProxy" in data:
                self.settings.update_proxy = str(data["updateProxy"]).strip()
                try:
                    from echo_app.updater import set_proxy
                    set_proxy(self.settings.update_proxy)
                except Exception:
                    pass
            save_settings(self.settings)
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "message": str(exc)}

    def pick_jobs_root(self):
        if self._window is None:
            return None
        import webview

        paths = self._window.create_file_dialog(webview.FOLDER_DIALOG)
        return paths[0] if paths else None

    def check_update(self):
        threading.Thread(target=lambda: self._run_check(False),
                         daemon=True).start()
        return {"ok": True}

    def manual_check_update(self):
        """The settings-page check: the user pressed the button, so
        they get an honest, specific answer back -- update / latest
        / error -- instead of the startup check's polite silence."""
        return self._run_check(True)

    def _run_check(self, manual):
        try:
            from echo_app.updater import (check_for_update_status,
                                          fetch_releases, pick_hop)
            status, info = check_for_update_status(APP_VERSION)
            if status == "update" and info is not None:
                self.update = {"version": info.version}
                self._update_info = info
                self._log(f"有新版本：{info.version}")
                # Prefer the hot path, megabytes instead of the full
                # installer. Packs are chained, so aim at the NEXT
                # release above this one, not blindly at the latest:
                # a client several versions behind climbs one hop,
                # relaunches, and the next check climbs the next.
                try:
                    from echo_app import hotupdate
                    hop = pick_hop(fetch_releases(), APP_VERSION)
                    if hop is not None:
                        hot = hotupdate.check_hot_update(
                            hop, hotupdate.install_root())
                        if hot is not None:
                            self._hot_info = hot
                            self.update = {
                                "version": hot.version, "hot": True,
                                "mb": round(hot.total_bytes / 1e6, 1)}
                            self._log("这个版本可以热更新，"
                                      f"只需下载约 {self.update['mb']} MB。")
                except Exception:
                    pass
                return {"status": "update", "version": info.version}
            if status == "error":
                # Never silent again: the log is where a failed check
                # explains itself (network / proxy trouble).
                self._log("更新检查没连上：多半是网络或代理的问题。"
                          "可以到设置里填代理地址，或到发布页手动更新。")
                if not manual and not getattr(self, "_check_retried",
                                              False):
                    self._check_retried = True
                    self._retry_timer = threading.Timer(
                        90.0, lambda: self._run_check(False))
                    self._retry_timer.daemon = True
                    self._retry_timer.start()
                return {"status": "error"}
            return {"status": "latest"}
        except Exception:
            return {"status": "error"}

    def download_update(self):
        hot = getattr(self, "_hot_info", None)
        if hot is not None:
            def _hot():
                try:
                    import subprocess

                    from echo_app import hotupdate
                    root = hotupdate.install_root()
                    hotupdate.download_and_stage(root, hot)
                    cmd = hotupdate.request_apply(root, hot)
                    self._log("更新已备好，正在重启完成更新…")
                    subprocess.Popen(cmd, close_fds=True)
                    threading.Timer(0.4, self._quit).start()
                except Exception:
                    self._hot_info = None
                    self._log("热更新没成，改走完整安装包。")
                    self.download_update()
            threading.Thread(target=_hot, daemon=True).start()
            return {"ok": True}
        info = getattr(self, "_update_info", None)
        if info is None:
            return {"ok": False}

        def _download():
            try:
                from echo_app.updater import download_installer
                dest = download_installer(info, self.jobs_root)
                self._log("新版本安装包已下载，正在打开。")
                backend.play_file(Path(dest))
            except Exception:
                self._log("更新下载失败。")
        threading.Thread(target=_download, daemon=True).start()
        return {"ok": True}

    def _quit(self):
        try:
            if self._window is not None:
                self._window.destroy()
        except Exception:
            pass
        import os
        os._exit(0)


def _polish_window(window):
    """Win11 window treatments, best effort: rounded corners and a
    caption bar wearing the app's own pale tint, so the frame stops
    looking like a stranger around the page. Windows-only and fully
    guarded -- a failure here must never cost a launch."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        native = getattr(window, "native", None)
        hwnd = None
        if native is not None:
            try:
                hwnd = int(native.Handle)
            except Exception:
                hwnd = None
        if not hwnd:
            return
        dwm = ctypes.windll.dwmapi

        def _set(attr, value):
            v = ctypes.c_int(value)
            dwm.DwmSetWindowAttribute(
                hwnd, attr, ctypes.byref(v), ctypes.sizeof(v))

        _set(20, 0)         # immersive dark caption: off (light glass)
        _set(33, 2)         # corner preference: round (the Win11 look)
        _set(35, 0xF7F2EE)  # caption colour: our pale glass #eef2f7 (BGR)
        _set(36, 0x211E1C)  # caption text: our ink #1c1e21 (BGR)
    except Exception:
        pass


def main():
    import webview

    if sys.platform == "win32":
        try:  # per-monitor DPI awareness: crisp text on scaled displays
            import ctypes

            ctypes.windll.user32.SetProcessDpiAwarenessContext(
                ctypes.c_void_p(-4))
        except Exception:
            pass

    bridge = Bridge()
    # The page ships in the loose payload: resolve it next to this
    # file first (that is the copy hot updates replace), and only
    # fall back to the frozen resource path in a source tree.
    _here = Path(__file__).resolve().parent / "web_ui" / "index.html"
    index = _here if _here.exists() else backend._resource_path(
        "web_ui/index.html")
    window = webview.create_window(
        "ECHO 回声", url=str(index),
        js_api=bridge, width=1040, height=836,
        min_size=(880, 620),
        background_color="#dbe2eb")
    bridge._window = window
    _polish_window(window)
    try:
        window.events.shown += lambda: _polish_window(window)
    except Exception:
        pass
    webview.start()


if __name__ == "__main__":
    main()
