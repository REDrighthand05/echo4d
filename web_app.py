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
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import app as backend  # the desktop shell's module-level backend
from echo_app import cli_events
from echo_app.settings import coerce_appearance, load_settings, save_settings
from echo_app.version import APP_VERSION

# 设置 → 显示： the page speaks camelCase, the settings file
# snake_case -- the same translation jobsRoot/updateProxy get,
# table-driven so the snapshot and the save path cannot drift.
_APPEARANCE_KEYS = [
    ("frostBlur", "frost_blur"), ("frostVeil", "frost_veil"),
    ("frostSaturate", "frost_saturate"), ("workHaze", "work_haze"),
    ("setBlur", "set_blur"), ("setVeil", "set_veil"),
    ("setSaturate", "set_saturate"), ("grafCount", "graf_count"),
    ("grafWidth", "graf_width"), ("grafOpacity", "graf_opacity"),
    ("grafSaturate", "graf_saturate"), ("mistDensity", "mist_density"),
    ("mistSize", "mist_size"), ("motionOn", "motion_on"),
    ("motionSpeed", "motion_speed"), ("motionAmp", "motion_amp"),
    ("sheenOn", "sheen_on"),
]


def _appearance_to_js(appearance):
    return {camel: getattr(appearance, snake)
            for camel, snake in _APPEARANCE_KEYS}


def _appearance_from_js(appearance, data):
    snake = {snake: data[camel] for camel, snake in _APPEARANCE_KEYS
             if isinstance(data, dict) and camel in data}
    return coerce_appearance(snake, base=appearance)

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
        self._inbox_sig = None
        self._cli_cursor = cli_events.load_cursor()
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

    # -- phone-capture inbox -----------------------------------------
    # The collector's captures sync themselves home (Syncthing)
    # into an "ECHO Captures" folder; the newest finished ones are
    # offered in the page, one click selects their video through
    # the same path as a drag-in. A capture counts as arrived only
    # when its meta says the phone finished it and no Syncthing
    # temp file remains -- a half-synced folder is never offered.
    @staticmethod
    def _inbox_roots():
        roots = []
        env = os.environ.get("ECHO_INBOX_DIR")
        if env:
            roots.append(Path(env))
        for letter in "CDEFG":
            roots.append(Path(f"{letter}:\\ECHO Captures"))
        roots.append(Path.home() / "ECHO Captures")
        return roots

    def _inbox(self):
        out = []
        for root in self._inbox_roots():
            try:
                if not root.is_dir():
                    continue
                for cap in root.glob("capture_*"):
                    if cap.is_dir():
                        entry = self._inbox_entry(cap)
                        if entry is not None:
                            out.append(entry)
            except OSError:
                continue
        out.sort(key=lambda e: e["mtime"], reverse=True)
        for entry in out:
            del entry["mtime"]
        return out

    @staticmethod
    def _inbox_entry(cap):
        video = cap / "video.mp4"
        if not video.exists():
            return None
        try:
            names = [p.name for p in cap.iterdir()]
        except OSError:
            return None
        if any(n.startswith(".syncthing") for n in names):
            return None
        meta = cap / "meta.json"
        if meta.exists():
            try:
                info = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            if not info.get("finalized", False):
                return None
        try:
            total = sum(p.stat().st_size
                        for p in cap.rglob("*") if p.is_file())
            mtime = video.stat().st_mtime
        except OSError:
            return None
        return {
            "name": cap.name,
            "dir": str(cap),
            "video": str(video),
            "when": time.strftime("%m-%d %H:%M",
                                  time.localtime(mtime)),
            "mb": round(total / 1e6),
            "mtime": mtime,
        }

    def select_inbox_capture(self, dir_path):
        if self.busy:
            return {"ok": False, "message": "正在忙，先等这一件事做完。"}
        cap = Path(str(dir_path))
        try:
            parent = cap.resolve().parent
        except OSError:
            parent = None
        roots = []
        for root in self._inbox_roots():
            try:
                roots.append(root.resolve())
            except OSError:
                pass
        if parent is None or parent not in roots:
            return {"ok": False, "message": "这段采集不在收件夹里。"}
        if self._inbox_entry(cap) is None:
            return {"ok": False, "message": "这段采集还没传完。"}
        return self.select_video_path(cap / "video.mp4")

    # -- CLI activity (silent toasts) ---------------------------------
    # The headless CLI is a separate process; it leaves one JSON
    # line per finished command in the settings folder (see
    # echo_app.cli_events), and the page shows each as a one-line
    # silent toast -- the user's eyeball check that the CLI is
    # alive. The cursor (highest seq already handed to the page)
    # persists in the same folder, so a restart never replays old
    # news: the snapshot catches up only unpresented events from
    # the last ten minutes, while poll drains new ones as they
    # land. Events skipped as stale still advance the cursor.
    def _drain_cli_events(self, recent_only=False):
        try:
            events, hi = cli_events.read_since(self._cli_cursor)
            if hi > self._cli_cursor:
                self._cli_cursor = hi
                cli_events.save_cursor(hi)
            if recent_only:
                cutoff = time.time() - 600
                events = [e for e in events
                          if float(e.get("ts", 0)) >= cutoff]
            return [{"cmd": e.get("cmd"), "ok": bool(e.get("ok")),
                     "summary": e.get("summary", "")} for e in events]
        except Exception:
            return []

    # -- snapshot + polling ------------------------------------------
    def get_snapshot(self):
        hist = self._history()
        self._hist_sig = json.dumps(hist, ensure_ascii=False)
        inbox = self._inbox()
        self._inbox_sig = json.dumps(inbox, ensure_ascii=False)
        return {
            "version": APP_VERSION,
            "jobsRoot": str(self.jobs_root),
            "settings": {
                "autostart": self.settings.autostart,
                "closeToTray": self.settings.close_to_tray,
                "updateProxy": self.settings.update_proxy,
                "appearance": _appearance_to_js(self.settings.appearance),
            },
            "history": hist,
            "inbox": inbox,
            "status": self.status,
            "frac": self.frac,
            "busy": self.busy,
            "log": list(self._all_log),
            "cliEvents": self._drain_cli_events(recent_only=True),
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
        cli_new = self._drain_cli_events()
        if cli_new:
            out["cliEvents"] = cli_new
        hist = self._history()
        sig = json.dumps(hist, ensure_ascii=False)
        if sig != self._hist_sig:
            self._hist_sig = sig
            out["history"] = hist
        inbox = self._inbox()
        isig = json.dumps(inbox, ensure_ascii=False)
        if isig != self._inbox_sig:
            self._inbox_sig = isig
            out["inbox"] = inbox
        if self._preview_data is not None:
            out["preview"] = self._preview_data
        if self.depth_out is not None:
            out["depthUrl"] = Path(self.depth_out).as_uri()
        if self.job is not None:
            # Real per-output availability, so the page lights only
            # the buttons whose file actually exists. (The page used
            # to light them all for any job, and clicks on a failed
            # job's buttons then silently did nothing.)
            job = Path(self.job)
            out["jobOutputs"] = {
                "any": job.exists(),
                "scene": backend.job_finished(job),
                "preview": (job / "preview.html").exists(),
                "replay": backend._replay_available(job),
            }
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
        self.frac = 0.0
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
        def _progress(stage, done, total):
            if total:
                # Extraction is quick; weight the bar to the depth
                # stage so it moves steadily through the long part.
                self.frac = (done / total) if stage != "frames" else 0.02
            if stage == "depth" and total:
                self.status = f"正在生成深度视频… {done}/{total}"
            elif stage == "encode":
                self.status = "正在合成深度视频…"

        try:
            self.depth_out = backend.run_depth_video(
                self.video, progress=_progress)
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

    def _open_target(self, target, what):
        """Open an output file, and NEVER fail silently: a missing
        file or a failed system open lands in the status line and
        the log, where the user can see what happened. (Silent
        no-ops here are what made the whole row of buttons look
        dead on 2026-11-02.)"""
        if target is not None and Path(target).exists():
            try:
                backend.play_file(Path(target))
                return {"ok": True}
            except Exception as exc:
                self.status = f"{what}打开失败，看一下日志。"
                self._log(f"{what}打开失败：{exc}")
                return {"ok": False}
        self.status = f"{what}还没有，暂时打不开。"
        self._log(f"{what}还没有，暂时打不开。")
        return {"ok": False}

    def open_scene(self, job_dir=None):
        job = self._job_for(job_dir)
        if job is None:
            return self._open_target(None, "场景")
        return self._open_target(backend.scene_file(job), "场景")

    def open_folder(self, job_dir=None):
        job = self._job_for(job_dir)
        if job is None or not Path(job).exists():
            return self._open_target(None, "任务文件夹")
        try:
            _open_folder(Path(job))
            return {"ok": True}
        except Exception as exc:
            self.status = "任务文件夹打开失败，看一下日志。"
            self._log(f"任务文件夹打开失败：{exc}")
            return {"ok": False}

    def open_preview_replay(self, job_dir=None):
        job = self._job_for(job_dir)
        if job is None:
            return self._open_target(None, "过程回放")
        return self._open_target(job / "preview.html", "过程回放")

    def open_route_replay(self, job_dir=None):
        job = self._job_for(job_dir)
        if job is None:
            return self._open_target(None, "路线回放")
        return self._open_target(backend.replay_file(job), "路线回放")

    def play_depth(self):
        if self.depth_out is None:
            return self._open_target(None, "深度视频")
        return self._open_target(Path(self.depth_out), "深度视频")

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
            if "appearance" in data:
                self.settings.appearance = _appearance_from_js(
                    self.settings.appearance, data["appearance"])
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
            from echo_app import updater
            from echo_app.updater import (check_for_update_status,
                                          fetch_releases, pick_hop)
            # The relay first: a static mirror with no API quota.
            # Only when it cannot answer do we spend GitHub API
            # calls (whose shared-IP rate limit kept locking the
            # user out with a misleading "cannot connect").
            status = info = hop_release = None
            try:
                plan = updater.relay_plan(
                    APP_VERSION, updater.fetch_relay_index())
            except Exception:
                plan = None
            if plan is not None:
                status, info, hop_release = plan
            else:
                status, info = check_for_update_status(APP_VERSION)
            if status != "error":
                # A probed local proxy answered: keep it, so no one
                # ever has to set anything for updates again.
                try:
                    found = updater.auto_proxy()
                    if found and not self.settings.update_proxy:
                        self.settings.update_proxy = found
                        save_settings(self.settings)
                        updater.set_proxy(found)
                        self._log("已自动找到能连上 GitHub 的本机代理"
                                  f"（{found}），以后的更新都走它，"
                                  "不用你管。")
                except Exception:
                    pass
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
                    hop = hop_release
                    if hop is None:
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
                # explains itself -- and by now every route was tried.
                self._log("更新检查没连上：直连和本机常见代理端口都"
                          "试过了，还是到不了 GitHub。可以到发布页"
                          "手动更新，或在设置里填你自己的代理地址。")
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

    # -- window controls (the window is frameless; the page carries
    # its own caption buttons in the same glass language) --------------
    def win_minimize(self):
        try:
            if self._window is not None:
                self._window.minimize()
        except Exception:
            pass
        return {"ok": True}

    def win_toggle_max(self):
        try:
            if self._window is not None:
                if getattr(self, "_maximized", False):
                    self._window.restore()
                    self._maximized = False
                else:
                    self._window.maximize()
                    self._maximized = True
        except Exception:
            pass
        return {"ok": True}

    def win_close(self):
        self._quit()
        return {"ok": True}


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
            try:  # the native handle may not exist yet at first call
                hwnd = ctypes.windll.user32.FindWindowW(
                    None, "ECHO 回声") or None
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
    try:  # say once, in the log, where depth compute will run
        import onnxruntime as _ort

        _provs = _ort.get_available_providers()
        if ("DmlExecutionProvider" in _provs
                or "CUDAExecutionProvider" in _provs):
            bridge._log("显卡加速可用，深度计算会走显卡。")
        else:
            bridge._log("深度计算使用 CPU（未检测到显卡运行时）。")
    except Exception:
        pass
    # The page ships in the loose payload: resolve it next to this
    # file first (that is the copy hot updates replace), and only
    # fall back to the frozen resource path in a source tree.
    _here = Path(__file__).resolve().parent / "web_ui" / "index.html"
    index = _here if _here.exists() else backend._resource_path(
        "web_ui/index.html")
    window = webview.create_window(
        "ECHO 回声", url=str(index),
        js_api=bridge, width=1040, height=836,
        min_size=(880, 620), frameless=True,
        background_color="#dbe2eb")
    bridge._window = window
    _polish_window(window)
    try:
        window.events.shown += lambda: _polish_window(window)
    except Exception:
        pass
    # The native handle often only exists a moment after start, so
    # polish again on a delay -- a single early attempt that found
    # no handle used to leave the frame unmodernised for the whole
    # session with no sign of why.
    for delay in (1.5, 4.0):
        timer = threading.Timer(delay, _polish_window, args=(window,))
        timer.daemon = True
        timer.start()
    webview.start()


if __name__ == "__main__":
    main()
