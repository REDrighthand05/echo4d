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
import re
import subprocess
import sys
import tempfile
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




def _reveal_file(path: Path):
    """Show a file in the system file manager, selected if possible.

    Never raises: a failed reveal falls back to opening the folder,
    and a failed folder open is swallowed -- the caller reports the
    outcome through the status line and the log, not an exception.
    """
    path = Path(path)
    try:
        if sys.platform.startswith("win"):
            subprocess.Popen(["explorer", "/select," + str(path)])
            return
    except Exception:
        pass
    try:
        _open_folder(path.parent)
    except Exception:
        pass


# ================= PERSISTENT APP LOG (v0.6.30) =================
# The app used to keep its log only in memory, so an unexpected
# exit could never be diagnosed afterwards: was it a crash, a
# restart, or did the user just close it? Now every _log line
# also lands in a rotating file under the per-user app-data
# folder (%APPDATA%\ECHO\app.log on Windows -- the same folder
# settings.py already uses, never the install dir, so a hot
# update can never wipe it). After any exit the last lines show
# what happened: a clean quit writes "已退出。", an update
# restart writes "正在重启…", and an exit with no shutdown line
# (plus an unhandled-exception traceback) is a crash.
# The file stays small (1 MB, two rotated backups) and never
# carries private paths or job content -- paths are redacted
# before they are written. Logging never raises: a logging
# failure must never take the app down.
# ------------------------------------------------------------------
_APPLOG_MAX_BYTES = 1 << 20  # 1 MB per file
_APPLOG_KEEP = 2             # app.log + app.log.1 + app.log.2
_applog_lock = threading.Lock()
_applog_path_override = None  # tests point it at a temp file


def _applog_path():
    if _applog_path_override is not None:
        return Path(_applog_path_override)
    try:
        from echo_app.settings import settings_dir
        return settings_dir() / "app.log"
    except Exception:
        return None


def _applog_redact(line):
    """Strip private paths from a log line before it reaches the
    file. The in-memory log the page shows is untouched -- only
    the file copy is redacted."""
    try:
        home = str(Path.home())
        if home:
            line = line.replace(home, "~")
    except Exception:
        pass
    # any remaining absolute Windows path -> <path>
    return re.sub(r"[A-Za-z]:\\[^\s\"']*", "<path>", line)


def _applog_write(line):
    """Append one timestamped line to app.log, rotating at ~1 MB.
    Never raises."""
    try:
        path = _applog_path()
        if path is None:
            return
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        text = f"{stamp} {line}\n"
        with _applog_lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                return
            try:
                if (path.exists()
                        and path.stat().st_size >= _APPLOG_MAX_BYTES):
                    for i in range(_APPLOG_KEEP, 0, -1):
                        src = (path if i == 1 else path.with_name(
                            f"{path.name}.{i - 1}"))
                        dst = path.with_name(f"{path.name}.{i}")
                        try:
                            if src.exists():
                                if dst.exists():
                                    dst.unlink()
                                src.rename(dst)
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                with open(path, "a", encoding="utf-8",
                          errors="replace") as f:
                    f.write(text)
            except Exception:
                pass
    except Exception:
        pass


_hooks_installed = False


def _install_excepthooks():
    """Log unhandled exceptions (any thread, plus the main one)
    with tracebacks to the persistent log, then keep the default
    behaviour (print to stderr). Installed once, at Bridge start.
    Without this, a worker thread dying on an exception leaves
    nothing behind at all."""
    global _hooks_installed
    if _hooks_installed:
        return
    _hooks_installed = True

    def _thread_hook(args):
        try:
            text = "".join(traceback.format_exception(
                args.exc_type, args.exc_value, args.exc_traceback))
            _applog_write("未处理的异常（线程 "
                          + (args.thread.name if getattr(
                              args, "thread", None) else "?")
                          + ")：\n" + text.rstrip())
        except Exception:
            pass
        try:
            traceback.print_exception(
                args.exc_type, args.exc_value, args.exc_traceback)
        except Exception:
            pass

    def _sys_hook(exc_type, exc_value, exc_tb):
        try:
            text = "".join(traceback.format_exception(
                exc_type, exc_value, exc_tb))
            _applog_write("未处理的异常（主线程）：\n" + text.rstrip())
        except Exception:
            pass
        try:
            sys.__excepthook__(exc_type, exc_value, exc_tb)
        except Exception:
            pass

    try:
        threading.excepthook = _thread_hook
    except Exception:
        pass
    try:
        sys.excepthook = _sys_hook
    except Exception:
        pass
# ================= END PERSISTENT APP LOG (v0.6.30) =================


# Silent auto-check cadence (v0.6.29): ten minutes between checks.
AUTO_CHECK_SECONDS = 600.0

# v0.6.38 (U-2): a PNG is complete only when it carries its
# IEND trailer (the last 12 bytes: zero length + "IEND" + CRC).
# The depth module writes its stills progressively, so a poll
# can catch one half-written; the signature + trailer check is
# the cheap way to tell a finished still from a torn one.
_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_PNG_IEND = b"\x00\x00\x00\x00IEND\xaeB`\x82"


def _png_complete(data: bytes) -> bool:
    return (len(data) >= len(_PNG_SIG) + len(_PNG_IEND)
            and data[:8] == _PNG_SIG
            and data[-12:] == _PNG_IEND)


class Bridge:
    """State + actions exposed to the page as window.pywebview.api."""

    def __init__(self):
        _install_excepthooks()
        global _drop_bridge
        _drop_bridge = self
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
        self._maximized = False
        self._pre_max_rect = None  # window rect before a work-area
        # maximise, so restore can put it back where it was
        self._inbox_sig = None
        self._cli_cursor = cli_events.load_cursor()
        self._preview_seq = 0
        self._preview_data = None
        self._preview_stop = threading.Event()
        self._depth_stop = threading.Event()
        self._depth_stop.set()
        self._view = None
        self._toolbox = None         # cached toolbox payload (lazy)
        self.update = None           # {"version": str} when available
        # v0.6.29: the silent 10-minute auto-check + the async
        # manual check + the download progress the page shows live.
        self._check_lock = threading.Lock()
        self._auto_check_started = False
        self._auto_check_stop = threading.Event()
        self._manual_checking = False
        self._manual_check_seq = 0
        self._manual_check_result = None  # {"seq", "status", "version"}
        self._update_phase = None  # {"phase", "frac", "text"} or None
        # v0.6.30: the auto-update gate + the persistent log.
        self._export_busy = 0      # exports running (they do not
        # set self.busy -- they run on worker threads)
        # v0.6.38 (U-10): the export counter is touched from
        # worker threads; the lock makes its increment /
        # decrement airtight -- a lost update could leave the
        # auto-update gate believing an export is (or isn't)
        # running.
        self._export_lock = threading.Lock()
        # v0.6.38 (U-1): guards the busy check-and-set in
        # start_scene/start_depth (and the release in the
        # workers' finally).
        self._busy_lock = threading.Lock()
        self._update_running = False  # an update download/apply
        # is in progress (manual or automatic)
        self._window = None
        # v0.6.30: the first line of the persistent log -- version,
        # process and time, so any later exit can be placed.
        self._log(f"启动：ECHO v{APP_VERSION}，进程 {os.getpid()}。")
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
        # v0.6.30: the same line goes to the persistent log (paths
        # redacted there); the page still shows the original.
        _applog_write(_applog_redact(str(line)))

    def _pipeline_log(self, line):
        """run_pipeline's log callback: stage lines become the
        plain-language status + progress; everything is logged."""
        text = str(line)
        # The pipeline logs each stage with a trailing ellipsis
        # ("正在抽帧…") while the PLAIN_STATUS keys carry none, so
        # strip it here -- exactly as the Tk fallback's handle_log
        # does -- or the stage is never recognised and the bar
        # never moves. The original text still goes to the log.
        key = text.rstrip("…").strip()
        if key in PLAIN_STATUS:
            if key in STAGE_ORDER:
                self._stage_idx = max(self._stage_idx,
                                      STAGE_ORDER.index(key) + 1)
            self.status = PLAIN_STATUS[key] + "…"
            self.frac = self._stage_idx / STAGE_COUNT
        self._log(text)

    def _check_video(self, path: Path):
        if path.suffix.lower() not in backend.VIDEO_SUFFIXES:
            return "这个文件不是视频，换个视频文件试试。"
        if not path.exists():
            return "没找到这个视频文件。"
        # v0.6.37: an empty file is not a video -- a 0-byte mp4
        # (interrupted copy, half-synced placeholder) used to
        # pass this gate, start a job, and fail only when the
        # extraction stage ran, leaving a dead job behind.
        try:
            if path.stat().st_size == 0:
                return "这个视频是空的（0 字节），换一个视频试试。"
        except OSError:
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
    def _history_safe(self):
        """History for the poll, never raising: one unexpected
        backend exception used to kill the whole twice-a-second
        poll response (U-7). None means "keep what the page
        already has"; the failure is logged, not swallowed
        silently."""
        try:
            return self._history()
        except Exception as exc:
            self._log(f"历史刷新失败：{exc}")
            return None

    def _inbox_safe(self):
        """Same guard for the capture inbox (U-7)."""
        try:
            return self._inbox()
        except Exception as exc:
            self._log(f"收件夹刷新失败：{exc}")
            return None

    def get_snapshot(self):
        hist = self._history_safe()
        if hist is None:
            hist = []
        self._hist_sig = json.dumps(hist, ensure_ascii=False)
        inbox = self._inbox_safe()
        if inbox is None:
            inbox = []
        self._inbox_sig = json.dumps(inbox, ensure_ascii=False)
        return {
            "version": APP_VERSION,
            "jobsRoot": str(self.jobs_root),
            "settings": {
                "autostart": self.settings.autostart,
                        "autoUpdate": self.settings.auto_update,
                "updateProxy": self.settings.update_proxy,
                "appearance": _appearance_to_js(self.settings.appearance),
                "windowMode": self.settings.window_mode,
        "textContrast": self.settings.text_contrast,
            },
            "history": hist,
            "inbox": inbox,
            "status": self.status,
            "frac": self.frac,
            "busy": self.busy,
            "resizeHook": self._resize_hook_status(),
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
        # v0.6.38 (U-7): a failed refresh must not kill the
        # poll -- keep the page's last history/inbox instead.
        hist = self._history_safe()
        if hist is not None:
            sig = json.dumps(hist, ensure_ascii=False)
            if sig != self._hist_sig:
                self._hist_sig = sig
                out["history"] = hist
        inbox = self._inbox_safe()
        if inbox is not None:
            isig = json.dumps(inbox, ensure_ascii=False)
            if isig != self._inbox_sig:
                self._inbox_sig = isig
                out["inbox"] = inbox
        # v0.6.29: the preview frame used to ride every poll as a
        # full base64 PNG, twice a second for the whole depth job --
        # while the page never reads it (it watches previewSeq and
        # fetches the frame through get_preview_frame). That is the
        # main jank during a depth job; nothing is sent here any
        # more (get_preview_frame still serves the latest frame).
        if self._manual_check_result is not None:
            out["manualCheck"] = dict(self._manual_check_result)
        if self._update_phase is not None:
            out["updateProgress"] = dict(self._update_phase)
        # v0.6.30: live resize-hook state, so the next "drag does
        # nothing" can be answered from the bridge state itself.
        out["resizeHook"] = self._resize_hook_status()
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

    def get_toolbox(self):
        """The toolbox page's payload: every registered echo_super
        kernel, built live from the registry + VERSIONS.md by
        echo_app.toolbox (never a hand-kept list, so it cannot
        drift). Built once and cached -- assembling the registry
        imports the kernel packages."""
        if self._toolbox is None:
            try:
                from echo_app.toolbox import build_toolbox
                self._toolbox = build_toolbox()
            except Exception as exc:  # the page says so, shell lives
                self._log(f"工具箱清单读取失败：{exc}")
                return {"count": 0, "deployedCount": 0,
                        "groups": [], "error": str(exc)}
        return self._toolbox

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
        # v0.6.39 (N-1): the busy check and the video set are one
        # atomic step under the claim lock -- a pick landing
        # between start_scene's claim and the worker's read of
        # self.video used to make the claimed job run on the newly
        # picked video instead of the one the claim validated.
        # (Same class as U-1; pywebview serializes bridge calls
        # in practice, so theoretical.) The two callers that
        # delegate here (pick_video, select_inbox_capture) are
        # covered too: their own busy checks are early exits,
        # the authoritative check-and-set is this one.
        with self._busy_lock:
            if self.busy:
                return {"ok": False, "message": "正在忙，先等这一件事做完。"}
            p = Path(str(path))
            err = self._check_video(p)
            if err:
                self.status = err
                return {"ok": False, "message": err}
            self.video = p
        self.frac = 0.0  # v0.6.29: a finished depth job leaves the bar
        # full; a newly picked video starts from an empty bar.
        self.status = "视频选好了。点「开始生成」，或者做深度视频。"
        self._log(f"已选视频：{p.name}")
        return {"ok": True, "name": p.name}

    # -- jobs -----------------------------------------------------------
    def start_scene(self):
        # v0.6.38 (U-1): the busy check-and-set is guarded by a
        # lock -- two bridge calls landing together used to both
        # pass the check and start two jobs. The video check runs
        # inside the lock too, so the whole claim is one atomic
        # step; the worker thread starts after the lock is out.
        with self._busy_lock:
            if self.busy or self.video is None:
                return {"ok": False}
            if self._update_in_flight():
                self.status = "正在更新，等更新完再开始。"
                return {"ok": False,
                        "message": "正在更新，等更新完再开始。"}
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
            # v0.6.38 (U-1): the release goes through the same
            # lock as the claim, so no check can slip between.
            with self._busy_lock:
                self.busy, self.busy_kind = False, None

    def start_depth(self):
        # v0.6.38 (U-1): same locked claim as start_scene --
        # the busy check-and-set is one atomic step.
        with self._busy_lock:
            if self.busy or self.video is None:
                return {"ok": False}
            if self._update_in_flight():
                self.status = "正在更新，等更新完再开始。"
                return {"ok": False,
                        "message": "正在更新，等更新完再开始。"}
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
                        data = newest.read_bytes()
                        # v0.6.38 (U-2): the stills land
                        # progressively -- a half-written PNG used
                        # to be published as the preview as-is.
                        # Only a complete PNG (signature + IEND
                        # trailer) is published; a torn one is
                        # skipped and retried on the next poll
                        # (seen is only updated on publish).
                        if _png_complete(data):
                            seen = (newest.name, size)
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
            # v0.6.38 (U-1): release through the claim lock.
            with self._busy_lock:
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

    # -- export ---------------------------------------------------------
    # One button, "导出通用文件": a finished job's coloured
    # solidified volume (voxel_color.glb or voxel_color.ply) is the
    # universal file -- Blender opens it natively, Three.js loads
    # it with GLTFLoader, Unreal imports it. The user picks the
    # format (GLB / PLY) next to the button; only the chosen file
    # is copied out. The "在 Blender 里打开" and "导出带贴图网格"
    # buttons are gone (v0.6.54): the universal file replaces
    # both, and the textured mesh was only ever a handoff format.
    # The coloured export is built on demand from points.bin (the
    # same echo_scene.voxel_export the headless CLI's `export`
    # uses), so a job with no GLB yet is not a dead button -- it
    # builds first. Nothing here ever fails silently: a missing
    # job, a missing point cloud, a failed build and a cancelled
    # folder pick all land in the status line and the log.
    def export_scene(self, job_dir=None, fmt="glb"):
        """Export the finished job's coloured solidified volume.

        fmt: "glb" (default) or "ply" -- only the chosen file is
             copied out, plus the info file. The chosen file drags
             straight into Blender / Three.js / Unreal. The build
             runs on a worker thread; the folder pick happens here,
             on the bridge thread, like pick_jobs_root."""
        if self._window is None:
            self.status = "导出需要先选一个文件夹，窗口还没准备好。"
            self._log("导出通用格式：窗口还没准备好，稍后再试。")
            return {"ok": False}
        import webview

        paths = self._window.create_file_dialog(
            webview.FOLDER_DIALOG)
        if not paths:
            self.status = "已取消导出。"
            self._log("导出通用格式：已取消。")
            return {"ok": False}
        dest_dir = Path(paths[0])
        if not self._spawn_export(self._run_export, job_dir, dest_dir, fmt):
            return {"ok": False}
        return {"ok": True}

    def _run_export(self, job_dir, dest_dir, fmt="glb"):
        job = self._job_for(job_dir)
        if job is None:
            self.status = "没有选中的任务，先做完一个场景。"
            self._log("导出：没有选中的任务。")
            return
        job = Path(job)
        glb = job / "voxel_color.glb"
        ply = job / "voxel_color.ply"
        if not glb.exists() or not ply.exists():
            self.status = "正在固化导出文件…"
            self._log("导出：正在把场景固化成带颜色的体积文件…")
            try:
                from echo_scene.voxel_export import (
                    export_colored_voxels)
                info = export_colored_voxels(job)
            except Exception as exc:
                self.status = "导出失败，看一下日志。"
                self._log(f"导出固化失败：{exc}")
                return
            if not info.get("ok"):
                self.status = "这个任务还导不出，看一下日志。"
                self._log("导出固化失败："
                          f"{info.get('reason', '未知原因')}")
                return
            self._log(f"固化好了：{info.get('voxels', '?')} 个体素，"
                      f"{info.get('units', '')}")
        # One button, one format: only the chosen file is copied
        # out (plus the info file). An unknown fmt falls back to
        # GLB rather than failing the export.
        fmt = fmt if fmt in ("glb", "ply") else "glb"
        self._export_copy_out(job, glb, ply, dest_dir, fmt)

    def _export_copy_out(self, job, glb, ply, dest_dir, fmt="glb"):
        """"通用格式": copy only the chosen file (+ info) to a
        folder the user picked. Never raises; the outcome lands in
        the log."""
        import shutil

        try:
            dest_dir = Path(dest_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            chosen = glb if fmt == "glb" else ply
            copied = []
            for src in (chosen, job / "voxel_color_info.json"):
                if src.exists():
                    shutil.copy2(src, dest_dir / src.name)
                    copied.append(src.name)
            if not copied:
                raise OSError("没有可复制的文件")
            self.status = f"已导出到 {dest_dir}"
            self._log("导出通用格式：已复制到 "
                      f"{dest_dir}：" + "、".join(copied))
        except Exception as exc:
            self.status = "导出失败，看一下日志。"
            self._log(f"导出通用格式失败：{exc}")



    def export_camera_route(self, job_dir=None):
        """Export the recovered camera route. The build (poses ->
        route JSON + GLB) runs on a worker thread; it is quick --
        no reconstruction -- but the thread keeps the UI honest
        all the same. Nothing fails silently."""
        if not self._spawn_export(self._run_route_export, job_dir):
            return {"ok": False}
        return {"ok": True}

    def _run_route_export(self, job_dir):
        job = self._job_for(job_dir)
        if job is None:
            self.status = "没有选中的任务，先做完一个场景。"
            self._log("导出相机路线：没有选中的任务。")
            return
        job = Path(job)
        self.status = "正在导出相机路线…"
        self._log("导出相机路线：开始（相机位姿 → 路线文件）…")
        try:
            from echo_scene.voxel_export import export_camera_route
            info = export_camera_route(job, log=self._log)
        except Exception as exc:
            self.status = "导出失败，看一下日志。"
            self._log(f"导出相机路线失败：{exc}")
            return
        if not info.get("ok"):
            self.status = "这个任务还导不出相机路线，看一下日志。"
            self._log("导出相机路线失败："
                      f"{info.get('reason', '未知原因')}")
            return
        glb = Path(info["files"]["glb"])
        _reveal_file(glb)
        self.status = "导出好了：camera_route.glb + camera_route.json"
        self._log(f"导出相机路线：{info.get('frame_count', '?')} 帧，"
                  f"路线 {info.get('route_length', 0):.2f} "
                  f"{info.get('units', '')}，"
                  f"{info.get('skipped_frames', 0)} 帧跳过。"
                  f"文件：{glb}，{info['files']['json']}")

    def export_batch(self, job_dirs=None, dest_dir=None):
        """Export several finished jobs in one run. job_dirs:
        the job folders the user ticked in the history list. The
        output folder is picked here, on the bridge thread, like
        the "general" export's folder pick; the build itself runs
        on a worker thread. Nothing fails silently."""
        jobs = [str(d) for d in (job_dirs or []) if d]
        if not jobs:
            self.status = "先在历史场景里勾上要导出的任务。"
            self._log("批量导出：一个任务都没勾上。")
            return {"ok": False}
        if dest_dir is None:
            if self._window is None:
                self.status = "导出需要先选一个文件夹，窗口还没准备好。"
                self._log("批量导出：窗口还没准备好，稍后再试。")
                return {"ok": False}
            import webview

            paths = self._window.create_file_dialog(
                webview.FOLDER_DIALOG)
            if not paths:
                self.status = "已取消批量导出。"
                self._log("批量导出：已取消。")
                return {"ok": False}
            dest_dir = Path(paths[0])
        if not self._spawn_export(self._run_batch_export, jobs, dest_dir):
            return {"ok": False}
        return {"ok": True}

    def _run_batch_export(self, job_dirs, dest_dir):
        dest_dir = Path(dest_dir)
        self.status = f"正在批量导出 {len(job_dirs)} 个任务…"
        self._log(f"批量导出：开始，{len(job_dirs)} 个任务，"
                  f"文件夹：{dest_dir}")
        try:
            from echo_scene.voxel_export import export_batch
            info = export_batch(job_dirs, dest_dir, log=self._log)
        except Exception as exc:
            self.status = "批量导出失败，看一下日志。"
            self._log(f"批量导出失败：{exc}")
            return
        if not info.get("ok"):
            self.status = "批量导出失败，看一下日志。"
            self._log("批量导出失败："
                      f"{info.get('reason', '未知原因')}")
            return
        done = info.get("exported", 0)
        skipped = info.get("skipped", 0)
        summary = dest_dir / "batch_export.json"
        if summary.exists():
            _reveal_file(summary)
        if skipped:
            self.status = (f"批量导出好了：{done} 个任务，"
                           f"{skipped} 个跳过，看一下日志。")
        else:
            self.status = f"批量导出好了：{done} 个任务。"
        self._log(f"批量导出：{done} 个任务导好了，{skipped} 个跳过。"
                  f"文件夹：{dest_dir}，汇总：batch_export.json")

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

    # -- auto-update gate + export bookkeeping (v0.6.30) --------------
    def _update_in_flight(self):
        """True while an update is downloading/staging/applying/
        restarting (v0.6.76): the job-start guard reads this, so
        no new job or export can start mid-update. The apply-time
        re-check in _hot is the other half -- refusing up front
        tells the user at once, instead of the update dying
        silently at apply time."""
        if getattr(self, "_update_running", False):
            return True
        phase = self._update_phase
        return phase is not None and phase.get("phase") in (
            "downloading", "staging", "applying", "restarting")

    def _work_running(self):
        """True while a scene/depth job (busy) or an export is
        running (v0.6.76) -- the work half of _is_idle, without
        the update's own in-flight state. The _hot apply-time
        re-check reads this: _is_idle is False for the whole
        update by design, so it cannot tell work from the update
        itself."""
        if self.busy:
            return True
        lock = getattr(self, "_export_lock", None)
        if lock is not None:
            with lock:
                return self._export_busy > 0
        return getattr(self, "_export_busy", 0) > 0

    def _is_idle(self):
        """True when nothing is running: no scene/depth job (busy),
        no export (exports run on worker threads without setting
        busy, so they are counted separately), and no update
        download/apply in progress. The auto-update gate reads
        this, not self.busy alone."""
        if self.busy:
            return False
        # v0.6.38 (U-10): read under the export lock, matching
        # the guarded increment/decrement in _spawn_export.
        lock = getattr(self, "_export_lock", None)
        if lock is not None:
            with lock:
                export_busy = self._export_busy
        else:
            export_busy = getattr(self, "_export_busy", 0)
        if export_busy > 0:
            return False
        if getattr(self, "_update_running", False):
            return False
        phase = self._update_phase
        if phase is not None and phase.get("phase") in (
                "downloading", "staging", "applying", "restarting"):
            return False
        return True

    def _spawn_export(self, target, *args):
        """Run an export worker and count it, so the auto-update
        gate knows an export is running. The counter drops in a
        finally -- an export that raises still clears it.
        v0.6.76: no new export while an update is in flight --
        the apply-time re-check in _hot is the other half, but
        refusing up front keeps the export from starting in the
        half-second between the re-check and the restart.
        Returns False when refused, True when spawned; callers
        answer the page from that."""
        if self._update_in_flight():
            self.status = "正在更新，等更新完再导出。"
            self._log("导出先等等：正在更新，等更新完再导出。")
            return False
        def _run():
            # v0.6.38 (U-10): += / -= across threads is not
            # atomic -- a lost update could leave the auto-update
            # gate believing an export is (or isn't) running.
            # The lock makes the count airtight.
            with self._export_lock:
                self._export_busy += 1
            try:
                target(*args)
            finally:
                with self._export_lock:
                    self._export_busy -= 1
        threading.Thread(target=_run, daemon=True).start()
        return True

    # -- settings + updates ----------------------------------------------
    @staticmethod
    def _validate_jobs_root(path):
        """None when the path is usable as the jobs root, else a
        plain-language reason (U-6). A missing folder is created
        here -- the settings page already lets the user pick a
        new one; a folder that cannot be created, or cannot be
        written to, is refused up front instead of breaking
        later jobs at run time. The write probe is a temp file
        that is deleted on close, so nothing is left behind."""
        if path.exists():
            if not path.is_dir():
                return "这个路径不是文件夹，换一个文件夹试试。"
        else:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError:
                return "这个文件夹建不出来，换一个文件夹试试。"
        try:
            with tempfile.TemporaryFile(dir=path):
                pass
        except OSError:
            return "这个文件夹写不进去，换一个文件夹试试。"
        return None

    def save_settings(self, data):
        try:
            if "jobsRoot" in data and data["jobsRoot"]:
                # v0.6.38 (U-6): a bad jobs folder used to be
                # stored and only break later jobs at run time.
                # Validate first -- a refusal saves nothing, so
                # the old folder stays in force.
                root = Path(str(data["jobsRoot"]))
                err = self._validate_jobs_root(root)
                if err:
                    self.status = err
                    return {"ok": False, "message": err}
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
            if "autoUpdate" in data:
                self.settings.auto_update = bool(data["autoUpdate"])
            if "updateProxy" in data:
                self.settings.update_proxy = str(data["updateProxy"]).strip()
                try:
                    from echo_app.updater import set_proxy
                    set_proxy(self.settings.update_proxy)
                except Exception:
                    pass
            if "windowMode" in data:
                mode = str(data["windowMode"])
                if mode in ("windowed", "borderless", "fullscreen"):
                    self.settings.window_mode = mode
            if "textContrast" in data:
                mode = str(data["textContrast"])
                if mode in ("standard", "high"):
                    self.settings.text_contrast = mode
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
        """Startup check + start the silent 10-minute auto-check
        loop (v0.6.29): the loop re-checks every ten minutes and
        stays silent unless the check cannot reach the update
        source at all -- _run_check logs only that failure (and a
        found update); a reachable source with no new version logs
        nothing. The manual button keeps its v0.6.25 behaviour
        (re-check on click, always answer)."""
        if not self._auto_check_started:
            self._auto_check_started = True
            threading.Thread(target=self._auto_check_loop,
                             daemon=True).start()
        threading.Thread(target=lambda: self._run_check(False),
                         daemon=True).start()
        return {"ok": True}

    def _auto_check_loop(self):
        while not self._auto_check_stop.is_set():
            self._auto_check_stop.wait(AUTO_CHECK_SECONDS)
            if self._auto_check_stop.is_set():
                break
            # v0.6.30: only the loop may auto-update (auto=True).
            # The startup check and the 90s retry only report.
            self._run_check(False, auto=True)

    def manual_check_update(self):
        """The settings-page check, now off the UI thread (v0.6.29):
        the old synchronous call ran the whole route probe on the
        bridge thread -- up to ~30 s of frozen window -- before
        answering. This returns at once; the answer lands in poll()
        as manualCheck (with a sequence number) and the page
        renders the note from there."""
        if getattr(self, "_manual_checking", False):
            return {"ok": True, "checking": True}
        self._manual_checking = True
        threading.Thread(target=self._manual_check_worker,
                         daemon=True).start()
        return {"ok": True, "checking": True}

    def _manual_check_worker(self):
        try:
            result = self._run_check(True)
        except Exception:
            result = {"status": "error"}
        finally:
            self._manual_checking = False
        self._manual_check_seq += 1
        self._manual_check_result = {"seq": self._manual_check_seq,
                                     **result}

    def _run_check(self, manual, auto=False):
        # One check at a time (v0.6.29): the 10-minute loop, the
        # 90-second startup retry and a manual check can otherwise
        # pile up on the same slow route probe.
        with self._check_lock:
            return self._run_check_locked(manual, auto=auto)

    def _run_check_locked(self, manual, auto=False):
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
                # v0.6.30: auto-update, only from the 10-minute
                # loop (auto=True), only for a hot update, only
                # when idle. A job running means: wait for the
                # next cycle, never interrupt it. A missing hot
                # path means: leave the update box for the user
                # (a 657 MB installer is never downloaded on its
                # own). The restart is explained: the status line
                # and the update progress say it is happening.
                if auto and self.settings.auto_update \
                        and getattr(self, "_hot_info", None) is not None:
                    if getattr(self, "_update_running", False):
                        pass  # an update is already running
                    elif self._is_idle():
                        self._log(f"发现新版 {self.update['version']}，"
                                  "现在没有任务，正在自动更新…")
                        self.status = "发现新版，正在自动更新…"
                        self.download_update(auto=True)
                    else:
                        self._log(f"发现新版 {self.update['version']}，"
                                  "正在忙，等下次检查时再更新。")
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

    def _set_update_progress(self, phase, frac=None, text=""):
        """Download/apply progress for the page (v0.6.29): phase in
        {"downloading", "staging", "applying", "restarting",
        "done", "failed"}; frac 0..1 when the download knows its
        total, else None (the page animates the bar
        indeterminately, so there is always visible motion)."""
        self._update_phase = {"phase": phase, "frac": frac, "text": text}

    def download_update(self, auto=False):
        # One update at a time: an auto-update racing a manual
        # click (or two loop ticks) must not download twice.
        if getattr(self, "_update_running", False):
            return {"ok": True}
        # v0.6.37: never apply an update while anything is
        # running. The auto path already checks _is_idle() before
        # calling here, but the manual button did not -- a click
        # mid-job staged the pack, asked the launcher to apply
        # it and restarted, killing the running scene/depth job
        # (and any export) outright. The check is repeated here
        # so both paths, and the race between the check and the
        # click, are covered: busy means wait, never interrupt.
        if not self._is_idle():
            self.status = "正在做任务，等做完再更新。"
            self._log("更新先等等：正在做任务，等做完再更新。")
            return {"ok": False, "message": "正在做任务，等做完再更新。"}
        hot = getattr(self, "_hot_info", None)
        if hot is not None:
            self._update_running = True
            def _hot():
                try:
                    import subprocess

                    from echo_app import hotupdate
                    root = hotupdate.install_root()
                    self._set_update_progress(
                        "downloading", 0.0, "正在下载更新…")
                    hotupdate.download_and_stage(
                        root, hot, progress=lambda done, total:
                        self._set_update_progress(
                            "downloading",
                            (done / total) if total else None,
                            "正在下载更新… "
                            f"{round(done / 1e6, 1)} MB"
                            + (f" / 约 {round(total / 1e6, 1)} MB"
                               if total else "")))
                    # v0.6.76: re-check work right before
                    # apply -- a job (or export) may have started
                    # during the download+stage window, after
                    # download_update's entry check passed.
                    # Restarting now would kill it mid-run, exactly
                    # what v0.6.37 fixed at the entry. Abort this
                    # attempt instead: the staged pack stays,
                    # _hot_info stays, the next check (or a manual
                    # click) retries. Never interrupt running work
                    # for an update. Reads _work_running, not
                    # _is_idle: _is_idle is False for the whole
                    # update by design.
                    if self._work_running():
                        self._update_running = False
                        self._set_update_progress(
                            "failed", None, "有任务在跑，这次先不更新。")
                        self.status = ("有任务在跑，这次先不更新，"
                                       "等做完再更新。")
                        self._log("更新先等等：下载时开始了新任务，"
                                  "这次先不重启，等下次检查。")
                        return
                    self._set_update_progress(
                        "applying", None, "正在应用更新…")
                    cmd = hotupdate.request_apply(root, hot)
                    self._set_update_progress(
                        "restarting", None, "正在重启完成更新…")
                    self._log("更新已备好，正在重启完成更新…")
                    subprocess.Popen(cmd, close_fds=True)
                    threading.Timer(0.4, self._quit,
                                    kwargs={"restart": True}).start()
                except Exception:
                    self._hot_info = None
                    self._update_running = False
                    if auto:
                        # An automatic update never falls back to
                        # the full installer on its own: a 657 MB
                        # download the user did not ask for. It
                        # says so and leaves the button for them.
                        self._set_update_progress(
                            "failed", None, "自动更新没成，点一下更新按钮手动更新。")
                        self.status = "自动更新没成，点一下更新按钮手动更新。"
                        self._log("自动更新没成（热更新失败），等手动更新。")
                        return
                    self._set_update_progress(
                        "failed", None, "热更新没成，改走完整安装包。")
                    self._log("热更新没成，改走完整安装包。")
                    # Let the failure show for a beat before the
                    # fallback starts: otherwise the installer
                    # download begins at once and the user never
                    # sees why a small update became a big one.
                    # This is the worker thread, never the UI.
                    time.sleep(1.2)
                    self.download_update()
            threading.Thread(target=_hot, daemon=True).start()
            return {"ok": True}
        info = getattr(self, "_update_info", None)
        if info is None:
            # No stored result: the startup check ran before any
            # newer release existed (or never ran at all).
            # Re-check now in the background instead of answering
            # silently -- this button must always say what happened.
            if getattr(self, "_update_checking", False):
                return {"ok": True}

            def _recheck():
                try:
                    self.status = "正在检查更新…"
                    self._log("正在重新检查更新…")
                    result = self._run_check(True)
                    if getattr(self, "_hot_info", None) is not None \
                            or getattr(self, "_update_info", None) is not None:
                        self.download_update()
                        return
                    if result.get("status") == "latest":
                        self.status = "已经是最新版本。"
                        self._log("已经是最新版本。")
                    else:
                        self.status = "更新检查没连上，看一下日志。"
                finally:
                    self._update_checking = False

            self._update_checking = True
            threading.Thread(target=_recheck, daemon=True).start()
            return {"ok": True}

        self._update_running = True

        def _download():
            try:
                from echo_app.updater import download_installer
                self._set_update_progress(
                    "downloading", 0.0, "正在下载安装包…")
                dest = download_installer(
                    info, self.jobs_root,
                    progress=lambda done, total:
                    self._set_update_progress(
                        "downloading",
                        (done / total) if total else None,
                        "正在下载安装包… "
                        f"{round(done / 1e6)} MB"
                        + (f" / 约 {round(total / 1e6)} MB"
                           if total else "")))
                self._set_update_progress(
                    "done", 1.0, "新版本安装包已下载，正在打开。")
                self._log("新版本安装包已下载，正在打开。")
                backend.play_file(Path(dest))
            except Exception:
                self._set_update_progress(
                    "failed", None, "更新下载失败，看一下日志。")
                self._log("更新下载失败。")
            finally:
                self._update_running = False
        threading.Thread(target=_download, daemon=True).start()
        return {"ok": True}

    def _quit(self, restart=False):
        # The file log must record how the process ended. A clean
        # quit writes "已退出。" here; an update restart is already
        # logged by its caller ("更新已备好，正在重启…"), so it
        # writes nothing -- the last lines must tell a restart
        # apart from a quit apart from a crash (a crash leaves no
        # shutdown line at all).
        if not restart:
            try:
                self._log("已退出。")
            except Exception:
                pass
        # v0.6.42: window.destroy() can hang (2026-11-07: the
        # log said "已退出。" but PID 65592 stayed alive, window
        # gone, process stuck -- destroy() never returned, so
        # os._exit never ran and the close button looked dead).
        # Destroy on a daemon thread so it can never block the
        # exit; the process ends here no matter what.
        try:
            import threading

            def _destroy():
                try:
                    if self._window is not None:
                        self._window.destroy()
                except Exception:
                    pass

            threading.Thread(target=_destroy, daemon=True).start()
        except Exception:
            pass
        import os
        os._exit(0)

    # -- window controls (the window is frameless; the page carries
    # its own caption buttons in the same glass language) --------------
    def win_minimize(self):
        # v0.6.42: pywebview's minimize() silently does nothing
        # on this frameless window (no log, no error, window
        # stays). Minimise with the same direct Win32 call the
        # maximise path uses, and log the result so a click that
        # does nothing leaves a reason instead of silence.
        try:
            import ctypes

            hwnd = self._window_hwnd()
            if hwnd:
                ok = bool(ctypes.windll.user32.ShowWindow(
                    hwnd, _SW_MINIMIZE))
                self._log(f"最小化：ShowWindow={ok}。")
                if ok:
                    return {"ok": True}
                self._log("最小化没生效，改试窗口自带最小化。")
        except Exception as exc:
            self._log(f"最小化出错：{exc}")
        try:
            if self._window is not None:
                self._window.minimize()
        except Exception:
            pass
        return {"ok": True}

    def _window_hwnd(self):
        """The native window handle, same route _polish_window uses."""
        window = self._window
        if window is None:
            return None
        try:
            native = getattr(window, "native", None)
            if native is not None:
                return int(native.Handle)
        except Exception:
            pass
        try:
            import ctypes

            return ctypes.windll.user32.FindWindowW(
                None, "ECHO 回声") or None
        except Exception:
            return None

    def _resize_hook_status(self):
        """Live resize-hook state (v0.6.30): the window handle,
        whether the hook is installed, whether the procedure
        currently installed is ours, and the last install attempt.
        (v0.6.31: also the top-level rect and the WebView2 child
        hook state -- per child: handle, class, installed,
        procIsOurs -- and the last child install attempt.) Off
        Windows the hook does not exist (installed False)."""
        hwnd = self._window_hwnd()
        installed = False
        proc_is_ours = False
        rect = None
        child_entries = []
        child_installed = False
        if hwnd is not None and sys.platform == "win32":
            try:
                current = _current_wndproc(hwnd)
                proc_is_ours = _resize_proc_is_ours(
                    current, _callback_addr(_resize_callback))
                installed = proc_is_ours and _resize_hook_hwnd == hwnd
            except Exception:
                pass
            try:
                facade = _real_win32_facade()
                if facade is not None:
                    rect = facade.get_window_rect(hwnd)
                    for ch_hwnd, entry in _resize_child_hooks.items():
                        try:
                            cur = facade.get_proc(ch_hwnd)
                        except Exception:
                            cur = None
                        ours = _resize_proc_is_ours(
                            cur, _callback_addr(entry.get("callback")))
                        child_entries.append({
                            "hwnd": ch_hwnd,
                            "class": entry.get("class", ""),
                            "installed": bool(ours),
                            "procIsOurs": bool(ours)})
                        child_installed = child_installed or bool(ours)
            except Exception:
                pass
        return {"hwnd": hwnd, "installed": installed,
                "procIsOurs": proc_is_ours,
                "windowRect": rect,
                "child": {"installed": child_installed,
                          "children": child_entries,
                          "lastAttempt": dict(_resize_child_last_attempt)},
                "lastAttempt": dict(_resize_last_attempt)}

    def _window_rect(self):
        """Current window rect as (left, top, width, height), or None."""
        if sys.platform != "win32":
            return None
        try:
            import ctypes
            from ctypes import wintypes

            hwnd = self._window_hwnd()
            if not hwnd:
                return None
            rect = wintypes.RECT()
            if not ctypes.windll.user32.GetWindowRect(
                    hwnd, ctypes.byref(rect)):
                return None
            return (rect.left, rect.top,
                    rect.right - rect.left, rect.bottom - rect.top)
        except Exception:
            return None

    def _work_area_rect(self):
        """Work area of the monitor the window sits on, as
        (left, top, width, height), or None."""
        if sys.platform != "win32":
            return None
        try:
            import ctypes
            from ctypes import wintypes

            hwnd = self._window_hwnd()
            if not hwnd:
                return None
            user32 = ctypes.windll.user32
            MONITOR_DEFAULTTONEAREST = 2
            hmon = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)

            class MONITORINFO(ctypes.Structure):
                _fields_ = [("cbSize", wintypes.DWORD),
                            ("rcMonitor", wintypes.RECT),
                            ("rcWork", wintypes.RECT),
                            ("dwFlags", wintypes.DWORD)]

            mi = MONITORINFO()
            mi.cbSize = ctypes.sizeof(MONITORINFO)
            if hmon and user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
                w = mi.rcWork
                return (w.left, w.top,
                        w.right - w.left, w.bottom - w.top)
            work = wintypes.RECT()
            SPI_GETWORKAREA = 0x0030
            if user32.SystemParametersInfoW(
                    SPI_GETWORKAREA, 0, ctypes.byref(work), 0):
                return (work.left, work.top,
                        work.right - work.left, work.bottom - work.top)
            return None
        except Exception:
            return None

    @staticmethod
    def _wh_rects_match(a, b, tol=2):
        """True when two (left, top, width, height) rects agree
        within a few pixels (DPI rounding)."""
        if a is None or b is None:
            return False
        return (abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol
                and abs(a[2] - b[2]) <= tol and abs(a[3] - b[3]) <= tol)

    def _is_maximized(self):
        """True when the window really fills the work area and we
        got there via the maximise button (a saved rect exists),
        or Windows reports it zoomed (style check). A window the
        user merely dragged to work-area size does not count, so
        the button can still maximise it properly."""
        if sys.platform != "win32":
            return False
        try:
            import ctypes

            work = self._work_area_rect()
            now = self._window_rect()
            if (self._wh_rects_match(now, work)
                    and getattr(self, "_pre_max_rect", None)):
                return True
            hwnd = self._window_hwnd()
            if hwnd:
                return bool(ctypes.windll.user32.IsZoomed(hwnd))
            return False
        except Exception:
            return False

    def _maximize_work_area(self):
        """Fill the monitor's work area (taskbar stays visible).

        v0.6.36 (web research, 2026-11-06): the proven pattern
        for a frameless window is a REAL maximise -- ShowWindow
        SW_MAXIMIZE -- after a WM_GETMINMAXINFO handler tells
        Windows the work area (installed in _resize_wndproc).
        SetWindowPos to the work area is only a pseudo-maximise:
        its size is clamped by ptMaxTrackSize, which is why the
        click used to move the window to the top-left without
        resizing it. The SetWindowPos path stays only as a
        fallback. Returns True only when the window really ends
        up filling the work area -- the rect is read back and
        checked, and every step logs the real numbers."""
        if sys.platform != "win32":
            return False
        try:
            import ctypes

            hwnd = self._window_hwnd()
            if not hwnd:
                self._log("最大化失败：拿不到窗口句柄。")
                return False
            now = self._window_rect()
            if not now:
                self._log("最大化失败：读不到窗口位置。")
                return False
            # save the rect so restore can put the window back
            self._pre_max_rect = now
            work = self._work_area_rect()
            if not work:
                self._log("最大化失败：读不到工作区。")
                return False
            self._log(f"最大化：现在窗口={now}，工作区={work}。")
            # make sure the maximise styles are on (WM_GETMINMAXINFO
            # is only sent to a window with WS_THICKFRAME/CAPTION)
            _ensure_maximize_styles(hwnd)
            # 1. the real maximise -- Windows uses the work area
            # our WM_GETMINMAXINFO handler just handed it.
            ctypes.windll.user32.ShowWindow(hwnd, _SW_MAXIMIZE)
            after = self._window_rect()
            zoomed = bool(ctypes.windll.user32.IsZoomed(hwnd))
            if self._wh_rects_match(after, work) or zoomed:
                self._log(f"已最大化到工作区：窗口={after}。")
                return True
            self._log(
                f"系统最大化没铺满：窗口={after}，目标={work}，"
                f"改试直接摆到工作区。")
            # 2. fallback: put the window on the work area by hand.
            ok = bool(ctypes.windll.user32.SetWindowPos(
                hwnd, 0, work[0], work[1], work[2], work[3],
                _SWP_NOZORDER | _SWP_SHOWWINDOW))
            after2 = self._window_rect()
            if ok and self._wh_rects_match(after2, work):
                self._log(f"直接摆到工作区生效：窗口={after2}。")
                return True
            self._log(
                f"最大化没生效：SetWindowPos={ok}，"
                f"现在窗口={after2}，目标={work}。")
            return False
        except Exception as exc:
            self._log(f"最大化失败：{exc}")
            return False

    def _restore_work_area(self):
        """Put the window back to its pre-maximise size. The real
        restore first (ShowWindow SW_RESTORE -- Windows saved the
        normal rect when it maximised), the manual SetWindowPos
        to the saved rect as a fallback. (v0.6.36)"""
        if sys.platform != "win32":
            return False
        try:
            import ctypes

            hwnd = self._window_hwnd()
            if not hwnd:
                self._log("还原失败：拿不到窗口句柄。")
                return False
            rect = getattr(self, "_pre_max_rect", None)
            if not rect:
                self._log("还原失败：没有保存的窗口位置。")
                return False
            ctypes.windll.user32.ShowWindow(hwnd, _SW_RESTORE)
            after = self._window_rect()
            if self._wh_rects_match(after, rect):
                self._log("已还原到原来大小。")
                return True
            ok = bool(ctypes.windll.user32.SetWindowPos(
                hwnd, 0, rect[0], rect[1], rect[2], rect[3],
                _SWP_NOZORDER | _SWP_SHOWWINDOW))
            after2 = self._window_rect()
            if ok and self._wh_rects_match(after2, rect):
                self._log("已还原到原来大小。")
                return True
            self._log(
                f"还原没生效：SetWindowPos={ok}，"
                f"现在窗口={after2}，目标={rect}。")
            return False
        except Exception as exc:
            self._log(f"还原失败：{exc}")
            return False

    def win_toggle_max(self):
        # "borderless" (the default) is the one the user asked for:
        # maximise the frameless window to the work area, taskbar
        # stays visible -- NOT a video fullscreen. "windowed" is
        # the plain maximise; "fullscreen" never reaches here, the
        # page routes it to the DOM fullscreen API instead.
        # v0.6.36: the maximise itself is now a real maximise
        # (ShowWindow SW_MAXIMIZE, work area from the
        # WM_GETMINMAXINFO handler) with the manual
        # SetWindowPos as a fallback; the toggle still reads
        # the real window rect, never the stored flag.
        # v0.6.34: the toggle no longer trusts the stored
        # _maximized flag -- it reads the real window rect and
        # compares it with the work area, so a failed maximise
        # can never leave the button stuck doing the opposite.
        # Every step is logged, so a click that does nothing
        # leaves a reason in app.log instead of silence.
        mode = getattr(self.settings, "window_mode", "borderless")
        try:
            if self._window is None:
                return {"ok": True, "maximized": False}
            if self._is_maximized():
                self._log("点最大化：窗口已是铺满，改为还原。")
                if mode == "borderless":
                    done = self._restore_work_area()
                    if not done:
                        self._window.restore()
                        done = not self._is_maximized()
                else:
                    self._window.restore()
                    done = not self._is_maximized()
                self._maximized = not done
                if not done:
                    self._log("还原没生效，仍按铺满记。")
            else:
                self._log("点最大化：铺满工作区。")
                if mode == "borderless":
                    done = self._maximize_work_area()
                    if not done:
                        self._window.maximize()
                        done = self._is_maximized()
                        if done:
                            self._log("窗口自带最大化生效。")
                else:
                    self._window.maximize()
                    done = self._is_maximized()
                self._maximized = done
                if not done:
                    self._log("最大化没生效，下次点会再试铺满。")
            # v0.6.42: nudge the page to reflow to the new size.
            # The window grows but the page can stay at the old
            # size, looking like "only moved to the corner".
            # SWP_FRAMECHANGED forces a frame recalc + layout.
            try:
                import ctypes

                _hwnd = self._window_hwnd()
                if _hwnd:
                    ctypes.windll.user32.SetWindowPos(
                        _hwnd, 0, 0, 0, 0, 0,
                        _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOZORDER
                        | _SWP_FRAMECHANGED)
            except Exception:
                pass
            # v0.6.57: no direct child resize here any more. The
            # maximise/restore above already generated a WM_SIZE,
            # and the window procedure queues the child resize for
            # the next message-loop turn (see
            # _post_resize_browser_child) -- the frame nudge plus
            # the deferred resize move the WebView2 control, which
            # is why the page follows after maximise. Calling
            # SetWindowPos on the child synchronously, from inside
            # the sizing that just ran, is what the OS reports as
            # STATUS_FATAL_USER_CALLBACK_EXCEPTION (c000041d) when
            # the child's procedure throws: the 2026-11-08 20:59
            # desktop crash on maximise->restore.
            return {"ok": True, "maximized": bool(self._maximized)}
        except Exception as exc:
            self._log(f"最大化出错：{exc}")
            return {"ok": True, "maximized": bool(
                getattr(self, "_maximized", False))}

    def win_close(self):
        self._quit()
        return {"ok": True}


# ================= CHROME-LIKE RESIZE (v0.6.28) =================
# A frameless window has no frame for Windows to grab, so out of
# the box it cannot be resized by dragging an edge or a corner --
# the one thing Chrome does that this window did not. The fix is
# the same one every frameless Win32 window uses: answer
# WM_NCHITTEST with the resize hit codes for a border band
# (HTLEFT / HTRIGHT / HTTOP / HTBOTTOM and the four corners).
# Windows then runs the resize drag itself -- no Python code runs
# during the drag, so it tracks the cursor exactly like Chrome,
# including the diagonal corner.
#
# The hook subclasses the window procedure, installed the same way
# _polish_window is (at start, on shown, and on two delayed timers,
# since the native handle may not exist at first call), and chained
# to whatever procedure was there before, so pywebview's own
# handling is untouched. Windows-only, fully guarded: a failure to
# install costs nothing -- the window simply keeps its old
# behaviour. The pure geometry (_hit_test_resize) is factored out
# so it can be unit-tested on any platform; the Win32 half cannot
# run on this Linux VM and is verified by code review only, and
# needs a real-desktop check.
#
# When the window fills its frame -- the monitor work area (the
# borderless-maximized mode) or the whole monitor (fullscreen) --
# the border band stops offering resize, exactly like a maximized
# Chrome window: restore first, then drag. The check is stateless
# (window rect vs the monitor's work/full rects), so it stays true
# no matter which path maximized the window.
# ------------------------------------------------------------------
_RESIZE_BORDER = 8  # px: the band at each edge that grabs a resize

_HTCLIENT = 1
_HTLEFT = 10
_HTRIGHT = 11
_HTTOP = 12
_HTTOPLEFT = 13
_HTTOPRIGHT = 14
_HTBOTTOM = 15
_HTBOTTOMLEFT = 16
_HTBOTTOMRIGHT = 17

_WM_NCHITTEST = 0x0084
_GWLP_WNDPROC = -4
_WM_NCCALCSIZE = 0x0083
_WM_GETMINMAXINFO = 0x0024
_GWL_STYLE = -16
_WS_THICKFRAME = 0x00040000
_WS_MAXIMIZEBOX = 0x00010000
_WS_MINIMIZEBOX = 0x00020000
_SW_MAXIMIZE = 3
_SW_RESTORE = 9
_SW_MINIMIZE = 6
_SWP_NOMOVE = 0x0002
_SWP_NOSIZE = 0x0001
_SWP_NOZORDER = 0x0004
_SWP_SHOWWINDOW = 0x0040
_SWP_FRAMECHANGED = 0x0020
_WM_SYSCOMMAND = 0x0112
_WM_SIZE = 0x0005
_WM_DROPFILES = 0x0233
# v0.6.57: private message for the deferred browser-child resize
# (WM_APP + 1). Posted by the WM_SIZE branch, handled in
# _resize_wndproc_inner, swallowed there -- no other procedure
# ever sees it.
_WM_RESIZE_CHILD_DEFERRED = 0x8001
_SC_MINIMIZE = 0xF020
_SC_RESTORE = 0xF120
_SC_MAXIMIZE = 0xF030
_SIZE_RESTORED = 0
_SIZE_MINIMIZED = 1
_SIZE_MAXIMIZED = 2

# hwnds already given the maximise styles (once per window).
_maximize_styles_done = set()

# the subclass chain: the previous procedure (an address) and the
# callback object, which MUST stay alive in this module -- if it is
# garbage-collected, the next WM_NCHITTEST crashes the process.
_resize_orig_proc = None
_resize_hook_hwnd = None
_resize_callback = None
_resize_argtypes_set = False

# v0.6.57: a deferred child resize is already queued (see
# _post_resize_browser_child). WM_SIZE fires repeatedly during a
# drag-resize; coalescing keeps the queue from filling with
# duplicate resize requests. Only ever touched on the UI thread
# (the wndproc and the deferred-message handler both run there).
_resize_child_pending = False


def _hit_test_resize(x, y, rect, border=_RESIZE_BORDER, frames=()):
    """Pure geometry: the resize hit code for a cursor at (x, y) in
    screen pixels against window rect (left, top, right, bottom).
    frames are the rects the window fills when it must NOT resize
    (monitor work area / full monitor); when the window fills one
    of them, or the cursor is not in the border band, HTCLIENT."""
    left, top, right, bottom = rect
    for frame in frames:
        fl, ft, fr, fb = frame
        if (abs(left - fl) <= 2 and abs(top - ft) <= 2
                and abs(right - fr) <= 2 and abs(bottom - fb) <= 2):
            return _HTCLIENT
    at_left = x < left + border
    at_right = x >= right - border
    at_top = y < top + border
    at_bottom = y >= bottom - border
    if at_top and at_left:
        return _HTTOPLEFT
    if at_top and at_right:
        return _HTTOPRIGHT
    if at_bottom and at_left:
        return _HTBOTTOMLEFT
    if at_bottom and at_right:
        return _HTBOTTOMRIGHT
    if at_left:
        return _HTLEFT
    if at_right:
        return _HTRIGHT
    if at_top:
        return _HTTOP
    if at_bottom:
        return _HTBOTTOM
    return _HTCLIENT


def _signed_short(v):
    """Low/high word of lParam as a signed 16-bit value (screen
    coordinates go negative on monitors left/above the primary)."""
    import ctypes

    return ctypes.c_short(v & 0xFFFF).value


def _window_frame_rects(hwnd):
    """(work area, full monitor) for the monitor the window sits on,
    as (left, top, right, bottom) tuples. None on any failure --
    the caller then treats the window as resizable."""
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        MONITOR_DEFAULTTONEAREST = 2
        hmon = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)

        class MONITORINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD),
                        ("rcMonitor", wintypes.RECT),
                        ("rcWork", wintypes.RECT),
                        ("dwFlags", wintypes.DWORD)]

        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)
        if hmon and user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
            work = mi.rcWork
            mon = mi.rcMonitor
            return ((work.left, work.top, work.right, work.bottom),
                    (mon.left, mon.top, mon.right, mon.bottom))
    except Exception:
        pass
    return None


def _win32_argtypes(user32, wndproc_type, set_long, get_long=None):
    """Pin the ctypes signatures for the user32 calls the hook
    makes. Without argtypes, ctypes passes Python ints as 32-bit
    C ints -- on 64-bit Windows that truncates the previous
    procedure's address when it is handed back to CallWindowProcW.
    Set once, guarded, best effort.
    (v0.6.30: pins the read-back call too -- a 64-bit procedure
    address must never be truncated to a 32-bit int on the way
    back, or the verification compares a chopped address.)"""
    global _resize_argtypes_set
    if _resize_argtypes_set:
        return
    try:
        import ctypes
        from ctypes import wintypes

        user32.CallWindowProcW.argtypes = [
            wndproc_type, wintypes.HWND, wintypes.UINT,
            wintypes.WPARAM, wintypes.LPARAM]
        user32.CallWindowProcW.restype = ctypes.c_ssize_t
        set_long.argtypes = [
            wintypes.HWND, ctypes.c_int, wndproc_type]
        set_long.restype = ctypes.c_void_p
        if get_long is not None:
            get_long.argtypes = [wintypes.HWND, ctypes.c_int]
            get_long.restype = ctypes.c_void_p
        user32.GetWindowRect.argtypes = [
            wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        user32.GetWindowRect.restype = wintypes.BOOL
        _resize_argtypes_set = True
    except Exception:
        pass


def _fill_maxinfo_from_work_area(hwnd, lparam):
    """Write the monitor work area into a MINMAXINFO struct (in
    place): maximised size = work area, maximised position =
    work-area offset from the monitor, max track size = the same.
    Best effort -- any failure leaves the struct untouched."""
    try:
        import ctypes
        from ctypes import wintypes

        class _MINMAXINFO(ctypes.Structure):
            _fields_ = [
                ("ptReserved", wintypes.POINT),
                ("ptMaxSize", wintypes.POINT),
                ("ptMaxPosition", wintypes.POINT),
                ("ptMinTrackSize", wintypes.POINT),
                ("ptMaxTrackSize", wintypes.POINT),
            ]

        class _MONITORINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD),
                        ("rcMonitor", wintypes.RECT),
                        ("rcWork", wintypes.RECT),
                        ("dwFlags", wintypes.DWORD)]

        user32 = ctypes.windll.user32
        mmi = ctypes.cast(lparam, ctypes.POINTER(_MINMAXINFO)).contents
        hmon = user32.MonitorFromWindow(hwnd, 2)
        if not hmon:
            return
        mi = _MONITORINFO()
        mi.cbSize = ctypes.sizeof(_MONITORINFO)
        if not user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
            return
        w = mi.rcWork
        m = mi.rcMonitor
        # v0.6.41: never hand Windows a degenerate work area (a
        # virtual/secondary monitor can report one): leave the
        # struct untouched instead of crashing the layout.
        if w.right - w.left <= 0 or w.bottom - w.top <= 0:
            return
        mmi.ptMaxPosition.x = w.left - m.left
        mmi.ptMaxPosition.y = w.top - m.top
        mmi.ptMaxSize.x = w.right - w.left
        mmi.ptMaxSize.y = w.bottom - w.top
        mmi.ptMaxTrackSize.x = mmi.ptMaxSize.x
        mmi.ptMaxTrackSize.y = mmi.ptMaxSize.y
        # v0.6.41: never let the minimum track size exceed the
        # maximum: WinForms throws ArgumentOutOfRange when its
        # MinimumSize is bigger than the maximised size.
        if mmi.ptMinTrackSize.x > mmi.ptMaxTrackSize.x:
            mmi.ptMinTrackSize.x = mmi.ptMaxTrackSize.x
        if mmi.ptMinTrackSize.y > mmi.ptMaxTrackSize.y:
            mmi.ptMinTrackSize.y = mmi.ptMaxTrackSize.y
    except Exception:
        pass


def _ensure_maximize_styles(hwnd):
    """Re-add WS_THICKFRAME / WS_MAXIMIZEBOX / WS_MINIMIZEBOX to
    the window style, once per hwnd. A FormBorderStyle=None
    window has neither WS_THICKFRAME nor WS_CAPTION, and without
    one of them Windows never sends WM_GETMINMAXINFO -- so the
    maximise handler below would never fire. The frame stays
    invisible because WM_NCCALCSIZE returns 0. Best effort."""
    if not hwnd or hwnd in _maximize_styles_done:
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32
        style = user32.GetWindowLongW(hwnd, _GWL_STYLE)
        want = style | _WS_THICKFRAME | _WS_MAXIMIZEBOX | _WS_MINIMIZEBOX
        if want != style:
            user32.SetWindowLongW(hwnd, _GWL_STYLE, want)
            user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0,
                                _SWP_NOMOVE | _SWP_NOSIZE
                                | _SWP_NOZORDER | _SWP_FRAMECHANGED)
        _maximize_styles_done.add(hwnd)
    except Exception:
        pass


def _resize_browser_child(hwnd, _win32=None):
    """Resize the largest visible child (the browser control) to
    fill the Form's client area. (v0.6.47)

    The window can fill the work area while the WebView2 control
    inside it stays at the old size -- maximise/restore moves the
    frame but nothing tells the child to follow, so the page
    looks like it "only moved to the top-left". This reads the
    client rect, finds the largest visible descendant (the
    browser) via the facade's enum_children, and SetWindowPos's
    it to (0, 0, client_w, client_h). Best effort: logs
    before/after, never raises. Returns True when a child was
    resized (or was already the right size), False when nothing
    could be done. _win32 is a test seam (a facade with
    enum_children); None means the real Win32."""
    if sys.platform != "win32" or not hwnd:
        return False
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        client = wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(client)):
            _applog_write("浏览器子窗口：读不到客户区。")
            return False
        cw = client.right - client.left
        ch = client.bottom - client.top
        if cw <= 0 or ch <= 0:
            _applog_write(f"浏览器子窗口：客户区无效 {cw}x{ch}。")
            return False
        if _win32 is None:
            _win32 = _real_win32_facade()
            if _win32 is None:
                return False
        kids = _win32.enum_children(hwnd)
        best = None
        best_area = -1
        for k in kids:
            if not k.get("visible") or not k.get("rect"):
                continue
            l, t, r, b = k["rect"]
            area = max(0, r - l) * max(0, b - t)
            if area > best_area:
                best_area = area
                best = k
        if best is None:
            _applog_write("浏览器子窗口：没找到可见子窗口。")
            return False
        bl, bt, br, bb = best["rect"]
        bw, bh = br - bl, bb - bt
        if bw == cw and bh == ch:
            return True
        ok = bool(user32.SetWindowPos(
            best["hwnd"], 0, 0, 0, cw, ch,
            _SWP_NOZORDER | _SWP_SHOWWINDOW))
        _applog_write(
            f"浏览器子窗口：{best['class']} {bw}x{bh} -> {cw}x{ch}，"
            f"SetWindowPos={ok}。")
        return bool(ok)
    except Exception as exc:
        _applog_write(f"浏览器子窗口调整出错：{exc}")
        return False


def _post_resize_browser_child(hwnd, _user32=None):
    """Queue the browser-child resize for the next message-loop
    turn (v0.6.57). Never raises; returns True when a resize was
    queued (or was already queued).

    The resize itself (SetWindowPos on the WebView2 control) must
    never run synchronously inside the parent window's WM_SIZE
    handler: SetWindowPos dispatches sizing messages to the child
    while Windows is still inside the parent's sizing callback,
    and a managed exception escaping the child's procedure there
    is what the OS reports as STATUS_FATAL_USER_CALLBACK_EXCEPTION
    (c000041d) -- the 2026-11-08 20:59 desktop crash on
    maximise->restore. Posting a private message moves the resize
    onto the normal message loop, outside any window-manager
    callback. Coalesced via _resize_child_pending: at most one
    deferred resize is ever queued. `_user32` is a test seam (a
    fake PostMessageW); None means the real user32. Argtypes are
    pinned here because without them ctypes passes the 64-bit
    hwnd as a 32-bit int."""
    global _resize_child_pending
    if not hwnd:
        return False
    if _resize_child_pending:
        return True
    try:
        import ctypes
        from ctypes import wintypes

        user32 = _user32 if _user32 is not None else ctypes.windll.user32
        try:
            user32.PostMessageW.argtypes = [
                wintypes.HWND, wintypes.UINT,
                wintypes.WPARAM, wintypes.LPARAM]
            user32.PostMessageW.restype = wintypes.BOOL
        except Exception:
            pass
        if user32.PostMessageW(
                hwnd, _WM_RESIZE_CHILD_DEFERRED, 0, 0):
            _resize_child_pending = True
            return True
        _applog_write("浏览器子窗口：延后调整投递失败。")
        return False
    except Exception:
        return False


def _run_deferred_resize_child(hwnd):
    """Run the deferred child resize posted by
    _post_resize_browser_child (v0.6.57). Called from the wndproc
    when the private message arrives, on the normal message loop
    -- never inside a window-manager callback. The pending flag
    is cleared BEFORE resizing, so a resize posted during the
    resize itself is not lost. Never raises (the resize itself
    never raises)."""
    global _resize_child_pending
    _resize_child_pending = False
    _resize_browser_child(hwnd)


# ----------------------------------------------------------------------
# Native drag-and-drop: a dropped video file lands here, not in JS.
# ----------------------------------------------------------------------
# WebView2 (plain Chromium) dropped File objects have NO .path --
# File.path is an Electron-only extension -- so the JS drop handler
# can never hand the backend a filesystem path. The drop is caught
# at the native window level instead: shell32 DragAcceptFiles on the
# Form HWND, and WM_DROPFILES in the wndproc below. The dropped path
# goes straight to the same select_video_path the file picker uses.
# This is deliberately small and local: it touches only the window
# setup (_polish_and_hook) and the existing wndproc chain; the
# resize hooks, the maximise logic and the v0.6.33 rule (the child
# hook stays OFF) are untouched.
_drop_bridge = None  # the live Bridge, set in Bridge.__init__


def _drop_files_to_paths(hdrop, _shell32=None):
    """Extract the file paths out of an HDROP (WM_DROPFILES handle).

    Returns a list of path strings (usually one). Never raises:
    a bad handle just yields an empty list. `_shell32` is a seam
    for the regression test (a fake DragQueryFileW/DragFinish).
    """
    try:
        import ctypes

        shell32 = _shell32
        if shell32 is None:
            shell32 = ctypes.windll.shell32
        count = shell32.DragQueryFileW(hdrop, 0xFFFFFFFF, None, 0)
        paths = []
        for index in range(count):
            buf = ctypes.create_unicode_buffer(32768)
            got = shell32.DragQueryFileW(hdrop, index, buf, 32768)
            if got:
                paths.append(buf.value)
        return paths
    except Exception:
        return []


def _handle_drop_files(hdrop, _shell32=None):
    """Handle one WM_DROPFILES: hand the first dropped file to the
    bridge's video picker, then free the handle. Returns True when a
    file was accepted, False otherwise. Never raises -- a failed
    drop must never break the window procedure."""
    import ctypes

    try:
        paths = _drop_files_to_paths(hdrop, _shell32=_shell32)
        if paths:
            bridge = _drop_bridge
            if bridge is not None:
                bridge.select_video_path(paths[0])
                _applog_write("拖放：%s" % paths[0])
                return True
        return False
    except Exception:
        return False
    finally:
        try:
            shell32 = _shell32 or ctypes.windll.shell32
            shell32.DragFinish(hdrop)
        except Exception:
            pass


def _install_drop_accept(window):
    """Arm native drag-and-drop on the main window (DragAcceptFiles).

    Called from _polish_and_hook (which re-runs on show/restore/
    maximise); DragAcceptFiles is idempotent, so re-arming is
    harmless, and the log line is written only the first time.
    Never raises: a window without drop support just keeps the
    file picker."""
    try:
        import ctypes

        hwnd = window.native.Handle
        if not hwnd:
            return
        ctypes.windll.shell32.DragAcceptFiles(hwnd, True)
        if not getattr(_install_drop_accept, "_logged", False):
            _install_drop_accept._logged = True
            _applog_write("拖放已开启：把视频拖进来就能选上。")
    except Exception:
        pass


# v0.6.48: WM_SIZE log throttle state -- last logged time/kind.
# Dragging a border fires WM_SIZE dozens of times a second; without
# this, app.log fills with one line per message.
_resize_size_log_at = 0.0
_resize_size_log_kind = None


def _resize_wndproc_inner(hwnd, msg, wparam, lparam):
    # v0.6.47: bind ctypes up front. The only import used to sit
    # inside the WM_NCHITTEST branch, which made ctypes a local
    # that stayed unbound for every other message -- so the chain
    # at the bottom raised UnboundLocalError and the message never
    # reached WinForms. That is why a taskbar restore (SC_RESTORE)
    # did nothing: the restore never arrived. Binding here makes
    # the chain work for all messages.
    import ctypes

    # v0.6.36: the maximise contract. WM_GETMINMAXINFO tells
    # Windows the maximised size/position (the work area) before
    # it maximises, so a real maximise lands on the work area
    # instead of the full monitor (taskbar covered). It is
    # filled in place and then chained on, so WinForms can still
    # apply MinimumSize (ptMinTrackSize) on top of our values.
    if msg == _WM_GETMINMAXINFO:
        _fill_maxinfo_from_work_area(hwnd, lparam)
    # WM_NCCALCSIZE: the whole window is client area (frameless).
    # Return 0 for wParam TRUE and FALSE alike.
    if msg == _WM_NCCALCSIZE:
        return 0
    if msg == _WM_NCHITTEST:
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            x = _signed_short(lparam & 0xFFFF)
            y = _signed_short((lparam >> 16) & 0xFFFF)
            rect = wintypes.RECT()
            if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                frames = _window_frame_rects(hwnd) or ()
                code = _hit_test_resize(
                    x, y,
                    (rect.left, rect.top, rect.right, rect.bottom),
                    frames=frames)
                if code != _HTCLIENT:
                    return code
        except Exception:
            pass
    # v0.6.47: log system commands and size changes, and keep
    # the browser child sized to the client area. The taskbar
    # restore (SC_RESTORE) and maximise/restore (WM_SIZE) arrive
    # here; if the child never follows, the window fills the work
    # area while the page stays at the old size. Size logging is
    # throttled (v0.6.48): at most one line a second while
    # dragging; kind changes always log.
    if msg == _WM_SYSCOMMAND:
        cmd = wparam & 0xFFF0
        name = {_SC_MINIMIZE: "最小化", _SC_RESTORE: "还原",
                _SC_MAXIMIZE: "最大化"}.get(cmd, f"系统命令 {cmd:#x}")
        _applog_write(f"窗口系统命令：{name}。")
    if msg == _WM_SIZE:
        kind = {_SIZE_RESTORED: "还原", _SIZE_MINIMIZED: "最小化",
                _SIZE_MAXIMIZED: "最大化"}.get(
                    wparam, f"尺寸变化 {wparam}")
        w = lparam & 0xFFFF
        h = (lparam >> 16) & 0xFFFF
        # v0.6.48: log at most one line a second while the size
        # keeps changing -- dragging a border fires WM_SIZE
        # dozens of times a second and filled app.log with one
        # line per message. A kind change (minimise / maximise /
        # restore) always logs, so a transition is never missed.
        # The deferred child resize below is NOT throttled: it
        # must be posted on every restored/maximised size message.
        global _resize_size_log_at, _resize_size_log_kind
        now = time.monotonic()
        if (kind != _resize_size_log_kind
                or now - _resize_size_log_at >= 1.0):
            _applog_write(f"窗口尺寸变化：{kind} {w}x{h}。")
            _resize_size_log_at = now
            _resize_size_log_kind = kind
        if wparam in (_SIZE_RESTORED, _SIZE_MAXIMIZED):
            # v0.6.57: only QUEUE the resize (PostMessage) -- never
            # SetWindowPos the child from inside this callback.
            _post_resize_browser_child(hwnd)
    if msg == _WM_DROPFILES:
        # A video file was dropped on the window: hand it to the
        # picker (the JS drop can never carry a path). Swallow the
        # message -- it is ours now, the chain must not see it.
        _handle_drop_files(wparam)
        return 0
    if msg == _WM_RESIZE_CHILD_DEFERRED:
        # v0.6.57: the deferred child resize, posted by the WM_SIZE
        # branch above -- runs on the normal message loop, outside
        # any window-manager callback. Ours only: swallow it.
        _run_deferred_resize_child(hwnd)
        return 0
    # every other message, and every hit-test inside the band's
    # inner edge, goes to the procedure that was there before --
    # pywebview's handling is untouched. If the chain was never
    # installed (only possible for a message racing the swap),
    # HTCLIENT is the safe answer: the cursor is over the page.
    orig = _resize_orig_proc
    if orig is None:
        return _HTCLIENT
    return ctypes.windll.user32.CallWindowProcW(
        orig, hwnd, msg, wparam, lparam)


def _resize_wndproc(hwnd, msg, wparam, lparam):
    """The window procedure installed by _install_resize_hook
    (v0.6.57: the whole body is now _resize_wndproc_inner behind
    this guard; the install still references this name).

    No exception may ever escape a window procedure: Windows calls
    it as a user-mode callback, and an exception escaping it is
    what the OS reports as STATUS_FATAL_USER_CALLBACK_EXCEPTION
    (c000041d) and kills the process for -- the 2026-11-08 20:59
    desktop crash on maximise->restore. If the inner handler raises
    for any reason, log it and hand the message to the original
    procedure; if even that fails, HTCLIENT (the cursor is over the
    page) is the safe answer."""
    try:
        return _resize_wndproc_inner(hwnd, msg, wparam, lparam)
    except Exception as exc:
        try:
            _applog_write(f"窗口过程出错，已交给原来处理：{exc}")
        except Exception:
            pass
        try:
            orig = _resize_orig_proc
            if orig is not None:
                import ctypes
                return ctypes.windll.user32.CallWindowProcW(
                    orig, hwnd, msg, wparam, lparam)
        except Exception:
            pass
        return _HTCLIENT


class _Win32Facade:
    """The Win32 calls the resize hook needs, behind one object
    so the bridge test can substitute a fake (this Linux VM has
    no Win32 at all)."""
    def __init__(self, user32, wndproc_type, set_long, get_long):
        self._user32 = user32
        self._wndproc_type = wndproc_type
        self._set_long = set_long
        self._get_long = get_long

    def make_callback(self, fn):
        return self._wndproc_type(fn)

    def wrap_proc(self, addr):
        """A callable WNDPROC for a raw procedure address (int) as
        returned by set_proc. CallWindowProcW's argtypes demand a
        WINFUNCTYPE instance -- passing the bare int raises
        TypeError on real Windows (v0.6.57 shipped exactly that:
        every message through the hook raised, the guard logged it
        and answered HTCLIENT, so no message ever reached WinForms).
        None stays None."""
        if not addr:
            return None
        return self._wndproc_type(addr)

    def set_proc(self, hwnd, callback):
        return self._set_long(hwnd, _GWLP_WNDPROC, callback)

    def get_proc(self, hwnd):
        value = self._get_long(hwnd, _GWLP_WNDPROC)
        return int(value) if value else None

    def get_window_rect(self, hwnd):
        import ctypes
        from ctypes import wintypes
        rect = wintypes.RECT()
        if self._user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return (rect.left, rect.top, rect.right, rect.bottom)
        return None

    def frame_rects(self, hwnd):
        return _window_frame_rects(hwnd)

    def call_proc(self, proc, hwnd, msg, wparam, lparam):
        if isinstance(proc, int):
            # a raw procedure address: wrap it, same reason as
            # wrap_proc (the dormant child-hook path has the
            # identical latent bug; the child hook itself stays
            # off per the v0.6.33 rule).
            proc = self._wndproc_type(proc)
        return self._user32.CallWindowProcW(
            proc, hwnd, msg, wparam, lparam)

    def enum_children(self, hwnd):
        """Every descendant window (BFS, depth <= 4): dicts with
        hwnd, class, rect (or None), visible. Best effort -- any
        failure returns what was collected so far. The callback
        is kept alive in a local until EnumChildWindows returns."""
        import ctypes
        from ctypes import wintypes
        user32 = self._user32
        out = []
        queue = [(hwnd, 0)]
        seen = set()
        while queue:
            parent, depth = queue.pop(0)
            if depth >= 4 or parent in seen:
                continue
            seen.add(parent)
            found = []

            def _collect(child, _lp, _found=found):
                try:
                    buf = ctypes.create_unicode_buffer(256)
                    user32.GetClassNameW(child, buf, 256)
                    cls = buf.value
                    rect = wintypes.RECT()
                    ok = bool(user32.GetWindowRect(
                        child, ctypes.byref(rect)))
                    vis = bool(user32.IsWindowVisible(child))
                    _found.append({
                        "hwnd": int(child), "class": cls,
                        "rect": (rect.left, rect.top, rect.right,
                                 rect.bottom) if ok else None,
                        "visible": vis})
                except Exception:
                    pass
                return True

            try:
                enum_cb = ctypes.WINFUNCTYPE(
                    ctypes.c_bool, wintypes.HWND,
                    wintypes.LPARAM)(_collect)
                user32.EnumChildWindows(
                    parent, ctypes.cast(enum_cb, ctypes.c_void_p), 0)
            except Exception:
                continue
            for ch in found:
                out.append(ch)
                queue.append((ch["hwnd"], depth + 1))
        return out


def _real_win32_facade():
    """The real facade from ctypes, or None (not Windows, or the
    calls are missing). Argtypes are pinned so 64-bit procedure
    addresses are never truncated."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        wndproc_type = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT,
            wintypes.WPARAM, wintypes.LPARAM)
        set_long = (getattr(user32, "SetWindowLongPtrW", None)
                    or getattr(user32, "SetWindowLongW", None))
        get_long = (getattr(user32, "GetWindowLongPtrW", None)
                    or getattr(user32, "GetWindowLongW", None))
        if set_long is None or get_long is None:
            return None
        _win32_argtypes(user32, wndproc_type, set_long, get_long)
        # v0.6.31: pin the calls the child enumeration needs.
        user32.MonitorFromWindow.argtypes = [
            wintypes.HWND, wintypes.DWORD]
        user32.MonitorFromWindow.restype = ctypes.c_void_p
        user32.GetMonitorInfoW.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p]
        user32.GetMonitorInfoW.restype = wintypes.BOOL
        user32.GetClassNameW.argtypes = [
            wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.GetClassNameW.restype = ctypes.c_int
        user32.IsWindowVisible.argtypes = [wintypes.HWND]
        user32.IsWindowVisible.restype = wintypes.BOOL
        user32.EnumChildWindows.argtypes = [
            wintypes.HWND, ctypes.c_void_p, wintypes.LPARAM]
        user32.EnumChildWindows.restype = wintypes.BOOL
        return _Win32Facade(user32, wndproc_type, set_long, get_long)
    except Exception:
        return None


def _callback_addr(cb):
    """The address of a WNDPROC callback, or None. The test fake
    carries .address; the real ctypes callback is cast."""
    if cb is None:
        return None
    try:
        return int(getattr(cb, "address"))
    except Exception:
        pass
    try:
        import ctypes
        value = ctypes.cast(cb, ctypes.c_void_p).value
        return int(value) if value else None
    except Exception:
        return None


def _resize_proc_is_ours(current, ours):
    """True when the procedure currently installed (current, an
    address read back with GetWindowLongPtr) is our callback
    (ours, the callback's address). None on either side is never
    ours -- a failed read-back must not count as installed."""
    if current is None or ours is None:
        return False
    try:
        return int(current) == int(ours)
    except Exception:
        return False


def _resize_needs_reinstall(current, ours, installed_hwnd, hwnd):
    """Pure decision: reinstall when the handle changed, or when
    the installed procedure is not ours. A handle of None never
    needs a reinstall -- there is no window to hook."""
    if hwnd is None:
        return False
    if installed_hwnd != hwnd:
        return True
    return not _resize_proc_is_ours(current, ours)


# the last resize-hook install attempt (for the bridge status)
_resize_last_attempt = {"ok": None, "hwnd": None, "reason": "",
                        "time": 0.0}


def _resize_log_attempt(hwnd, ok, reason):
    """Record one resize-hook install attempt (for the bridge
    status) and write it to the persistent log: success/failure,
    handle, reason. Every attempt is logged -- the attempts are
    infrequent (start, shown, restore, maximise, timers), so
    this cannot flood the file."""
    global _resize_last_attempt
    _resize_last_attempt = {"ok": bool(ok), "hwnd": hwnd,
                            "reason": reason, "time": time.time()}
    _applog_write(
        f"窗口改大小挂钩：{'已装上' if ok else '没装上'}"
        f"（句柄 {hwnd}，{reason}）")


def _current_wndproc(hwnd):
    """The window procedure currently installed on hwnd, or None
    off Windows / on any failure. Read-only; never installs."""
    if sys.platform != "win32" or not hwnd:
        return None
    try:
        facade = _real_win32_facade()
        if facade is None:
            return None
        return facade.get_proc(hwnd)
    except Exception:
        return None


def _install_resize_hook(window, _win32=None):
    """Subclass the native window procedure so WM_NCHITTEST answers
    resize hit codes on the border band. (v0.6.28: pywebview gives
    no resize handle for frameless windows, so the hook is the
    whole Chrome-like resize.) (v0.6.30: the install is VERIFIED
    -- after setting the procedure we read it back with
    GetWindowLongPtr and only count the hook as installed when
    the procedure really is our callback. The old "same handle,
    so already installed" guard lied whenever pywebview/WinForms
    swapped the procedure afterwards, and then the hook was never
    reinstalled. _win32 is a test seam: a facade with
    get_proc/set_proc/make_callback; None means the real Win32.)
    The installed callback is always kept referenced: dropping
    it while the window still points at it would crash on the
    next hit test, so a failed (re)install never clears the
    reference."""
    global _resize_orig_proc, _resize_hook_hwnd, _resize_callback
    if _win32 is None:
        _win32 = _real_win32_facade()
        if _win32 is None:
            return False
    try:
        hwnd = None
        native = getattr(window, "native", None)
        if native is not None:
            try:
                hwnd = int(native.Handle)
            except Exception:
                hwnd = None
        if not hwnd:
            try:
                import ctypes
                hwnd = ctypes.windll.user32.FindWindowW(
                    None, "ECHO 回声") or None
            except Exception:
                hwnd = None
        if not hwnd:
            _resize_log_attempt(None, False, "找不到窗口句柄")
            return False
        # 1. read back: is our callback still the procedure? A
        # handle recreated by WinForms, or a procedure swapped by
        # pywebview, lands here as "not ours" -- and gets a
        # reinstall instead of a false "already installed".
        try:
            current = _win32.get_proc(hwnd)
        except Exception:
            current = None
        if not _resize_needs_reinstall(
                current, _callback_addr(_resize_callback),
                _resize_hook_hwnd, hwnd):
            _resize_hook_hwnd = hwnd
            _resize_log_attempt(hwnd, True, "已装上（读回确认）")
            return True
        # 2. (re)install: subclass, chaining to whatever procedure
        # is there now (that is the one our callback must call).
        cb = _win32.make_callback(_resize_wndproc)
        prev = _win32.set_proc(hwnd, cb)
        if not prev:
            _resize_log_attempt(hwnd, False, "换过程失败")
            return False
        # 3. verify the read-back before claiming success.
        try:
            current = _win32.get_proc(hwnd)
        except Exception:
            current = None
        if not _resize_proc_is_ours(current, _callback_addr(cb)):
            _resize_callback = cb  # keep it referenced either way
            _resize_log_attempt(hwnd, False, "装完读回的过程不是我们的")
            return False
        _resize_callback = cb
        # v0.6.59: prev is a bare int address (set_long restype is
        # c_void_p). CallWindowProcW's argtypes demand a WINFUNCTYPE
        # instance -- handing it the int raises TypeError on real
        # Windows, which is what 0.6.57 logged on every message.
        # Wrap it once, here, so both chain sites get a callable.
        _resize_orig_proc = _win32.wrap_proc(prev)
        _resize_hook_hwnd = hwnd
        _resize_log_attempt(hwnd, True, "已装上")
        return True
    except Exception as exc:
        _resize_log_attempt(None, False, f"异常：{exc}")
        return False


# ================= CHROME-LIKE RESIZE, CHILD HOOK (v0.6.31) =================
# v0.6.30 proved the Form's hook IS installed (desktop app.log:
# handle 289019012, read-back confirmed four times) -- yet the
# border still cannot be dragged. The hook is on the wrong
# window: pywebview hosts a WebView2 child window that fills the
# whole frameless Form, including the 8px border band, so
# WM_NCHITTEST is sent to the window under the cursor -- the
# child -- and the Form's subclassed procedure never sees it.
# The child answers HTCLIENT and the border is dead.
#
# The fix subclasses the child window(s) too. The child's
# procedure answers the SAME border hit-test, computed against
# the TOP-LEVEL window's rect (the child fills the Form, so the
# border band lives on the child), and chains everything else to
# the child's original procedure. The Form's hook stays -- the
# two never fight: the child answers first for the border band
# (it is the window under the cursor), everything else chains
# through untouched on both.
#
# Which children? Never hard-coded: every descendant is logged
# (class, rect, visible) and only the visible ones whose rect
# covers the border band are subclassed. The WebView2 content
# window's class is "Chrome_WidgetWin_1" -- if it is among them,
# the log will say so; if the tree looks different on the real
# machine, the log shows what is actually there instead of
# guessing.
# ------------------------------------------------------------------

# one entry per subclassed child: {"orig": previous procedure
# address, "callback": the callback object (must stay referenced
# -- dropping it while the window still points at it would crash
# on the next hit test), "class": the window class, for the log
# and the bridge status}
_resize_child_hooks = {}

# the last child-hook install attempt (for the bridge status)
_resize_child_last_attempt = {"ok": None, "hwnd": None, "reason": "",
                              "time": 0.0}


def _rects_match(a, b, tol=2):
    """True when two (left, top, right, bottom) rects are the same
    within tol px -- the same tolerance _hit_test_resize uses for
    its frame match."""
    if a is None or b is None:
        return False
    return (abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol
            and abs(a[2] - b[2]) <= tol and abs(a[3] - b[3]) <= tol)


def _resize_window_state(rect, frames):
    """Windowed vs borderless-maximized vs fullscreen, from the
    rect alone (no settings needed): fills the whole monitor ->
    fullscreen; fills the work area -> borderless-maximized;
    otherwise windowed. For the log, so the parent can tell which
    mode the window was in when the hook was (not) installed."""
    if rect is None:
        return "？"
    if frames:
        if len(frames) > 1 and _rects_match(rect, frames[1]):
            return "全屏"
        if _rects_match(rect, frames[0]):
            return "无边框最大化"
    return "窗口化"


def _child_covers_border(child_rect, parent_rect, border=_RESIZE_BORDER):
    """Pure geometry: does the child window's rect cover any part
    of the parent rect's border band? The child is clipped to the
    parent first; a child inset by more than the band on every
    side covers nothing."""
    if child_rect is None or parent_rect is None:
        return False
    cl, ct, cr, cb = child_rect
    pl, pt, pr, pb = parent_rect
    left = max(cl, pl)
    top = max(ct, pt)
    right = min(cr, pr)
    bottom = min(cb, pb)
    if left >= right or top >= bottom:
        return False
    return (left < pl + border or right > pr - border
            or top < pt + border or bottom > pb - border)


def _resize_child_wndproc_factory(parent_hwnd, win32):
    """A WNDPROC for a WebView2 child window: answer the border
    hit-test against the TOP-LEVEL window's rect (the child fills
    the whole frameless Form, so the border band lives on the
    child, not the Form); everything else chains to the child's
    original procedure, looked up by the hwnd the message
    arrived on. The facade is captured at install time so the
    test can substitute a fake (this Linux VM has no Win32)."""
    def _child_wndproc(hwnd, msg, wparam, lparam):
        if msg == _WM_NCHITTEST:
            try:
                x = _signed_short(lparam & 0xFFFF)
                y = _signed_short((lparam >> 16) & 0xFFFF)
                rect = win32.get_window_rect(parent_hwnd)
                if rect is not None:
                    frames = win32.frame_rects(parent_hwnd) or ()
                    code = _hit_test_resize(x, y, rect, frames=frames)
                    if code != _HTCLIENT:
                        return code
            except Exception:
                pass
        entry = _resize_child_hooks.get(hwnd)
        orig = entry["orig"] if entry else None
        if orig is None:
            return _HTCLIENT
        return win32.call_proc(orig, hwnd, msg, wparam, lparam)
    return _child_wndproc


def _resize_child_log_attempt(hwnd, ok, reason):
    """Record one child-hook install attempt (for the bridge
    status) and write it to the persistent log: success/failure,
    handle, reason. Infrequent (same moments as the parent hook),
    so this cannot flood the file."""
    global _resize_child_last_attempt
    _resize_child_last_attempt = {"ok": bool(ok), "hwnd": hwnd,
                                  "reason": reason, "time": time.time()}
    _applog_write(
        f"窗口改大小子挂钩：{'已装上' if ok else '没装上'}"
        f"（句柄 {hwnd}，{reason}）")


def _install_resize_child_hooks(window, _win32=None):
    """Subclass every visible descendant window whose rect covers
    the top-level window's border band (v0.6.31: the WebView2
    child eats WM_NCHITTEST, so the Form's hook never sees it).
    Logs every descendant found (class, rect, visible), the
    top-level rect + state, and every install attempt. The Form's
    hook is untouched -- this only adds the child hook(s)."""
    global _resize_child_hooks
    if _win32 is None:
        _win32 = _real_win32_facade()
        if _win32 is None:
            return False
    try:
        hwnd = None
        native = getattr(window, "native", None)
        if native is not None:
            try:
                hwnd = int(native.Handle)
            except Exception:
                hwnd = None
        if not hwnd:
            try:
                import ctypes
                hwnd = ctypes.windll.user32.FindWindowW(
                    None, "ECHO 回声") or None
            except Exception:
                hwnd = None
        if not hwnd:
            _resize_child_log_attempt(None, False, "找不到窗口句柄")
            return False
        parent_rect = _win32.get_window_rect(hwnd)
        try:
            frames = _win32.frame_rects(hwnd) or ()
        except Exception:
            frames = ()
        _applog_write(
            f"窗口位置：句柄 {hwnd}，矩形 {parent_rect}，"
            f"状态 {_resize_window_state(parent_rect, frames)}")
        try:
            children = _win32.enum_children(hwnd)
        except Exception:
            children = []
        for ch in children:
            _applog_write(
                f"窗口子窗口：句柄 {ch.get('hwnd')}，类 {ch.get('class')}，"
                f"矩形 {ch.get('rect')}，可见 {'是' if ch.get('visible') else '否'}")
        covering = [ch for ch in children
                    if ch.get("visible")
                    and _child_covers_border(ch.get("rect"), parent_rect)]
        if not covering:
            _resize_child_log_attempt(None, False, "没有覆盖边框的子窗口")
            return False
        ok_all = True
        for ch in covering:
            child_hwnd = ch["hwnd"]
            entry = _resize_child_hooks.get(child_hwnd)
            try:
                current = _win32.get_proc(child_hwnd)
            except Exception:
                current = None
            ours = _callback_addr(entry["callback"]) if entry else None
            if entry is not None and _resize_proc_is_ours(current, ours):
                _resize_child_log_attempt(
                    child_hwnd, True,
                    f"已装上（读回确认，类 {ch.get('class')}）")
                continue
            child_fn = _resize_child_wndproc_factory(hwnd, _win32)
            cb = _win32.make_callback(child_fn)
            prev = _win32.set_proc(child_hwnd, cb)
            if not prev:
                _resize_child_log_attempt(
                    child_hwnd, False, f"换过程失败（类 {ch.get('class')}）")
                ok_all = False
                continue
            try:
                current = _win32.get_proc(child_hwnd)
            except Exception:
                current = None
            if not _resize_proc_is_ours(current, _callback_addr(cb)):
                # keep the callback referenced either way -- the
                # window may still point at it -- and try again
                # on the next attempt.
                _resize_child_hooks[child_hwnd] = {
                    "orig": prev, "callback": cb,
                    "class": ch.get("class", "")}
                _resize_child_log_attempt(
                    child_hwnd, False,
                    f"装完读回的过程不是我们的（类 {ch.get('class')}）")
                ok_all = False
                continue
            _resize_child_hooks[child_hwnd] = {
                "orig": prev, "callback": cb,
                "class": ch.get("class", "")}
            _resize_child_log_attempt(
                child_hwnd, True, f"已装上（类 {ch.get('class')}）")
        return ok_all
    except Exception as exc:
        _resize_child_log_attempt(None, False, f"异常：{exc}")
        return False


def _polish_and_hook(window):
    """Three native treatments, at every moment the handle may
    exist: the DWM polish, the Chrome-like resize hook and the
    native drag-and-drop arm, all on the top-level window.
    (v0.6.31 also hooked the WebView2
    child window(s); v0.6.33 turns that off -- it broke the
    JS bridge and left the shell in demo mode.)"""
    _polish_window(window)
    _install_resize_hook(window)
    # v0.6.54: arm native drag-and-drop (DragAcceptFiles on the
    # Form HWND). Idempotent, so re-arming on every polish is
    # harmless; the WM_DROPFILES handler lives in _resize_wndproc.
    _install_drop_accept(window)
    # v0.6.41: the maximise styles are NOT set here any more.
    # Setting them eagerly -- on show, on restore, on a timer --
    # rewrites the window style (SetWindowLongW + SWP_FRAMECHANGED)
    # while WinForms is still adding the browser control and doing
    # layout; that forced layout reads a half-built control's
    # window text and throws .NET ArgumentOutOfRangeException
    # (StringBuilder in Control.get_WindowText, Event 1026,
    # 2026-11-06 17:08 + hang 2026-11-07 09:37). The styles are
    # set lazily instead, only inside _maximize_work_area right
    # before ShowWindow(SW_MAXIMIZE) -- the one moment they are
    # needed, when the window is already stable. Maximise
    # behaviour is unchanged.
    # v0.6.33: the WebView2-child hook is OFF. v0.6.31 subclassed
    # the WebView2 child window (Chrome_WidgetWin_0) and the
    # pywebview JS bridge (window.pywebview.api) never appeared
    # again -- the shell has sat in demo mode since 23:08 on
    # 2026-11-05, the exact moment 0.6.31 landed. The Form's
    # own hook (0.6.30, proven harmless) stays; the child hook
    # stays defined but is never installed. Window drag/resize
    # goes back to the 0.6.30 behaviour (parked by the user).
# ================= END CHROME-LIKE RESIZE (v0.6.28) =================


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
    # easy_drag stays OFF: pywebview's easy drag arms a window move
    # on *every* mousedown anywhere in the page, which ate the
    # settings sliders alive (the window chased the cursor out from
    # under the thumb; only clicking could set a value). Dragging
    # lives where it belongs -- the .pywebview-drag-region titlebar
    # -- through pywebview's drag-region path alone.
    window = webview.create_window(
        "ECHO 回声", url=str(index),
        js_api=bridge, width=1040, height=836,
        min_size=(880, 620), frameless=True, easy_drag=False,
        background_color="#dbe2eb")
    bridge._window = window
    _polish_and_hook(window)
    try:
        window.events.shown += lambda: _polish_and_hook(window)
    except Exception:
        pass
    # WinForms can rebuild the native handle (or swap the window
    # procedure) long after start, which drops the subclassing;
    # re-attempt the hook on every show / restore / maximise, so
    # a lost hook comes back instead of staying dead. pywebview
    # exposes no "activated" event -- these are the moments a
    # handle can be recreated.
    for _ev in ("restored", "maximized"):
        try:
            _ev_obj = getattr(window.events, _ev)
            _ev_obj += (lambda: _polish_and_hook(window))
        except Exception:
            pass
    # The native handle often only exists a moment after start, so
    # polish again on a delay -- a single early attempt that found
    # no handle used to leave the frame unmodernised for the whole
    # session with no sign of why. (v0.6.30: the retries run
    # longer -- 10s and 30s -- because the handle can also appear
    # late, after the old 4s cutoff.)
    for delay in (1.5, 4.0, 10.0, 30.0):
        timer = threading.Timer(delay, _polish_and_hook, args=(window,))
        timer.daemon = True
        timer.start()
    webview.start()


if __name__ == "__main__":
    main()
