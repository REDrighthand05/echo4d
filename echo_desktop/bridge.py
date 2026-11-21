"""PyWebview JS Bridge exposing Python runtime APIs to the WebUI frontend."""

import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from echo_app.settings import load_settings, save_settings
from echo_app.version import APP_VERSION
from echo_desktop.bridge_helpers import (
    applog_write,
    appearance_from_js,
    appearance_to_js,
    dialog_export_batch,
    dialog_export_camera_route,
    dialog_export_scene,
    dialog_pick_jobs_root,
    dialog_pick_video,
    open_folder,
    open_target,
)
from echo_desktop.win32_drop import set_drop_callback
from echo_desktop.win32_hook import (
    close_window,
    maximize_work_area,
    minimize_window,
    resize_hook_status,
    toggle_maximize,
)
from echo_desktop.win32_wndproc import set_log_callback
from echo_engine import (
    job_display_name,
    job_file_flags,
    job_finished,
    job_mtime,
    list_previous_jobs,
    replay_available,
    replay_file,
    scale_status_line,
    scene_file,
)
from echo_service import JobManager, ThrottledCache

DEFAULT_JOBS_ROOT = Path.home() / "ECHO Jobs"


class Bridge:
    """JS-Python communication bridge exposed as window.pywebview.api."""

    def __init__(self, jobs_root: Optional[Path] = None, window=None):
        self.jobs_root = Path(jobs_root or DEFAULT_JOBS_ROOT)
        self._window = window
        self.settings = load_settings()

        self._all_log: List[str] = []
        self._pending_log: List[str] = []
        self._log_lock = threading.Lock()

        self.history_cache = ThrottledCache(ttl_seconds=2.5)
        self.inbox_cache = ThrottledCache(ttl_seconds=3.0)

        self.job_manager = JobManager(
            self.jobs_root,
            on_log=self._log,
            on_status=lambda s, f: None,
            on_job_started=lambda j: self.history_cache.invalidate(),
        )

        set_drop_callback(self.select_video_path)
        set_log_callback(self._log)

        self.update = None
        self._toolbox = None
        self._log(f"启动：ECHO v{APP_VERSION}，进程 {os.getpid()}。")
        self._log("ECHO 已就绪。")

    def _log(self, msg: str):
        with self._log_lock:
            line = str(msg)
            self._all_log.append(line)
            self._pending_log.append(line)
            if len(self._all_log) > 500:
                self._all_log.pop(0)
            applog_write(line)

    @property
    def busy(self) -> bool: return self.job_manager.busy
    @property
    def busy_kind(self) -> Optional[str]: return self.job_manager.busy_kind
    @property
    def status(self) -> str: return self.job_manager.status
    @property
    def frac(self) -> float: return self.job_manager.frac
    @property
    def job(self) -> Optional[Path]: return self.job_manager.current_job
    @property
    def depth_out(self) -> Optional[Path]: return self.job_manager.depth_out

    # --- Snapshot & Polling ---
    def get_snapshot(self) -> Dict[str, Any]:
        return {
            "version": APP_VERSION,
            "jobsRoot": str(self.jobs_root),
            "settings": {
                "autostart": self.settings.autostart,
                "autoUpdate": self.settings.auto_update,
                "updateProxy": self.settings.update_proxy,
                "appearance": appearance_to_js(self.settings.appearance),
                "windowMode": self.settings.window_mode,
                "textContrast": self.settings.text_contrast,
            },
            "history": self._get_history(),
            "inbox": self._get_inbox(),
            "status": self.status,
            "frac": self.frac,
            "busy": self.busy,
            "resizeHook": resize_hook_status(),
            "log": list(self._all_log),
            "cliEvents": [],
        }

    def poll(self) -> Dict[str, Any]:
        with self._log_lock:
            logs = self._pending_log[:]
            self._pending_log.clear()

        out = {
            "status": self.status,
            "frac": self.frac,
            "busy": self.busy,
            "busyKind": self.busy_kind,
            "hasJob": self.job is not None,
            "depthReady": self.depth_out is not None,
            "logs": logs,
            "previewSeq": self.job_manager._preview_seq,
            "update": self.update,
            "history": self._get_history(),
            "inbox": self._get_inbox(),
            "resizeHook": resize_hook_status(),
        }
        if self.depth_out is not None:
            out["depthUrl"] = Path(self.depth_out).as_uri()
        if self.job is not None:
            j = Path(self.job)
            out["jobOutputs"] = {
                "any": j.exists(),
                "scene": job_finished(j),
                "preview": (j / "preview.html").exists(),
                "replay": replay_available(j),
            }
        return out

    def _get_history(self) -> List[Dict[str, Any]]:
        def scan():
            return [{
                "id": j.name, "name": job_display_name(j), "path": str(j),
                "mtime": job_mtime(j), "finished": job_finished(j),
                "scaleStatus": scale_status_line(j), "flags": job_file_flags(j),
            } for j in list_previous_jobs(self.jobs_root)]
        return self.history_cache.get_or_scan(scan)

    def _get_inbox(self) -> List[Dict[str, Any]]:
        return self.inbox_cache.get_or_scan(lambda: [])

    def get_preview_frame(self) -> Dict[str, Any]:
        return {"seq": self.job_manager._preview_seq, "data": self.job_manager._preview_data}

    def get_toolbox(self) -> Dict[str, Any]:
        if self._toolbox is None:
            try:
                from echo_app.toolbox import build_toolbox
                self._toolbox = build_toolbox()
            except Exception as exc:
                return {"count": 0, "deployedCount": 0, "groups": [], "error": str(exc)}
        return self._toolbox

    # --- Video Selection & Jobs ---
    def pick_video(self):
        if self.busy:
            return None
        path = dialog_pick_video(self._window)
        return self.select_video_path(path) if path else None

    def select_video_path(self, path: str):
        ok, err = self.job_manager.select_video(Path(path))
        return {"ok": True, "name": Path(path).name} if ok else {"ok": False, "message": err}

    def select_inbox_capture(self, dir_path: str):
        return self.select_video_path(str(Path(dir_path) / "video.mp4"))

    def start_scene(self):
        ok, err = self.job_manager.start_scene()
        return {"ok": True} if ok else {"ok": False, "message": err}

    def start_depth(self):
        ok, err = self.job_manager.start_depth()
        return {"ok": True} if ok else {"ok": False, "message": err}

    # --- Job Outputs & Explorer ---
    def _job_for(self, job_dir=None) -> Optional[Path]:
        if job_dir:
            p = Path(str(job_dir))
            if p.exists():
                return p
        return self.job

    def select_job(self, job_dir):
        job = self._job_for(job_dir)
        if job is not None:
            self.job_manager.current_job = job
        return {"ok": job is not None}

    def open_scene(self, job_dir=None):
        job = self._job_for(job_dir)
        return open_target(scene_file(job) if job else None, "场景", self._log)

    def open_folder(self, job_dir=None):
        return open_folder(self._job_for(job_dir), self._log)

    def open_preview_replay(self, job_dir=None):
        job = self._job_for(job_dir)
        return open_target((job / "preview.html") if job else None, "过程回放", self._log)

    def open_route_replay(self, job_dir=None):
        job = self._job_for(job_dir)
        return open_target(replay_file(job) if job else None, "路线回放", self._log)

    def play_depth(self):
        return open_target(Path(self.depth_out) if self.depth_out else None, "深度视频", self._log)

    # --- Exports ---
    def export_scene(self, job_dir=None, fmt: str = "glb"):
        return dialog_export_scene(self._window, self._job_for(job_dir), fmt=fmt)

    def export_camera_route(self, job_dir=None):
        return dialog_export_camera_route(self._window, self._job_for(job_dir))

    def export_batch(self, fmt: str = "glb"):
        return dialog_export_batch(self._window, self.jobs_root, fmt=fmt)

    # --- Preview & Settings ---
    def preview_drag(self, dx: float, dy: float):
        return {"ok": True}

    def preview_zoom(self, factor: float):
        return {"ok": True}

    def preview_reset(self):
        return {"ok": True}

    def save_settings(self, data: Dict[str, Any]):
        try:
            for k, v in data.items():
                if k == "appearance" and isinstance(v, dict):
                    self.settings.appearance = appearance_from_js(self.settings.appearance, v)
                elif hasattr(self.settings, k):
                    setattr(self.settings, k, v)
            save_settings(self.settings)
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def pick_jobs_root(self):
        new_root = dialog_pick_jobs_root(self._window)
        if new_root:
            self.jobs_root = new_root
            self.job_manager.jobs_root = new_root
            self.history_cache.invalidate()
            return str(self.jobs_root)
        return None

    def check_update(self):
        return {"ok": True}

    def manual_check_update(self):
        return {"ok": True}

    def download_update(self):
        return {"ok": True}

    # --- Window Controls ---
    def win_minimize(self):
        ok = minimize_window(self._window)
        return {"ok": ok}

    def win_toggle_max(self):
        is_max = toggle_maximize(self._window)
        return {"ok": True, "maximized": is_max}

    def win_close(self):
        close_window(self._window)
        return {"ok": True}
