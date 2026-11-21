"""Action handlers and worker dispatch for Tkinter desktop shell."""

import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from echo_engine import (
    job_display_name,
    job_finished,
    job_mtime,
    list_previous_jobs,
    play_file,
    replay_file,
    run_depth_video,
    run_pipeline,
    scene_file,
)
from echo_service import check_video


class TkActionHandler:
    """Dispatches asynchronous tasks and UI actions for Tkinter shell."""

    def __init__(self, state: Dict[str, Any], ui_queue, jobs_root: Path, log_fn: Callable[[str], None]):
        self.state = state
        self.queue = ui_queue
        self.jobs_root = Path(jobs_root)
        self.log_fn = log_fn

    def select_video(self, path: Path) -> bool:
        err = check_video(path, self.jobs_root)
        if err:
            self.log_fn(err)
            return False
        self.state["video"] = path
        self.log_fn(f"已选视频：{path.name}")
        return True

    def start_pipeline(self):
        video = self.state.get("video")
        if not video or self.state.get("busy"):
            return
        self.state["busy"] = True
        self.state["busy_kind"] = "scene"
        threading.Thread(target=self._scene_worker, args=(video,), daemon=True).start()

    def _scene_worker(self, video: Path):
        def _log_cb(line):
            self.queue.put(("log", line))

        def _sink_cb(job):
            self.queue.put(("job_started", job))

        try:
            job = run_pipeline(video, self.jobs_root, _log_cb, job_sink=_sink_cb)
            self.queue.put(("done", job))
        except Exception as exc:
            self.queue.put(("error", str(exc)))

    def start_depth(self):
        video = self.state.get("video")
        if not video or self.state.get("busy"):
            return
        self.state["busy"] = True
        self.state["busy_kind"] = "depth"
        threading.Thread(target=self._depth_worker, args=(video,), daemon=True).start()

    def _depth_worker(self, video: Path):
        def _prog(stage, done, total):
            self.queue.put(("depth_progress", (stage, done, total)))

        try:
            out = run_depth_video(video, progress=_prog)
            self.queue.put(("depth_done", out))
        except Exception as exc:
            self.queue.put(("depth_error", str(exc)))

    def open_scene(self):
        job = self.state.get("job")
        if job:
            play_file(scene_file(job))

    def open_folder(self):
        job = self.state.get("job")
        if job and job.exists():
            if sys.platform == "win32":
                os.startfile(str(job))

    def open_preview(self):
        job = self.state.get("job")
        if job and (job / "preview.html").exists():
            play_file(job / "preview.html")

    def open_replay(self):
        job = self.state.get("job")
        if job:
            play_file(replay_file(job))

    def play_depth(self):
        depth_out = self.state.get("depth_video")
        if depth_out and Path(depth_out).exists():
            play_file(Path(depth_out))
