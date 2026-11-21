"""Background task queue and job lifecycle manager for ECHO 3D reconstruction."""

import base64
import gc
import threading
import traceback
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from echo_engine import (
    PLAIN_STATUS,
    STAGE_COUNT,
    STAGE_ORDER,
    run_depth_video,
    run_pipeline,
    scale_status_line,
)
from echo_service.validator import check_video


class TaskState(str, Enum):
    """Unified 10-state lifecycle machine for product tasks."""
    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PAUSE_REQUESTED = "PAUSE_REQUESTED"
    PAUSED = "PAUSED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_PNG_IEND = b"\x00\x00\x00\x00IEND\xaeB`\x82"


def is_png_complete(data: bytes) -> bool:
    """Verify PNG file integrity by checking header signature and IEND trailer."""
    return (
        len(data) >= len(_PNG_SIG) + len(_PNG_IEND)
        and data[:8] == _PNG_SIG
        and data[-12:] == _PNG_IEND
    )


class JobManager:
    """Manages active scene reconstruction and depth tasks with thread synchronization."""

    def __init__(
        self,
        jobs_root: Path,
        on_log: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str, float], None]] = None,
        on_job_started: Optional[Callable[[Path], None]] = None,
        on_preview_frame: Optional[Callable[[str, int], None]] = None,
    ):
        self.jobs_root = Path(jobs_root)
        self.on_log = on_log or (lambda msg: None)
        self.on_status = on_status or (lambda status, frac: None)
        self.on_job_started = on_job_started or (lambda job: None)
        self.on_preview_frame = on_preview_frame or (lambda data, seq: None)

        self._busy_lock = threading.Lock()
        self.busy: bool = False
        self.busy_kind: Optional[str] = None
        self.status: str = "就绪"
        self.frac: float = 0.0
        self._stage_idx: int = 0

        self.video: Optional[Path] = None
        self.current_job: Optional[Path] = None
        self.depth_out: Optional[Path] = None

        self._preview_seq: int = 0
        self._preview_data: Optional[str] = None
        self._depth_stop = threading.Event()
        self.task_state: TaskState = TaskState.CREATED
        self._cancel_event = threading.Event()
        self._pause_event = threading.Event()

    def select_video(self, path: Path) -> Tuple[bool, Optional[str]]:
        """Atomically validate and select a video file."""
        with self._busy_lock:
            if self.busy:
                return False, "正在忙，先等这一件事做完。"
            p = Path(path)
            err = check_video(p, self.jobs_root)
            if err:
                self.status = err
                return False, err
            self.video = p

        self.frac = 0.0
        self.status = "视频选好了。点「开始生成」，或者做深度视频。"
        self._log(f"已选视频：{p.name}")
        return True, None

    def start_scene(self) -> Tuple[bool, Optional[str]]:
        """Start async 3D reconstruction pipeline under busy lock."""
        with self._busy_lock:
            if self.busy or self.video is None:
                return False, "忙碌中或未选择视频"
            err = check_video(self.video, self.jobs_root)
            if err:
                self.status = err
                self.task_state = TaskState.FAILED
                return False, err
            self.busy, self.busy_kind = True, "scene"
            self.task_state = TaskState.RUNNING
            self._cancel_event.clear()
            self._pause_event.clear()

        self._stage_idx = 0
        self.frac = 0.0
        self._set_status("开始了。", 0.0)
        threading.Thread(target=self._scene_worker, daemon=True).start()
        return True, None

    def request_cancel(self) -> Tuple[bool, str]:
        """Request cooperative cancellation at nearest safe pipeline stage."""
        with self._busy_lock:
            if not self.busy:
                return False, "当前没有正在执行的任务"
            self.task_state = TaskState.CANCEL_REQUESTED
            self._cancel_event.set()
            self._depth_stop.set()
            msg = "取消请求已接收，正在等待安全检查点终止…"
            self._set_status(msg, self.frac)
            return True, msg

    def request_pause(self) -> Tuple[bool, str]:
        """Request cooperative pause at nearest stage checkpoint."""
        with self._busy_lock:
            if not self.busy:
                return False, "当前没有正在执行的任务"
            self.task_state = TaskState.PAUSE_REQUESTED
            self._pause_event.set()
            msg = "暂停请求已接收，正在等待当前阶段完成…"
            self._set_status(msg, self.frac)
            return True, msg

    def _pipeline_log(self, line: str):
        if self._cancel_event.is_set():
            self.task_state = TaskState.CANCEL_REQUESTED
            raise InterruptedError("Task cancelled by cooperative request")
        text = str(line)
        key = text.rstrip("…").strip()
        if key in PLAIN_STATUS:
            if key in STAGE_ORDER:
                self._stage_idx = max(self._stage_idx, STAGE_ORDER.index(key) + 1)
            self._set_status(PLAIN_STATUS[key] + "…", self._stage_idx / STAGE_COUNT)
        self._log(text)

    def _scene_worker(self):
        try:
            job = run_pipeline(
                self.video,
                self.jobs_root,
                self._pipeline_log,
                job_sink=self._handle_job_started,
            )
            self.current_job = job
            self.task_state = TaskState.COMPLETED
            self._set_status("场景做好了。点「打开场景」就能看。", 1.0)
            scale = scale_status_line(job)
            if scale:
                self._log(scale)
            self._log("场景生成完成。")
        except InterruptedError:
            self.task_state = TaskState.CANCELLED
            self._set_status("任务已安全取消。", self.frac)
            self._log("任务已按用户请求安全取消。")
        except Exception:
            self.task_state = TaskState.FAILED
            self._set_status("生成失败，看一下日志。", self.frac)
            self._log(traceback.format_exc(limit=2))
        finally:
            with self._busy_lock:
                self.busy, self.busy_kind = False, None
            gc.collect()

    def start_depth(self) -> Tuple[bool, Optional[str]]:
        """Start async depth estimation under busy lock."""
        with self._busy_lock:
            if self.busy or self.video is None:
                return False, "忙碌中或未选择视频"
            err = check_video(self.video, self.jobs_root)
            if err:
                self.status = err
                return False, err
            self.busy, self.busy_kind = True, "depth"

        self.depth_out = None
        self.frac = 0.0
        self._set_status("正在生成深度视频…", 0.0)

        out = self.video.with_name(f"{self.video.stem}_depth.mp4")
        base, n = out, 1
        while out.exists():
            n += 1
            out = base.with_name(f"{base.stem}_{n}.mp4")

        self._depth_stop = threading.Event()
        stills_dir = out.parent / (out.stem + "_depth")
        threading.Thread(
            target=self._depth_stills_loop,
            args=(stills_dir, self._depth_stop),
            daemon=True,
        ).start()
        threading.Thread(target=self._depth_worker, daemon=True).start()
        return True, None

    def _depth_stills_loop(self, stills_dir: Path, stop: threading.Event):
        seen = None
        while not stop.is_set():
            try:
                stills = sorted(stills_dir.glob("d_*.png"))
                if stills:
                    newest = stills[-1]
                    size = newest.stat().st_size
                    if size and (newest.name, size) != seen:
                        data = newest.read_bytes()
                        if is_png_complete(data):
                            seen = (newest.name, size)
                            data_url = (
                                "data:image/png;base64,"
                                + base64.b64encode(data).decode()
                            )
                            self._preview_data = data_url
                            self._preview_seq += 1
                            self.on_preview_frame(data_url, self._preview_seq)
            except Exception:
                pass
            stop.wait(0.8)

    def _depth_worker(self):
        def _progress(stage, done, total):
            if total:
                self.frac = (done / total) if stage != "frames" else 0.02
            if stage == "depth" and total:
                self._set_status(f"正在生成深度视频… {done}/{total}", self.frac)
            elif stage == "encode":
                self._set_status("正在合成深度视频…", self.frac)

        try:
            self.depth_out = run_depth_video(self.video, progress=_progress)
            self._set_status("深度视频做好了。", 1.0)
            self._log("深度视频完成。")
        except Exception:
            self._set_status("深度视频生成失败，看一下日志。", self.frac)
            self._log(traceback.format_exc(limit=2))
        finally:
            self._depth_stop.set()
            with self._busy_lock:
                self.busy, self.busy_kind = False, None
            gc.collect()

    def _handle_job_started(self, job: Path):
        self.current_job = job
        try:
            self.on_job_started(job)
        except Exception:
            pass

    def _set_status(self, msg: str, frac: float):
        self.status = msg
        self.frac = frac
        try:
            self.on_status(msg, frac)
        except Exception:
            pass

    def _log(self, msg: str):
        try:
            self.on_log(msg)
        except Exception:
            pass

    def get_state(self) -> Dict[str, Any]:
        """Return snapshot of job manager state."""
        return {
            "busy": self.busy,
            "busyKind": self.busy_kind,
            "status": self.status,
            "frac": self.frac,
            "video": str(self.video) if self.video else None,
            "currentJob": str(self.current_job) if self.current_job else None,
            "depthOut": str(self.depth_out) if self.depth_out else None,
            "taskState": self.task_state.value,
            "previewSeq": self._preview_seq,
        }
