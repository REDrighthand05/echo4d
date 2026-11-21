"""Helper functions for bridge file dialogs, openers, and exporters."""

import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from echo_engine import (
    list_previous_jobs,
    play_file,
    replay_available,
    replay_file,
    scene_file,
)
from echo_service import (
    batch_export_to_dir,
    export_camera_trajectory,
    export_voxel_model,
)


def open_target(target: Optional[Path], what: str, log_fn: Callable[[str], None]) -> Dict[str, Any]:
    """Safely open an output target using system default viewer."""
    if target is not None and Path(target).exists():
        try:
            play_file(Path(target))
            return {"ok": True}
        except Exception as exc:
            log_fn(f"{what}打开失败：{exc}")
            return {"ok": False, "error": str(exc)}
    log_fn(f"{what}还没有，暂时打不开。")
    return {"ok": False, "error": "file not found"}


def open_folder(job: Optional[Path], log_fn: Callable[[str], None]) -> Dict[str, Any]:
    """Open job output folder in Windows File Explorer."""
    if job and job.exists():
        try:
            if sys.platform == "win32":
                os.startfile(str(job))
            return {"ok": True}
        except Exception as exc:
            log_fn(f"任务文件夹打开失败：{exc}")
            return {"ok": False, "error": str(exc)}
    return {"ok": False, "error": "folder not found"}


def dialog_pick_video(window) -> Optional[str]:
    """Display system open-file dialog to pick a video file."""
    if window is None:
        return None
    import webview
    paths = window.create_file_dialog(
        webview.OPEN_DIALOG,
        allow_multiple=False,
        file_types=("视频 (*.mp4;*.mov;*.mkv;*.avi)", "所有文件 (*.*)"),
    )
    return paths[0] if paths else None


def dialog_pick_jobs_root(window) -> Optional[Path]:
    """Display system folder dialog to change jobs root."""
    if window is None:
        return None
    import webview
    folders = window.create_file_dialog(webview.FOLDER_DIALOG)
    return Path(folders[0]) if folders else None


def dialog_export_scene(window, job: Optional[Path], fmt: str = "glb") -> Dict[str, Any]:
    """Prompt user for target folder and export solidified voxel model."""
    if not job or not window:
        return {"ok": False, "error": "window or job not ready"}
    import webview
    dest = window.create_file_dialog(webview.FOLDER_DIALOG)
    if dest:
        return export_voxel_model(job, Path(dest[0]), fmt=fmt)
    return {"ok": False, "cancelled": True}


def dialog_export_camera_route(window, job: Optional[Path]) -> Dict[str, Any]:
    """Prompt user for target folder and export camera route poses and model."""
    if not job or not window:
        return {"ok": False, "error": "window or job not ready"}
    import webview
    dest = window.create_file_dialog(webview.FOLDER_DIALOG)
    if dest:
        return export_camera_trajectory(job, Path(dest[0]))
    return {"ok": False, "cancelled": True}


def dialog_export_batch(window, jobs_root: Path, fmt: str = "glb") -> Dict[str, Any]:
    """Prompt user for target folder and export all past jobs."""
    if not window:
        return {"ok": False, "error": "window not ready"}
    import webview
    dest = window.create_file_dialog(webview.FOLDER_DIALOG)
    if dest:
        jobs = list_previous_jobs(jobs_root)
        return batch_export_to_dir(jobs, Path(dest[0]), fmt=fmt)
    return {"ok": False, "cancelled": True}


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


def appearance_to_js(appearance) -> Dict[str, Any]:
    """Serialize Appearance dataclass into JSON-compatible camelCase dict."""
    if not appearance:
        return {}
    return {camel: getattr(appearance, snake)
            for camel, snake in _APPEARANCE_KEYS
            if hasattr(appearance, snake)}


def appearance_from_js(appearance, data: Dict[str, Any]):
    """Update Appearance instance from camelCase JSON dict."""
    from echo_app.settings import coerce_appearance
    snake = {snake: data[camel] for camel, snake in _APPEARANCE_KEYS
             if isinstance(data, dict) and camel in data}
    return coerce_appearance(snake, base=appearance)


_applog_lock = threading.Lock()


def applog_write(line: str):
    """Append one timestamped line to %APPDATA%/ECHO/app.log with 1MB rotation."""
    try:
        from echo_app.settings import settings_dir

        path = settings_dir() / "app.log"
        import time

        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        text = f"{stamp} {line}\n"
        with _applog_lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and path.stat().st_size >= 1048576:
                    for i in range(2, 0, -1):
                        src = path if i == 1 else path.with_name(f"{path.name}.{i - 1}")
                        dst = path.with_name(f"{path.name}.{i}")
                        if src.exists():
                            if dst.exists():
                                dst.unlink()
                            src.rename(dst)
            except Exception:
                pass
            with open(path, "a", encoding="utf-8", errors="replace") as f:
                f.write(text)
    except Exception:
        pass


