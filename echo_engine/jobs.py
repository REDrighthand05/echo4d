"""Job inspection, file discovery, and output resolution helpers."""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from echo_app.appstate import FileFlags


def play_file(path: Path) -> None:
    """Open a file with the system default player/viewer."""
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # noqa: S606 - opening user output file
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def scene_file(job: Path) -> Path:
    """Resolve the standalone scene page (scene.html or viewer.html)."""
    scene = job / "scene.html"
    if not scene.exists() and (job / "points.bin").exists():
        try:
            from echo_scene.pipeline import stage_present
            stage_present(job)
        except Exception:
            pass
    return scene


def replay_file(job: Path) -> Path:
    """Resolve the standalone route replay page (replay.html)."""
    replay = job / "replay.html"
    if not replay.exists() and (job / "points.bin").exists() and (job / "poses.json").exists():
        try:
            from echo_scene.pipeline import stage_replay
            stage_replay(job)
        except Exception:
            pass
    return replay


def replay_available(job: Path) -> bool:
    """Check if route replay can be shown or generated on demand."""
    return (job / "replay.html").exists() or (
        (job / "points.bin").exists() and (job / "poses.json").exists()
    )


def job_finished(job: Path) -> bool:
    """A job is finished once it holds a scene page."""
    return (job / "scene.html").exists() or (job / "viewer.html").exists()


def job_mtime(job: Path) -> float:
    """Sort timestamp for a job folder; returns -1.0 when missing."""
    try:
        return job.stat().st_mtime
    except OSError:
        return -1.0


def list_previous_jobs(jobs_root: Path) -> List[Path]:
    """List jobs under jobs_root, newest first, including in-progress ones."""
    if not jobs_root.exists():
        return []
    try:
        entries = list(jobs_root.iterdir())
    except OSError:
        return []

    jobs = [
        d for d in entries
        if d.is_dir() and (job_finished(d) or (d / "job.json").exists())
    ]
    jobs.sort(key=job_mtime, reverse=True)
    return jobs


def scale_status_line(job: Optional[Path]) -> Optional[str]:
    """Read metric scale status from scale.json."""
    if job is None:
        return None
    path = job / "scale.json"
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
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


def job_display_name(job: Path) -> str:
    """Build a human-readable display string with timestamp and status."""
    name = job.name
    if name.startswith("job_"):
        name = name[len("job_"):]
    try:
        stamp = datetime.fromtimestamp(job.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        stamp = ""
    suffix = "" if job_finished(job) else "（未完成）"
    line = scale_status_line(job)
    if line:
        suffix += f" · {line}"
    return f"{name}（{stamp}）{suffix}"


def job_file_flags(job: Optional[Path]) -> FileFlags:
    """Determine available action flags for a job folder."""
    if job is None:
        return FileFlags()
    scene = (job / "scene.html").exists() or (job / "viewer.html").exists() or (job / "points.bin").exists()
    return FileFlags(
        scene=scene,
        folder=job.exists(),
        preview=(job / "preview.html").exists(),
        replay=replay_available(job),
    )
