"""Type annotations and data structures for echo_engine."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Dict, Any

LogCallback = Callable[[str], None]
JobSinkCallback = Callable[[Path], None]
ProgressCallback = Callable[[float, str], None]


@dataclass
class JobOutput:
    """Represents the output artifacts of a finished or running job."""
    job_dir: Path
    scene_file: Optional[Path] = None
    replay_file: Optional[Path] = None
    preview_file: Optional[Path] = None
    points_file: Optional[Path] = None
    is_finished: bool = False
    scale_status: Optional[str] = None
