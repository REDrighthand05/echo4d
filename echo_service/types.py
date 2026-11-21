"""Data structures and schemas for echo_service."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class JobInfo:
    """Serializable metadata for a reconstructed job."""
    id: str
    name: str
    path: str
    mtime: float
    finished: bool
    scale_status: Optional[str] = None
    flags: Dict[str, bool] = field(default_factory=dict)


@dataclass
class ServiceStatus:
    """Live status of the echo service."""
    busy: bool = False
    busy_kind: Optional[str] = None  # "scene" | "depth" | None
    status_text: str = "就绪"
    frac: float = 0.0
    stage_idx: int = 0
    current_job: Optional[str] = None
    last_error: Optional[str] = None
