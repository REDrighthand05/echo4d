"""Checkpoint management, state serialization, and validation for safe pause/resume."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CheckpointData:
    """Atomic checkpoint data contract for safe resumption."""
    schema_version: int = 1
    job_id: str = ""
    input_sha256: str = ""
    pipeline_version: str = "1.0.0"
    completed_stages: List[str] = field(default_factory=list)
    active_stage: str = ""
    stage_parameters_sha256: str = ""
    model_versions: Dict[str, str] = field(default_factory=dict)
    output_artifacts: List[str] = field(default_factory=list)
    validated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointData":
        return cls(**data)


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 of a file efficiently in 1MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class CheckpointManager:
    """Validates and persists job checkpoint data."""

    @staticmethod
    def save_checkpoint(
        job_dir: Path,
        checkpoint: CheckpointData,
    ) -> Path:
        job_dir = Path(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = job_dir / "checkpoint.json"
        temp_path = job_dir / "checkpoint.json.tmp"

        temp_path.write_text(
            json.dumps(checkpoint.to_dict(), indent=2), encoding="utf-8"
        )
        temp_path.replace(ckpt_path)
        return ckpt_path

    @staticmethod
    def load_and_validate(
        job_dir: Path,
        expected_input_path: Optional[Path] = None,
    ) -> Optional[CheckpointData]:
        job_dir = Path(job_dir)
        ckpt_path = job_dir / "checkpoint.json"
        if not ckpt_path.exists():
            return None

        try:
            data = json.loads(ckpt_path.read_text(encoding="utf-8"))
            ckpt = CheckpointData.from_dict(data)
        except Exception:
            return None

        # If expected input video provided, verify SHA-256
        if expected_input_path and expected_input_path.exists():
            curr_sha = compute_file_sha256(expected_input_path)
            if curr_sha != ckpt.input_sha256:
                return None

        # Verify all output artifacts exist on disk
        for artifact in ckpt.output_artifacts:
            art_p = job_dir / artifact
            if not art_p.exists() or art_p.stat().st_size == 0:
                ckpt.validated = False
                return None

        ckpt.validated = True
        return ckpt

    @staticmethod
    def invalidate(job_dir: Path, reason: str = "parameter_mismatch") -> None:
        job_dir = Path(job_dir)
        ckpt_path = job_dir / "checkpoint.json"
        if ckpt_path.exists():
            try:
                data = json.loads(ckpt_path.read_text(encoding="utf-8"))
                data["validated"] = False
                data["invalidation_reason"] = reason
                ckpt_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception:
                pass
