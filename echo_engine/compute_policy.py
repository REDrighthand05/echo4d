"""Adaptive compute policy and hardware resource manager.

Provides transparent execution tier classification, GPU resource probing,
adaptive batch size determination, and resilient CPU fallback.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExecutionTier(str, Enum):
    """Execution tiers distinguishing real GPU execution from presence."""
    NVIDIA_DETECTED = "nvidia_detected"
    PROVIDER_AVAILABLE = "provider_available"
    SESSION_CREATED = "session_created"
    GPU_EXECUTED = "gpu_executed"
    CPU_FALLBACK = "cpu_fallback"


@dataclass
class ComputePolicy:
    """Resource management policy for heterogeneous inference."""
    preference: str = "auto"  # 'auto', 'cuda', 'directml', 'cpu'
    backends: List[str] = field(
        default_factory=lambda: ["cuda", "directml", "cpu"]
    )
    use_dynamic_batch: bool = True
    retain_ui_reserve: bool = True
    prevent_unbounded_allocation: bool = True
    fallback_on_unsupported: bool = True
    fallback_on_gpu_oom: bool = True
    retry_same_task_indefinitely: bool = False
    profile_timing: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComputePolicy":
        mem = data.get("memory", {})
        fb = data.get("fallback", {})
        return cls(
            preference=data.get("preference", "auto"),
            backends=data.get("backends", ["cuda", "directml", "cpu"]),
            use_dynamic_batch=mem.get("use_dynamic_batch", True),
            retain_ui_reserve=mem.get("retain_ui_reserve", True),
            prevent_unbounded_allocation=mem.get(
                "prevent_unbounded_allocation", True
            ),
            fallback_on_unsupported=fb.get("on_unsupported_operator", True),
            fallback_on_gpu_oom=fb.get("on_gpu_oom", True),
            retry_same_task_indefinitely=fb.get(
                "retry_same_task_indefinitely", False
            ),
            profile_timing=data.get("profiling", {}).get("stage_timing", True),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preference": self.preference,
            "backends": self.backends,
            "memory": {
                "use_dynamic_batch": self.use_dynamic_batch,
                "retain_ui_reserve": self.retain_ui_reserve,
                "prevent_unbounded_allocation": self.prevent_unbounded_allocation,
            },
            "fallback": {
                "on_unsupported_operator": self.fallback_on_unsupported,
                "on_gpu_oom": self.fallback_on_gpu_oom,
                "retry_same_task_indefinitely": self.retry_same_task_indefinitely,
            },
            "profiling": {"stage_timing": self.profile_timing},
        }


def probe_hardware() -> Dict[str, Any]:
    """Inspect local GPU, driver, CUDA UMD, and ONNX Runtime providers."""
    info: Dict[str, Any] = {
        "gpu_vendor": "unknown",
        "gpu_model": "unknown",
        "driver_version": "unknown",
        "cuda_version": "unknown",
        "vram_total_mb": 0,
        "vram_free_mb": 0,
        "ort_version": "unknown",
        "available_providers": [],
    }

    # Query nvidia-smi if available
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            cmd = [
                smi,
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ]
            res = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5, check=True
            )
            line = res.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                info["gpu_vendor"] = "NVIDIA"
                info["gpu_model"] = parts[0]
                info["driver_version"] = parts[1]
                info["vram_total_mb"] = int(float(parts[2]))
                info["vram_free_mb"] = int(float(parts[3]))
        except Exception:
            pass

    # Query onnxruntime
    try:
        import onnxruntime as ort
        info["ort_version"] = getattr(ort, "__version__", "unknown")
        info["available_providers"] = list(ort.get_available_providers())
    except ImportError:
        pass

    return info


def select_providers(
    policy: Optional[ComputePolicy] = None,
    forced_provider: Optional[str] = None,
) -> List[str]:
    """Resolve ordered execution providers based on policy and availability."""
    if policy is None:
        policy = ComputePolicy()

    try:
        import onnxruntime as ort
        available = list(ort.get_available_providers())
    except ImportError:
        return ["CPUExecutionProvider"]

    env_forced = forced_provider or os.environ.get("ECHO_DEPTH_PROVIDER")
    if env_forced and env_forced in available:
        return [env_forced]

    pref = (policy.preference or "auto").lower()
    if pref == "cpu":
        return ["CPUExecutionProvider"]

    candidates = []
    if pref in ("auto", "cuda") and "CUDAExecutionProvider" in available:
        candidates.append("CUDAExecutionProvider")
    if pref in ("auto", "directml") and "DmlExecutionProvider" in available:
        candidates.append("DmlExecutionProvider")

    # Add fallback to available GPU providers not matched by exact pref
    if pref == "auto":
        for p in ("CUDAExecutionProvider", "DmlExecutionProvider"):
            if p in available and p not in candidates:
                candidates.append(p)

    candidates.append("CPUExecutionProvider")
    return candidates


def calculate_batch_size(
    vram_free_mb: int,
    is_large_model: bool = False,
    user_override: Optional[int] = None,
) -> int:
    """Calculate safe dynamic batch size respecting VRAM budget and UI reserve."""
    if user_override is not None and user_override > 0:
        return min(8, user_override)

    env_val = os.environ.get("ECHO_DEPTH_BATCH_SIZE")
    if env_val:
        try:
            return max(1, min(8, int(env_val)))
        except ValueError:
            pass

    # If VRAM info unknown or very low, default to 1
    if vram_free_mb <= 0:
        return 1

    # Base model (389MB weights, ~1.2GB per frame forward activation)
    # Small model (99MB weights, ~400MB per frame forward activation)
    cost_per_frame = 1200 if is_large_model else 400
    ui_reserve_mb = 1024  # reserve 1GB for OS dwm + browser preview

    usable_vram = max(0, vram_free_mb - ui_reserve_mb)
    raw_batch = usable_vram // cost_per_frame
    return max(1, min(4 if is_large_model else 8, raw_batch))


@dataclass
class ExecutionReceipt:
    """Audit log verifying actual execution device and metrics."""
    tier: ExecutionTier
    actual_provider: str
    model_name: str
    batch_size: int
    frame_count: int
    warmup_seconds: float
    total_seconds: float
    avg_frame_ms: float
    peak_vram_mb: int = 0
    fallback_occurred: bool = False
    fallback_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tier"] = self.tier.value
        return d
