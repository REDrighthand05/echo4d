"""3D Scene reconstruction pipeline orchestrator."""

import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from echo_engine.types import LogCallback, JobSinkCallback

PLAIN_STATUS = {
    "正在抽帧": "正在看画面",
    "正在挑清晰的帧": "正在挑清楚的画面",
    "正在算相机轨迹": "正在找回拍摄的路线",
    "正在算每帧的远近": "正在量每处的远近",
    "正在拼场景": "正在拼场景",
    "正在清噪": "正在把场景收拾干净",
    "正在定真尺度": "正在找参照物，量出真实大小",
    "正在生成网格": "正在生成表面",
    "正在生成像素模型": "正在把场景变成一个个小方块",
    "正在收尾": "马上就好",
    "正在生成过程回放": "正在做一段过程回放",
    "正在生成路线回放": "正在做一段路线回放",
}
STAGE_COUNT = len(PLAIN_STATUS)
STAGE_ORDER = list(PLAIN_STATUS.keys())


def run_pipeline(
    video: Path,
    jobs_root: Path,
    log: LogCallback,
    job_sink: Optional[JobSinkCallback] = None,
) -> Path:
    """Run every stage in order with memory collection and atomic error capture."""
    from echo_scene.pipeline import (
        stage_clean,
        stage_depth,
        stage_extract,
        stage_mesh,
        stage_poses,
        stage_present,
        stage_preview,
        stage_replay,
        stage_scale,
        stage_select,
        stage_voxel,
    )
    from echo_scene.fuse import stage_fuse

    job = jobs_root / f"job_{video.stem[:24]}"
    n = 1
    base = job
    while job.exists():
        n += 1
        job = jobs_root / f"{base.name}_{n}"
    job.mkdir(parents=True)

    if job_sink is not None:
        try:
            job_sink(job)
        except Exception:
            pass

    # Crash-recovery marker
    try:
        (job / "job.json").write_text(
            json.dumps(
                {
                    "stage": "started",
                    "ok": False,
                    "status": "running",
                    "video": str(video),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass

    # Phone-capture sensor record import
    try:
        from echo_scene import sensor_record as sr
        sr.import_capture_record(video.parent, job)
    except Exception:
        pass

    steps = [
        ("正在抽帧", lambda: stage_extract(video, job)),
        ("正在挑清晰的帧", lambda: stage_select(job)),
        ("正在算相机轨迹", lambda: stage_poses(job)),
        ("正在算每帧的远近", lambda: stage_depth(job)),
        ("正在拼场景", lambda: stage_fuse(job)),
        ("正在清噪", lambda: stage_clean(job)),
        ("正在定真尺度", lambda: stage_scale(job)),
        ("正在生成网格", lambda: stage_mesh(job)),
        ("正在生成像素模型", lambda: stage_voxel(job)),
        ("正在收尾", lambda: stage_present(job)),
        ("正在生成过程回放", lambda: stage_preview(job)),
        ("正在生成路线回放", lambda: stage_replay(job)),
    ]

    current_label = "初始化"
    try:
        for label, fn in steps:
            current_label = label
            log(label + "…")
            info = fn()
            if isinstance(info, dict) and info.get("ok") is False:
                raise RuntimeError(f"{label}失败：{info.get('reason', '未知原因')}")
            # Explicit GC between heavy operators
            gc.collect()

        log("完成。")
        return job

    except Exception as exc:
        try:
            (job / "job.json").write_text(
                json.dumps(
                    {
                        "stage": "failed",
                        "ok": False,
                        "status": "failed",
                        "failed_step": current_label,
                        "error": str(exc),
                        "video": str(video),
                        "failed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass
        raise
