"""Execution handlers for echo_cli subcommands."""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from echo_engine import (
    job_display_name,
    job_file_flags,
    job_finished,
    job_mtime,
    list_previous_jobs,
    run_depth_video,
    run_pipeline,
    scale_status_line,
)
from echo_service import (
    EchoServer,
    JobManager,
    check_video,
    export_camera_trajectory,
    export_voxel_model,
)

DEFAULT_JOBS_ROOT = Path.home() / "ECHO Jobs"


def _resolve_jobs_root(custom: Optional[str]) -> Path:
    if custom:
        p = Path(custom)
        p.mkdir(parents=True, exist_ok=True)
        return p
    DEFAULT_JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    return DEFAULT_JOBS_ROOT


def handle_run(args: argparse.Namespace) -> int:
    video_path = Path(args.video).resolve()
    jobs_root = _resolve_jobs_root(args.jobs_root)

    print(f"[ECHO CLI] Validating input video: {video_path}")
    err = check_video(video_path, jobs_root)
    if err:
        print(f"[ERROR] Validation failed: {err}", file=sys.stderr)
        return 1

    print(f"[ECHO CLI] Target jobs directory: {jobs_root}")
    start_time = time.time()

    def log_cb(line: str):
        print(f"  -> {line}")

    def sink_cb(job: Path):
        print(f"[ECHO CLI] Job allocated: {job.name}")

    try:
        job = run_pipeline(video_path, jobs_root, log_cb, job_sink=sink_cb)
        elapsed = time.time() - start_time
        print(f"[ECHO CLI] Pipeline completed successfully in {elapsed:.1f}s.")
        print(f"[ECHO CLI] Result directory: {job}")

        scale = scale_status_line(job)
        if scale:
            print(f"[ECHO CLI] Scale status: {scale}")

        if args.export:
            dest = Path(args.dest or (job / "export")).resolve()
            print(f"[ECHO CLI] Exporting GLB model to: {dest}")
            res = export_voxel_model(job, dest, fmt="glb")
            if res.get("ok"):
                print(f"[ECHO CLI] Export succeeded: {res.get('files')}")
            else:
                print(f"[WARN] Export failed: {res.get('reason')}")

        return 0
    except Exception as exc:
        print(f"[ERROR] Reconstruction pipeline failed: {exc}", file=sys.stderr)
        return 2


def handle_depth(args: argparse.Namespace) -> int:
    video_path = Path(args.video).resolve()
    err = check_video(video_path)
    if err:
        print(f"[ERROR] Validation failed: {err}", file=sys.stderr)
        return 1

    print(f"[ECHO CLI] Starting depth estimation on: {video_path.name}")
    start_time = time.time()

    def progress_cb(stage: str, done: int, total: int):
        pct = (done / total * 100) if total > 0 else 0
        sys.stdout.write(f"\r  [{stage.upper()}] {done}/{total} ({pct:.1f}%)")
        sys.stdout.flush()

    try:
        out = run_depth_video(video_path, progress=progress_cb)
        print()
        elapsed = time.time() - start_time
        print(f"[ECHO CLI] Depth video completed in {elapsed:.1f}s: {out}")
        return 0
    except Exception as exc:
        print(f"\n[ERROR] Depth estimation failed: {exc}", file=sys.stderr)
        return 2


def handle_serve(args: argparse.Namespace) -> int:
    jobs_root = _resolve_jobs_root(args.jobs_root)
    host, port = args.host, args.port

    job_mgr = JobManager(
        jobs_root,
        on_log=lambda m: print(f"[SERVICE LOG] {m}"),
        on_status=lambda s, f: print(f"[STATUS {int(f*100)}%] {s}"),
    )
    server = EchoServer((host, port), job_mgr, jobs_root)
    print(f"[ECHO REST API] Listening at http://{host}:{port}")
    print(f"[ECHO REST API] Jobs root: {jobs_root}")
    print("[ECHO REST API] Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ECHO REST API] Stopping server...")
    finally:
        server.stop()
    return 0


def handle_export(args: argparse.Namespace) -> int:
    job = Path(args.job).resolve()
    dest = Path(args.dest).resolve()
    if not job.exists():
        print(f"[ERROR] Job folder not found: {job}", file=sys.stderr)
        return 1

    print(f"[ECHO CLI] Exporting job {job.name} (format: {args.format}) to {dest}...")
    if args.format == "route":
        res = export_camera_trajectory(job, dest)
    else:
        res = export_voxel_model(job, dest, fmt=args.format)

    if res.get("ok"):
        print(f"[ECHO CLI] Export completed successfully: {res.get('files')}")
        return 0
    else:
        print(f"[ERROR] Export failed: {res.get('reason')}", file=sys.stderr)
        return 2


def handle_status(args: argparse.Namespace) -> int:
    jobs_root = _resolve_jobs_root(args.jobs_root)
    jobs = list_previous_jobs(jobs_root)
    limit = args.limit
    recent = jobs[:limit]

    print(f"[ECHO CLI] Found {len(jobs)} past jobs in {jobs_root} (showing {len(recent)}):")
    print("-" * 75)
    print(f"{'Job Name':<30} | {'Status':<10} | {'Modified':<18} | {'Flags'}")
    print("-" * 75)

    for j in recent:
        name = job_display_name(j)[:30]
        fin = "Finished" if job_finished(j) else "Pending/Failed"
        mtime_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(job_mtime(j)))
        flags = job_file_flags(j)
        flags_str = ",".join(k for k, v in flags.items() if v) or "none"
        print(f"{name:<30} | {fin:<10} | {mtime_str:<18} | {flags_str}")

    print("-" * 75)
    return 0
