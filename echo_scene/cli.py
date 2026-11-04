#!/usr/bin/env python3
"""EchoScene command line (user directive, 2026-11-03): the desktop
software must be drivable without its window -- import, run, export,
all from the shell, so Muse (or a script) can operate it directly.

    python -m echo_scene.cli import <video|capture_dir> [--jobs-root D]
    python -m echo_scene.cli run <job>
    python -m echo_scene.cli depth-video <video> [--out P] [--fps N]
    python -m echo_scene.cli export <job> --format F [--out P]
    python -m echo_scene.cli status [<job>]

A <job> is a job folder path or a bare name under the jobs root
(default: the same jobs root the desktop app uses, from the saved
settings). Every subcommand reuses the existing pipeline stages and
modules -- this file is only argument parsing, sequencing and
reporting. Progress goes to stdout, failures to stderr, and the exit
code carries the verdict: 0 ok, 1 failed, 2 usage error (argparse).
Each finished command also leaves one activity-event line for the
desktop frontend's silent toast strip (echo_app.cli_events) --
best-effort only; a failed event write changes nothing here.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v",
                  ".mpg", ".mpeg"}

# Existing finished-job artifacts `export` can hand out as they are
# (name -> file inside the job folder).
KNOWN_ARTIFACTS = {
    "scene": "scene.html",
    "points-ply": "scene_points.ply",
    "mesh-ply": "mesh.ply",
    "voxel-model-glb": "voxel_model.glb",
    "voxel-model-ply": "voxel_model.ply",
}
VOXEL_FORMATS = {"voxel": ("glb", "ply"), "voxel-glb": ("glb",),
                 "voxel-ply": ("ply",)}

# The one output name a depth-video / export command produced, set
# by the command itself just before it reports success, so the
# activity event (below) can name the actual file. Purely cosmetic
# -- nothing branches on it.
_EVENT_DETAIL = {"out": ""}


def default_jobs_root() -> Path:
    """The desktop app's jobs root (saved settings, else ~/ECHO Jobs)."""
    try:
        from echo_app.settings import load_settings

        return load_settings().jobs_root_path()
    except Exception:
        return Path.home() / "ECHO Jobs"


def resolve_job(jobs_root: Path, name_or_path: str) -> Path:
    """A job argument is either a path to a job folder or a bare name
    under the jobs root. Raises FileNotFoundError when neither fits."""
    cand = Path(name_or_path)
    if cand.is_dir():
        return cand.resolve()
    under = Path(jobs_root) / name_or_path
    if under.is_dir():
        return under.resolve()
    raise FileNotFoundError(f"job not found: {name_or_path}")


def _read_job_json(job: Path) -> dict:
    try:
        data = json.loads((job / "job.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _new_job_dir(jobs_root: Path, stem: str) -> Path:
    """Same naming the desktop app uses: job_<stem>, then _2, _3..."""
    base = jobs_root / f"job_{stem[:24]}"
    job, n = base, 1
    while job.exists():
        n += 1
        job = jobs_root / f"{base.name}_{n}"
    job.mkdir(parents=True)
    return job


# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------

def cmd_import(args) -> int:
    src = Path(args.source).expanduser()
    if not src.exists():
        print(f"import: no such file or folder: {src}", file=sys.stderr)
        return 1
    src = src.resolve()
    if src.is_dir():
        video = src / "video.mp4"
        if not video.exists():
            print(f"import: capture folder has no video.mp4: {src}",
                  file=sys.stderr)
            return 1
        kind, stem = "capture", src.name
    elif src.suffix.lower() in VIDEO_SUFFIXES:
        video, kind, stem = src, "video", src.stem
    else:
        print(f"import: not a video or capture folder: {src}",
              file=sys.stderr)
        return 1
    jobs_root = Path(args.jobs_root)
    jobs_root.mkdir(parents=True, exist_ok=True)
    job = _new_job_dir(jobs_root, stem)
    record = {"stage": "imported", "ok": True, "status": "imported",
              "kind": kind, "source": str(src), "video": str(video)}
    if kind == "capture":
        # Land the capture's sensor record inside the job (shared
        # with the desktop import -- see sensor_record.py): the fuse
        # stage orients by IMU gravity when the record is present.
        from echo_scene import sensor_record as sr
        landed = sr.import_capture_record(src, job)
        if landed is not None:
            record.update(landed)
    (job / "job.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2))
    print(f"job: {job}")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


# -------------------------------------------------------------------
# run
# -------------------------------------------------------------------

def _run_steps(job: Path):
    """The desktop app's stage sequence (app.run_pipeline), driven
    headless: same stages, same order, same pipeline functions."""
    from echo_scene.fuse import stage_fuse
    from echo_scene.pipeline import (stage_clean, stage_depth,
                                     stage_extract, stage_mesh,
                                     stage_poses, stage_present,
                                     stage_preview, stage_replay,
                                     stage_scale, stage_select,
                                     stage_voxel)

    record = _read_job_json(job)
    video = record.get("video")
    if not video:
        raise RuntimeError("job.json has no video (import it first)")
    steps = [
        ("正在抽帧", lambda: stage_extract(Path(video), job)),
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
    total = len(steps)
    for i, (label, fn) in enumerate(steps, 1):
        print(f"[{i}/{total}] {label}…", flush=True)
        info = fn()
        if isinstance(info, dict) and info.get("ok") is False:
            raise RuntimeError(f"{label}失败：{info.get('reason', '未知原因')}")
    return total


def cmd_run(args) -> int:
    try:
        job = resolve_job(Path(args.jobs_root), args.job)
    except FileNotFoundError as e:
        print(f"run: {e}", file=sys.stderr)
        return 1
    try:
        _run_steps(job)
    except Exception as e:  # the exit code is the contract
        print(f"run: failed: {e}", file=sys.stderr)
        return 1
    print(f"完成。job: {job}")
    return 0


# -------------------------------------------------------------------
# depth-video
# -------------------------------------------------------------------

def cmd_depth_video(args) -> int:
    from echo_scene.depth_video import make_depth_video

    video = Path(args.video).expanduser()
    if not video.is_file():
        print(f"depth-video: no such video: {video}", file=sys.stderr)
        return 1
    video = video.resolve()
    if args.out:
        out = Path(args.out).expanduser().resolve()
    else:  # same naming as the desktop app: <stem>_depth.mp4, then _2...
        out = video.with_name(f"{video.stem}_depth.mp4")
        base, n = out, 1
        while out.exists():
            n += 1
            out = base.with_name(f"{base.stem}_{n}.mp4")

    def progress(stage, done, total):
        print(f"{stage} {done}/{total}", flush=True)

    try:
        info = make_depth_video(video, out, fps=args.fps,
                                progress=progress)
    except Exception as e:
        print(f"depth-video: failed: {e}", file=sys.stderr)
        return 1
    if isinstance(info, dict) and info.get("ok") is False:
        print(f"depth-video: failed: {info.get('reason', '未知原因')}",
              file=sys.stderr)
        return 1
    _EVENT_DETAIL["out"] = out.name
    print(f"depth video: {out}")
    return 0


# -------------------------------------------------------------------
# export
# -------------------------------------------------------------------

def cmd_export(args) -> int:
    try:
        job = resolve_job(Path(args.jobs_root), args.job)
    except FileNotFoundError as e:
        print(f"export: {e}", file=sys.stderr)
        return 1
    fmt = args.format
    if fmt in VOXEL_FORMATS:
        from echo_scene.voxel_export import export_colored_voxels

        info = export_colored_voxels(
            job, out_dir=Path(args.out) if args.out else None,
            voxel_size=args.voxel_size, formats=VOXEL_FORMATS[fmt])
        print(json.dumps(info, ensure_ascii=False, indent=2))
        if info.get("ok") and info.get("files"):
            files = info["files"]
            pick = files.get("glb") or next(iter(files.values()), "")
            if pick:
                _EVENT_DETAIL["out"] = Path(pick).name
        return 0 if info.get("ok") else 1
    name = KNOWN_ARTIFACTS[fmt]
    src = job / name
    if not src.exists():
        print(f"export: {name} not in this job (run it first)",
              file=sys.stderr)
        return 1
    dest = src
    if args.out:
        out = Path(args.out).expanduser()
        dest = out / name if out.is_dir() or not out.suffix else out
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    _EVENT_DETAIL["out"] = dest.name
    print(f"exported: {dest}")
    return 0


# -------------------------------------------------------------------
# status
# -------------------------------------------------------------------

def _job_line(job: Path) -> str:
    rec = _read_job_json(job)
    stage = rec.get("stage", "?")
    ok = rec.get("ok")
    state = "ok" if ok is True else ("failed" if ok is False else "?")
    scene = "scene.html ✓" if (job / "scene.html").exists() else "scene.html -"
    return f"{job.name}\tstage={stage}\t{state}\t{scene}\t{job}"


def cmd_status(args) -> int:
    jobs_root = Path(args.jobs_root)
    if args.job:
        try:
            job = resolve_job(jobs_root, args.job)
        except FileNotFoundError as e:
            print(f"status: {e}", file=sys.stderr)
            return 1
        print(_job_line(job))
        for name in sorted(p.name for p in job.iterdir() if p.is_file()):
            print(f"  {name}")
        rec = _read_job_json(job)
        if rec:
            print(json.dumps(rec, ensure_ascii=False, indent=2))
        return 0
    if not jobs_root.is_dir():
        print(f"no jobs root yet: {jobs_root}")
        return 0
    jobs = sorted(p for p in jobs_root.iterdir()
                  if p.is_dir() and (p / "job.json").exists())
    if not jobs:
        print(f"no jobs in {jobs_root}")
        return 0
    for job in jobs:
        print(_job_line(job))
    return 0


# -------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="echo_scene.cli",
        description="EchoScene 命令行：导入、运行、导出，全程不用打开窗口。")
    parser.add_argument("--jobs-root", default=str(default_jobs_root()),
                        help="job 存放目录（默认：桌面端同款 %(default)s）")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("import", help="把视频或采集包登记为一个 job")
    p.add_argument("source", help="视频文件，或含 video.mp4 的采集文件夹")
    p.set_defaults(fn=cmd_import)

    p = sub.add_parser("run", help="跑完整重建（进度打到 stdout）")
    p.add_argument("job", help="job 名或 job 文件夹路径")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("depth-video", help="生成黑白深度视频")
    p.add_argument("video", help="视频文件")
    p.add_argument("--out", help="输出 mp4 路径（默认：视频旁 <名>_depth.mp4）")
    p.add_argument("--fps", type=int, default=8, help="采样帧率（默认 8）")
    p.set_defaults(fn=cmd_depth_video)

    p = sub.add_parser("export", help="导出 job 产物为指定格式")
    p.add_argument("job", help="job 名或 job 文件夹路径")
    p.add_argument("--format", required=True,
                   choices=sorted(VOXEL_FORMATS) + sorted(KNOWN_ARTIFACTS),
                   help="voxel/voxel-glb/voxel-ply = 彩色固化体积（由点云现做）；"
                        "其余为 job 里已有的成品文件")
    p.add_argument("--out", help="输出文件或目录（默认：job 文件夹内）")
    p.add_argument("--voxel-size", type=float, default=None,
                   help="固化体素边长（场景单位；默认按场景大小自动）")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("status", help="列出 job 与产物")
    p.add_argument("job", nargs="?", help="只看这一个 job（名或路径）")
    p.set_defaults(fn=cmd_status)
    return parser


# -------------------------------------------------------------------
# activity events (the frontend's silent toast strip)
# -------------------------------------------------------------------

def _summary_for(args, rc: int) -> str:
    """One product-language line describing what the command just
    did -- the words the desktop frontend's toast shows."""
    ok = rc == 0
    cmd = getattr(args, "command", "?")
    out = _EVENT_DETAIL["out"]
    if cmd == "import":
        src = Path(args.source)
        label = (src.stem if src.suffix.lower() in VIDEO_SUFFIXES
                 else src.name)
        return f"已导入 {label}" if ok else f"导入 {label} 失败"
    if cmd == "run":
        label = Path(args.job).name
        return f"{label} 跑完" if ok else f"{label} 没跑成"
    if cmd == "depth-video":
        if ok:
            return f"深度视频做好了：{out}" if out else "深度视频做好了"
        return "深度视频没做成"
    if cmd == "export":
        label = Path(args.job).name
        if ok:
            return (f"已导出 {out}" if out
                    else f"已导出 {label} 的 {args.format}")
        return f"导出 {label} 失败"
    if cmd == "status":
        return "查了状态" if ok else "查状态失败"
    return f"{cmd} 完成" if ok else f"{cmd} 失败"


def _emit_event(args, rc: int) -> None:
    """Leave one event line for the desktop frontend (see
    echo_app.cli_events): every finished command, success or
    failure, so the user can eyeball that the CLI is alive from
    inside the app. Fully contained -- reporting must never change
    a command's behaviour or its exit code."""
    try:
        from echo_app import cli_events

        cli_events.append_event(getattr(args, "command", "?"),
                                rc == 0, _summary_for(args, rc))
    except Exception:
        pass


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _EVENT_DETAIL["out"] = ""
    try:
        rc = args.fn(args)
    except Exception:
        _emit_event(args, 1)
        raise
    _emit_event(args, rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
