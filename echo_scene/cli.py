#!/usr/bin/env python3
"""EchoScene command line (user directive, 2026-11-03): the desktop
software must be drivable without its window -- import, run, export,
all from the shell, so Muse (or a script) can operate it directly.

    python -m echo_scene.cli import <video|capture_dir> [--jobs-root D]
    python -m echo_scene.cli run <job>
    python -m echo_scene.cli depth-video <video> [--out P] [--fps N]
    python -m echo_scene.cli export <job> --format F [--out P]
    python -m echo_scene.cli status [<job>]
    python -m echo_scene.cli shutdown [--force]
    python -m echo_scene.cli restart [--force]
    python -m echo_scene.cli update [--check] [--no-restart] [--force]

shutdown / restart / update are the app-lifecycle commands (user
directive, 2026-11-04): the software updates often, and clicking
through its window is not always reliable, so closing, updating
and relaunching must all work as one remote CLI command each.
They reuse the existing hot-update machinery end to end
(echo_app.hotupdate to check/stage, echo_app.hotapply to swap,
echo_app.appctl for process control) -- no second update logic
lives here.

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
# activity event (below) can name the actual file. "note" lets a
# lifecycle command dictate its event line outright (a shutdown
# refused because a job is running must say exactly that).
# Purely cosmetic -- nothing branches on either.
_EVENT_DETAIL = {"out": "", "note": ""}


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
# app lifecycle: shutdown / restart / update
# -------------------------------------------------------------------

# Stages after which a job counts as finished for the lifecycle
# guards. Anything earlier with fresh files on disk is treated as
# mid-run (see _running_job).
_TERMINAL_STAGES = {"present", "done", "complete"}


def _running_job(jobs_root: Path, fresh_seconds: float = 180.0,
                 now: float | None = None):
    """The job under jobs_root that looks mid-run right now, or
    None. Cross-process evidence is all we have: a job is running
    when its record is not in a finished state AND its folder was
    written to within `fresh_seconds` -- the heavy stages (poses,
    depth, fuse) write into the job folder as they go, so a live
    job's folder is never quiet for long. A finished job (ok False,
    or a terminal stage with ok True) never counts, however fresh
    its files are. Deliberately conservative in one direction only:
    callers refuse to close/update over a 'running' verdict unless
    --force, and say so out loud."""
    import time as _time

    now = _time.time() if now is None else now
    if not jobs_root.is_dir():
        return None
    for job in sorted(jobs_root.iterdir()):
        if not job.is_dir() or not (job / "job.json").exists():
            continue
        rec = _read_job_json(job)
        if rec.get("ok") is False:
            continue                      # failed = finished
        if rec.get("ok") is True and rec.get("stage") in _TERMINAL_STAGES:
            continue                      # done = finished
        if rec.get("stage") in (None, "imported"):
            continue                      # never started
        newest = 0.0
        for path in job.rglob("*"):
            try:
                newest = max(newest, path.stat().st_mtime)
            except OSError:
                continue
            if now - newest <= fresh_seconds:
                return job
    return None


def _lifecycle_event(summary: str, ok: bool = True) -> None:
    """One step line for the frontend toast strip, in the same
    format every other CLI event uses (see _emit_event)."""
    try:
        from echo_app import cli_events

        cli_events.append_event("update", ok, summary)
    except Exception:
        pass


def _shutdown_flow(jobs_root: Path, force: bool):
    """Shared by shutdown/restart/update: (rc, was_running).
    rc 0 means the app is down (or was never up) and the caller
    may proceed; the job guard refuses over a mid-run job unless
    force, and leaves the app untouched when it refuses."""
    from echo_app import appctl, hotupdate

    root = hotupdate.install_root()
    if not appctl.find_app_pids(root):
        return 0, False
    if not force:
        job = _running_job(jobs_root)
        if job is not None:
            print(f"有 job 正在跑（{job.name}），先等它跑完，"
                  "或加 --force 强行关闭。", file=sys.stderr)
            _EVENT_DETAIL["note"] = "有 job 在跑，没关软件"
            return 1, True
    result = appctl.shutdown_app(root)
    if not result["ok"]:
        print("软件进程没能退干净，关不掉。", file=sys.stderr)
        _EVENT_DETAIL["note"] = "软件关不掉"
        return 1, True
    return 0, True


def cmd_shutdown(args) -> int:
    rc, was_running = _shutdown_flow(Path(args.jobs_root), args.force)
    if rc == 0:
        print("EchoScene 已退出。" if was_running
              else "EchoScene 没有在运行。")
        _EVENT_DETAIL["note"] = "已关掉软件" if was_running else "软件没开着"
    return rc


def _confirm_started(root, timeout: float = 6.0) -> int | None:
    """Poll for the relaunched app's pid; None when it never shows."""
    import time as _time

    from echo_app import appctl

    deadline = _time.time() + timeout
    while _time.time() < deadline:
        pids = appctl.find_app_pids(root)
        if pids:
            return pids[0]
        _time.sleep(0.25)
    return None


def cmd_restart(args) -> int:
    from echo_app import appctl, hotupdate

    root = hotupdate.install_root()
    rc, was_running = _shutdown_flow(Path(args.jobs_root), args.force)
    if rc != 0:
        return rc
    pid = appctl.start_app(root)
    shown = _confirm_started(root) if pid is not None else None
    if shown is None:
        print("restart: 软件没能重新拉起来。", file=sys.stderr)
        _EVENT_DETAIL["note"] = "重启没拉起来"
        return 1
    print(f"EchoScene 已{'重新' if was_running else ''}启动（pid {shown}）。")
    _EVENT_DETAIL["note"] = "软件已重启"
    return 0


def _update_plan(current: str):
    """(latest, [release dicts in climb order]) for the update
    command: the relay chain first (no API quota), the GitHub
    releases chain as fallback -- the same sources, in the same
    order, the app itself checks. Raises when neither answers."""
    from echo_app import updater

    try:
        from echo_app.settings import load_settings

        updater.set_proxy(load_settings().update_proxy)
    except Exception:
        pass
    try:
        chain = updater.relay_chain(updater.fetch_relay_index(), current)
    except Exception:
        chain = None
    if chain is not None:
        return chain
    hops = updater.github_chain(updater.fetch_releases(), current)
    if not hops:
        return current, []
    from echo_app.version import parse_version

    key = parse_version(hops[-1].get("tag_name"))
    latest = ".".join(str(p) for p in key) if key else current
    return latest, hops


def _relaunch_after_update(root, was_running: bool, no_restart: bool):
    """Bring the app back after an apply phase, per the command's
    contract: only when it was running before, and never under
    --no-restart. Returns the new pid or None."""
    if not was_running or no_restart:
        return None
    from echo_app import appctl

    pid = appctl.start_app(root)
    if pid is None:
        return None
    return _confirm_started(root)


def cmd_update(args) -> int:
    from echo_app import appctl, hotapply, hotupdate

    root = hotupdate.install_root()
    jobs_root = Path(args.jobs_root)
    try:
        hotupdate.recover(root)   # repair a previously interrupted apply
    except Exception:
        pass
    current = hotupdate.local_version(root)
    _lifecycle_event("开始检查更新")
    print(f"当前版本：{current}", flush=True)
    try:
        latest, hops = _update_plan(current)
    except Exception as e:
        print(f"update: 检查更新没连上（{type(e).__name__}: {e}）",
              file=sys.stderr)
        _EVENT_DETAIL["note"] = "检查更新没连上"
        return 1
    if not hops:
        print(f"已是最新版（{current}），不用更新。")
        _EVENT_DETAIL["note"] = f"已是最新 {current}"
        return 0
    print(f"可更新到 {latest}，共 {len(hops)} 跳。", flush=True)
    if args.check:
        _EVENT_DETAIL["note"] = f"有新版 {latest}（{len(hops)} 跳）"
        return 0

    was_running = bool(appctl.find_app_pids(root))
    if was_running and not args.force:
        job = _running_job(jobs_root)
        if job is not None:
            print(f"有 job 正在跑（{job.name}），更新取消；"
                  "等它跑完再更，或加 --force。", file=sys.stderr)
            _EVENT_DETAIL["note"] = "有 job 在跑，更新取消"
            return 1
    closed = False
    applied = current
    try:
        for i, hop in enumerate(hops, 1):
            hot = hotupdate.check_hot_update(hop, root)
            if hot is None:
                tag = hop.get("tag_name", "?")
                raise RuntimeError(
                    f"第 {i} 跳（{tag}）走不了热更新通道，"
                    "需要完整安装包，CLI 不自动装整包")
            _lifecycle_event(f"正在下载更新（第 {i}/{len(hops)} 跳）")
            print(f"[{i}/{len(hops)}] 下载并校验 v{hot.version}…",
                  flush=True)
            hotupdate.download_and_stage(root, hot)
            if was_running and not closed:
                rc, _ = _shutdown_flow(jobs_root, args.force)
                if rc != 0:
                    raise RuntimeError("软件没能关掉，更新中止")
                closed = True
                _lifecycle_event("已关闭软件，开始换文件")
            hotupdate.request_apply(root, hot)
            result = hotapply.apply(root)
            if not result.get("ok"):
                raise RuntimeError(
                    f"第 {i} 跳应用失败（已回滚）：{result.get('error')}")
            applied = hot.version
            _lifecycle_event(f"已更新到 {applied}")
            print(f"[{i}/{len(hops)}] 已应用 v{applied}。", flush=True)
    except Exception as e:
        print(f"update: failed: {e}", file=sys.stderr)
        try:
            hotupdate.recover(root)
        except Exception:
            pass
        if not _EVENT_DETAIL["note"]:
            # _shutdown_flow may already have written the specific
            # reason (a job started mid-download); keep it.
            _EVENT_DETAIL["note"] = (f"更新没成（停在 {applied}）"
                                      if applied != current
                                      else "更新没成")
        _relaunch_after_update(root, was_running, args.no_restart)
        return 1

    print(f"完成。当前版本：{applied}")
    if was_running and not args.no_restart:
        shown = _relaunch_after_update(root, was_running, args.no_restart)
        if shown is None:
            print("update: 更新好了，但软件没能重新拉起来。",
                  file=sys.stderr)
            _EVENT_DETAIL["note"] = "更新好了，但没拉起来"
            return 1
        print(f"EchoScene 已重新启动（pid {shown}）。")
        _lifecycle_event(f"更新完成，已重启到 {applied}")
        _EVENT_DETAIL["note"] = f"已更新到 {applied} 并重启"
    elif was_running:
        print("更新已应用，软件保持关闭（--no-restart）。")
        _EVENT_DETAIL["note"] = f"已更新到 {applied}"
    else:
        _EVENT_DETAIL["note"] = f"已更新到 {applied}"
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

    p = sub.add_parser("shutdown", help="把正在运行的软件干净关掉"
                                        "（没在运行也算成功）")
    p.add_argument("--force", action="store_true",
                   help="有 job 在跑也关（默认先拒绝）")
    p.set_defaults(fn=cmd_shutdown)

    p = sub.add_parser("restart", help="关掉软件再重新拉起（没开着就直接拉起）")
    p.add_argument("--force", action="store_true",
                   help="有 job 在跑也重启（默认先拒绝）")
    p.set_defaults(fn=cmd_restart)

    p = sub.add_parser("update", help="热更新全自动：逐跳下载、校验、"
                                      "关软件、换文件、重新拉起")
    p.add_argument("--check", action="store_true",
                   help="只查不更：报当前版、最新版和待跳数")
    p.add_argument("--no-restart", action="store_true",
                   help="只应用更新，不重新拉起软件")
    p.add_argument("--force", action="store_true",
                   help="有 job 在跑也更新（默认先拒绝）")
    p.set_defaults(fn=cmd_update)
    return parser


# -------------------------------------------------------------------
# activity events (the frontend's silent toast strip)
# -------------------------------------------------------------------

def _summary_for(args, rc: int) -> str:
    """One product-language line describing what the command just
    did -- the words the desktop frontend's toast shows."""
    ok = rc == 0
    cmd = getattr(args, "command", "?")
    if _EVENT_DETAIL["note"]:
        return _EVENT_DETAIL["note"]
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
    if cmd == "shutdown":
        return "已关掉软件" if ok else "关软件失败"
    if cmd == "restart":
        return "软件已重启" if ok else "重启失败"
    if cmd == "update":
        return "更新完成" if ok else "更新没成"
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
    _EVENT_DETAIL["note"] = ""
    try:
        rc = args.fn(args)
    except Exception:
        _emit_event(args, 1)
        raise
    _emit_event(args, rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
