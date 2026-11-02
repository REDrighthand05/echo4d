"""Bench for the web shell's bridge (web_app.Bridge).

The bridge must expose the desktop backend faithfully: snapshot
shape, history from real job folders (finished + unfinished),
video validation with the same messages, one-job-at-a-time
guards, and a settings roundtrip -- all without opening a window.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

failures = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))
    if not cond:
        failures.append(name)


def main():
    import web_app
    from echo_app.settings import Settings

    tmp = Path(tempfile.mkdtemp())
    jobs_root = tmp / "jobs"
    jobs_root.mkdir()

    finished = jobs_root / "job_客厅"
    finished.mkdir()
    (finished / "scene.html").write_text("<html></html>")
    (finished / "preview.html").write_text("<html></html>")

    running = jobs_root / "job_卧室"
    running.mkdir()
    (running / "job.json").write_text(json.dumps({"stage": "started"}))
    old = time.time() - 3600
    import os
    os.utime(running, (old, old))

    web_app.load_settings = lambda: Settings(jobs_root=str(jobs_root))
    bridge = web_app.Bridge()

    snap = bridge.get_snapshot()
    check("snapshot carries version + jobs root",
          snap["version"] and snap["jobsRoot"] == str(jobs_root))
    names = [h["name"] for h in snap["history"]]
    check("history lists finished + unfinished, newest first",
          len(names) == 2 and snap["history"][0]["finished"] is True
          and snap["history"][1]["finished"] is False, str(names))
    check("history flags outputs",
          snap["history"][0]["preview"] is True
          and snap["history"][0]["replay"] is False)

    bad = bridge.select_video_path(str(tmp / "note.txt"))
    check("non-video rejected with the desktop message",
          bad["ok"] is False and "不是视频" in bad["message"])
    missing = bridge.select_video_path(str(tmp / "gone.mp4"))
    check("missing video rejected", missing["ok"] is False)

    video = tmp / "walk.mp4"
    video.write_bytes(b"\0" * 16)
    ok = bridge.select_video_path(str(video))
    check("real video selects", ok["ok"] is True and ok["name"] == "walk.mp4")

    poll = bridge.poll()
    check("poll drains log lines once",
          any("已选视频" in line for line in poll["logs"])
          and bridge.poll()["logs"] == [])
    check("poll reports idle state",
          poll["busy"] is False and poll["hasJob"] is False)

    # a job with no outputs cannot open a scene
    bridge.job = running
    check("open_scene on unfinished job is a no-op",
          bridge.open_scene()["ok"] is False)
    check("select_job takes a finished job",
          bridge.select_job(str(finished))["ok"] is True
          and bridge.job == finished)

    saved = bridge.save_settings({"autostart": True})
    check("settings roundtrip", saved["ok"] is True
          and bridge.settings.autostart is True)

    check("start guards: no double start while busy",
          (bridge.__dict__.update(busy=True) or
           bridge.start_scene()["ok"] is False))

    print(f"FAILURES: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
