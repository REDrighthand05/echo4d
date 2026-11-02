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

    from echo_app import updater
    saved = bridge.save_settings(
        {"updateProxy": " http://127.0.0.1:7890 "})
    check("proxy setting roundtrip + live-applied",
          saved["ok"] is True
          and bridge.settings.update_proxy == "http://127.0.0.1:7890"
          and updater._PROXY == "http://127.0.0.1:7890")
    snap = bridge.get_snapshot()
    check("snapshot carries the proxy",
          snap["settings"]["updateProxy"] == "http://127.0.0.1:7890")
    bridge.save_settings({"updateProxy": ""})
    updater.set_proxy("")

    def rel(tag, draft=False):
        return {"tag_name": tag, "draft": draft, "prerelease": False,
                "assets": []}

    releases = [rel("v0.5.1"), rel("v0.5.0"), rel("v0.4.0"),
                rel("v0.6.0", draft=True), rel("junk")]
    hop = updater.pick_hop(releases, "0.4.0")
    check("hop picks the smallest newer step",
          hop is not None and hop["tag_name"] == "v0.5.0")
    hop = updater.pick_hop(releases, "0.5.0")
    check("hop climbs one step", hop is not None
          and hop["tag_name"] == "v0.5.1")
    check("hop at the top is None",
          updater.pick_hop(releases, "0.5.1") is None)
    check("manual check never raises and answers honestly",
          bridge.manual_check_update()["status"] in
          ("update", "latest", "error"))

    # Route cascade: direct dead, first candidate dead, second
    # answers -> it wins, is cached, and is reported as auto-found.
    tried = []
    real_open = updater._urlopen_with

    class _Resp:
        def close(self):
            pass

    def _fake(route, req, timeout):
        tried.append(route)
        if route in ("", updater.CANDIDATE_PROXIES[0]):
            raise OSError("no route")
        return _Resp()

    updater._urlopen_with = _fake
    updater._ROUTE = None
    updater._AUTO_PROXY = ""
    try:
        updater._open("req", 6.0)
        check("cascade walks routes until one answers",
              tried[0] == ""
              and tried[1] == updater.CANDIDATE_PROXIES[0]
              and updater._ROUTE == updater.CANDIDATE_PROXIES[1]
              and updater.auto_proxy() == updater.CANDIDATE_PROXIES[1])

        def _dead(route, req, timeout):
            raise OSError("all dead")

        updater._urlopen_with = _dead
        updater._ROUTE = None
        try:
            updater._open("req", 6.0)
            raised = False
        except OSError:
            raised = True
        check("cascade raises when every route is dead", raised)
    finally:
        updater._urlopen_with = real_open
        updater._ROUTE = None
        updater._AUTO_PROXY = ""

    check("start guards: no double start while busy",
          (bridge.__dict__.update(busy=True) or
           bridge.start_scene()["ok"] is False))

    print(f"FAILURES: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
