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

    # Outputs must be reported per real file, never blanket-lit:
    # a failed job's buttons once lit up and then silently no-oped.
    bridge.job = finished
    outs = bridge.poll().get("jobOutputs") or {}
    check("poll jobOutputs for a finished job",
          outs.get("scene") is True and outs.get("preview") is True
          and outs.get("replay") is False, str(outs))
    bridge.job = running
    outs = bridge.poll().get("jobOutputs") or {}
    check("poll jobOutputs for an unfinished job stay dark",
          outs.get("any") is True and outs.get("scene") is False
          and outs.get("preview") is False
          and outs.get("replay") is False, str(outs))

    # Open actions must speak when they cannot open: no silent no-op.
    bridge.job = running
    r = bridge.open_scene()
    check("open_scene on a job without a scene speaks up",
          r["ok"] is False and "打不开" in bridge.status)
    r = bridge.play_depth()
    check("play_depth with no depth video speaks up",
          r["ok"] is False and "打不开" in bridge.status
          and any("深度视频还没有" in line
                  for line in bridge.poll()["logs"]))

    # And a real depth file goes through play_file exactly once.
    depth_mp4 = tmp / "walk_depth.mp4"
    depth_mp4.write_bytes(b"\0" * 32)
    played = []
    real_play = web_app.backend.play_file
    web_app.backend.play_file = played.append
    try:
        bridge.depth_out = depth_mp4
        r = bridge.play_depth()
    finally:
        web_app.backend.play_file = real_play
    check("play_depth opens the finished depth video",
          r["ok"] is True and played == [depth_mp4])

    check("start guards: no double start while busy",
          (bridge.__dict__.update(busy=True) or
           bridge.start_scene()["ok"] is False))
    bridge.__dict__.update(busy=False)

    # -- phone-capture inbox --------------------------------------
    inbox_root = tmp / "ECHO Captures"
    inbox_root.mkdir()
    os.environ["ECHO_INBOX_DIR"] = str(inbox_root)
    try:
        cap_ok = inbox_root / "capture_20261103_190000"
        cap_ok.mkdir()
        (cap_ok / "video.mp4").write_bytes(b"\0" * 2048)
        (cap_ok / "meta.json").write_text(
            json.dumps({"finalized": True}))
        cap_open = inbox_root / "capture_20261103_190100"
        cap_open.mkdir()
        (cap_open / "video.mp4").write_bytes(b"\0" * 16)
        (cap_open / "meta.json").write_text(
            json.dumps({"finalized": False}))
        cap_tmp = inbox_root / "capture_20261103_190200"
        cap_tmp.mkdir()
        (cap_tmp / "video.mp4").write_bytes(b"\0" * 16)
        (cap_tmp / ".syncthing.video.mp4.tmp").write_bytes(b"\0")

        inbox = bridge._inbox()
        check("inbox lists only the finished capture",
              [e["name"] for e in inbox] == [cap_ok.name],
              str([e["name"] for e in inbox]))
        snap = bridge.get_snapshot()
        check("snapshot carries the inbox", snap["inbox"] == inbox)
        poll = bridge.poll()
        check("poll stays silent when the inbox is unchanged",
              "inbox" not in poll)

        bridge.video = None
        sel = bridge.select_inbox_capture(str(cap_ok))
        check("inbox click selects the capture's video",
              sel["ok"] is True
              and bridge.video == cap_ok / "video.mp4")
        outside = bridge.select_inbox_capture(str(tmp))
        check("a folder outside the inbox roots is refused",
              outside["ok"] is False)
        notyet = bridge.select_inbox_capture(str(cap_tmp))
        check("a half-synced capture cannot be selected",
              notyet["ok"] is False)
    finally:
        del os.environ["ECHO_INBOX_DIR"]

    print(f"FAILURES: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
