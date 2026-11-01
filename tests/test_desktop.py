#!/usr/bin/env python3
"""v0.4 desktop-shell locks: version, settings, autostart, updater,
notify, tray -- and the state x action matrix.

The matrix is the durable form of the hardening: every main-window
action is walked across every job state and every combination of
output-file flags, and its enabled-ness must equal the rules restated
independently in expected_actions() below. A future button gets a
row here, not a bug report.

Also covered: crash recovery at the app seam -- a job folder holding
only the marker run_pipeline writes at job start (plus whatever
partial outputs exist) must list as unfinished, with usable outputs
still offered.

No network, no display: the updater parses a recorded API payload
(tests/fixtures/releases_latest.json) and downloads through an
injected opener; the tray runs against a fake backend.

Run: .venv/bin/python tests/test_desktop.py
"""
import itertools
import json
import queue
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from echo_app import appstate, autostart, settings as settings_mod  # noqa: E402
from echo_app import tray as tray_mod, updater                     # noqa: E402
from echo_app.notify import notify                                  # noqa: E402
from echo_app.version import (APP_VERSION, is_newer,                # noqa: E402
                              parse_version, parse_version_inc)

FAILURES = []


def _check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------
def version_checks():
    print("[version]")
    inc_text = (ROOT / "installer" / "version.inc").read_text(encoding="utf-8")
    _check("app version == installer/version.inc (single source)",
           APP_VERSION == parse_version_inc(inc_text), APP_VERSION)
    _check("app version parses", parse_version(APP_VERSION) is not None)
    _check("parse v-tag", parse_version("v1.2.3") == (1, 2, 3))
    _check("parse bare", parse_version("0.10.9") == (0, 10, 9))
    _check("parse rejects junk",
           parse_version("v1.2") is None and parse_version("x") is None
           and parse_version(None) is None and parse_version(12) is None)
    _check("newer: minor bump", is_newer("v0.4.0", "0.3.0"))
    _check("newer: patch bump", is_newer("0.3.1", "0.3.0"))
    _check("newer: 1.0.0 > 0.9.9", is_newer("v1.0.0", "0.9.9"))
    _check("not newer: equal", not is_newer("v0.4.0", "0.4.0"))
    _check("not newer: older", not is_newer("v0.3.0", "0.4.0"))
    _check("not newer: garbled tag", not is_newer("release-x", "0.3.0"))
    _check("inc parse from text",
           parse_version_inc('#define AppVersion "2.0.1"') == "2.0.1")
    _check("inc parse missing", parse_version_inc("nothing") is None)


def settings_checks():
    print("[settings]")
    _check("win settings path uses APPDATA",
           settings_mod.settings_path("win32", {"APPDATA": r"C:\U\AppData\Roaming"})
           == Path(r"C:\U\AppData\Roaming") / "ECHO" / "settings.json")
    _check("default jobs root matches the app's historic root",
           settings_mod.default_jobs_root() == Path.home() / "ECHO Jobs")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "sub" / "settings.json"
        fresh = settings_mod.load_settings(path)
        _check("missing file -> defaults",
               fresh.jobs_root == "" and fresh.autostart is False
               and fresh.close_to_tray is True)
        _check("default jobs_root_path", fresh.jobs_root_path()
               == settings_mod.default_jobs_root())
        custom = settings_mod.Settings(jobs_root="/somewhere/else",
                                       autostart=True, close_to_tray=False)
        _check("save ok", settings_mod.save_settings(custom, path))
        back = settings_mod.load_settings(path)
        _check("roundtrip", back == custom, str(back))
        _check("no torn temp left", not path.with_name("settings.json.tmp").exists())
        path.write_text("{not json", encoding="utf-8")
        _check("corrupt file -> defaults",
               settings_mod.load_settings(path) == settings_mod.Settings())
        path.write_text(json.dumps({"jobs_root": 42, "autostart": "yes",
                                    "close_to_tray": False, "extra": 1}),
                        encoding="utf-8")
        coerced = settings_mod.load_settings(path)
        _check("wrong types fall back per-field",
               coerced.jobs_root == "" and coerced.autostart is False
               and coerced.close_to_tray is False)
        blocker = Path(td) / "blocker"
        blocker.write_text("file, not a dir")
        _check("save failure is contained (False, no raise)",
               settings_mod.save_settings(custom, blocker / "s.json") is False)


def autostart_checks():
    print("[autostart]")
    _check("frozen command shape",
           autostart.launch_command(r"C:\Apps\EchoScene.exe")
           == r'"C:\Apps\EchoScene.exe" --tray')
    with tempfile.TemporaryDirectory() as td:
        _check("off by default",
               not autostart.is_autostart_enabled("linux", td))
        _check("enable writes entry",
               autostart.set_autostart(True, platform="linux", home=td))
        cmd = autostart.autostart_command("linux", td)
        _check("entry carries --tray", cmd is not None and "--tray" in cmd, str(cmd))
        _check("disable is idempotent",
               autostart.set_autostart(False, platform="linux", home=td)
               and autostart.set_autostart(False, platform="linux", home=td)
               and not autostart.is_autostart_enabled("linux", td))


class _FakeResponse:
    def __init__(self, data: bytes):
        self._data = data

    def read(self, n=-1):
        chunk, self._data = self._data[:n], self._data[n:]
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def updater_checks():
    print("[updater]")
    payload = json.loads(
        (ROOT / "tests" / "fixtures" / "releases_latest.json")
        .read_text(encoding="utf-8"))
    info = updater.parse_release(payload, "0.3.0")
    _check("fixture: update found for 0.3.0", info is not None)
    _check("fixture: version/tag",
           info.version == "0.4.0" and info.tag == "v0.4.0")
    _check("fixture: setup exe chosen (not the zip)",
           info.download_url.endswith("/EchoScene-Setup.exe"),
           info.download_url)
    _check("same version -> no update",
           updater.parse_release(payload, "0.4.0") is None)
    _check("newer current -> no update",
           updater.parse_release(payload, "0.5.0") is None)

    zip_only = dict(payload, assets=[a for a in payload["assets"]
                                     if a["name"].endswith(".zip")])
    _check("zip-only release -> no nag",
           updater.parse_release(zip_only, "0.3.0") is None)
    _check("draft -> no update",
           updater.parse_release(dict(payload, draft=True), "0.3.0") is None)
    _check("prerelease -> no update",
           updater.parse_release(dict(payload, prerelease=True), "0.3.0") is None)
    for bad in (None, [], "x", {"tag_name": 5}, {"tag_name": "v9.9.9"},
                {"tag_name": "v9.9.9", "assets": "nope"},
                {"tag_name": "not-a-version", "assets": payload["assets"]}):
        _check(f"malformed payload contained: {str(bad)[:28]!r}",
               updater.parse_release(bad, "0.3.0") is None)

    _check("check via injected fetch",
           updater.check_for_update("0.3.0", fetch=lambda: payload) is not None)

    def boom():
        raise OSError("offline")

    _check("fetch failure -> None (fail-silent)",
           updater.check_for_update("0.3.0", fetch=boom) is None)
    _check("status: update",
           updater.check_for_update_status("0.3.0", fetch=lambda: payload)[0]
           == "update")
    _check("status: latest",
           updater.check_for_update_status("0.4.0", fetch=lambda: payload)
           == ("latest", None))
    _check("status: error",
           updater.check_for_update_status("0.3.0", fetch=boom)
           == ("error", None))

    with tempfile.TemporaryDirectory() as td:
        got = updater.download_installer(
            info, Path(td), open_url=lambda url: _FakeResponse(b"SETUP-BYTES"))
        _check("download lands with versioned name",
               got.name == "EchoScene-Setup-0.4.0.exe"
               and got.read_bytes() == b"SETUP-BYTES")
        _check("no .part left after success",
               not list(Path(td).glob("*.part")))

        def bad_open(url):
            raise OSError("connection reset")

        try:
            updater.download_installer(info, Path(td), open_url=bad_open)
            raised = False
        except OSError:
            raised = True
        _check("download failure raises for the caller to phrase", raised)
        _check("no .part left after failure",
               not list(Path(td).glob("*.part")))


def tray_checks():
    print("[tray]")
    spec = tray_mod.build_menu_spec("正在拼场景 · 42%")
    rows = {row[0]: row for row in spec if row[0] != "separator"}
    _check("menu has a separator", any(r[0] == "separator" for r in spec))
    _check("open row: enabled, labelled",
           rows["open"][1] == "打开主窗口" and rows["open"][3] is True)
    _check("status row: shows progress, disabled",
           "42%" in rows["status"][1] and rows["status"][3] is False)
    _check("quit row: enabled", rows["quit"][1] == "退出"
           and rows["quit"][3] is True)
    _check("tooltip carries state",
           "正在拼场景" in tray_mod.tooltip_for("正在拼场景 · 42%"))

    class FakeBackend:
        def __init__(self):
            self.started = None
            self.statuses = []
            self.stopped = False

        def start(self, spec_, tooltip, on_action):
            self.started = (spec_, tooltip)
            self.on_action = on_action

        def set_status(self, spec_, tooltip):
            self.statuses.append((spec_, tooltip))

        def stop(self):
            self.stopped = True

    events = queue.Queue()
    fake = FakeBackend()
    ctl = tray_mod.TrayController(events, backend=fake)
    _check("controller starts on fake backend", ctl.start() and ctl.running)
    _check("start is idempotent", ctl.start() and ctl.running)
    ctl.set_status("已完成")
    _check("status reaches backend",
           fake.statuses and "已完成" in fake.statuses[-1][1])
    fake.on_action("open")
    fake.on_action("status")   # a disabled row firing anyway: contained
    fake.on_action("quit")
    got = []
    while not events.empty():
        got.append(events.get_nowait())
    _check("actions become UI-queue events",
           got == [("tray_open", None), ("tray_quit", None)], str(got))
    ctl.stop()
    _check("stop reaches backend, running cleared",
           fake.stopped and not ctl.running)

    class ExplodingBackend:
        def start(self, *a):
            raise RuntimeError("no display")

        def set_status(self, *a):
            raise RuntimeError("no display")

        def stop(self):
            raise RuntimeError("no display")

    ctl2 = tray_mod.TrayController(queue.Queue(), backend=ExplodingBackend())
    _check("exploding backend: start contained",
           ctl2.start() is False and not ctl2.running)
    ctl2.set_status("x")   # must not raise
    ctl2.stop()            # must not raise
    _check("exploding backend: status/stop contained", True)

    real = tray_mod.TrayController(queue.Queue())
    outcome = real.start()   # this VM is headless: False; a desktop: True
    _check("real backend: start never raises", outcome in (True, False))
    real.set_status("空闲")
    real.stop()
    _check("real backend: stop never raises", not real.running)


def notify_checks():
    print("[notify]")
    result = notify("标题", "内容")
    if sys.platform.startswith("win"):
        _check("notify returns a bool on Windows", isinstance(result, bool))
    else:
        _check("notify is a silent no-op off Windows", result is False)


# ---------------------------------------------------------------------
def expected_actions(ctx):
    """The matrix rules, restated in the test's own words."""
    busy = ctx.job_state is appstate.JobState.RUNNING
    cur, hist = ctx.current, ctx.history
    return {
        appstate.PICK_VIDEO: not busy,
        appstate.PICK_DEPTH: not busy,
        appstate.PLAY_DEPTH: not busy and ctx.depth_ready,
        appstate.OPEN_SCENE: cur.scene,
        appstate.OPEN_FOLDER: cur.folder,
        appstate.OPEN_PREVIEW: cur.preview,
        appstate.OPEN_REPLAY: cur.replay,
        appstate.HIST_OPEN: hist.scene,
        appstate.HIST_FOLDER: hist.folder,
        appstate.HIST_PREVIEW: hist.preview,
        appstate.HIST_REPLAY: hist.replay,
        appstate.OPEN_SETTINGS: True,
        appstate.CHANGE_JOBS_FOLDER: not busy,
        appstate.CHECK_UPDATE: not ctx.update_checking
        and not ctx.update_downloading,
        appstate.DOWNLOAD_UPDATE: not ctx.update_downloading,
        appstate.TRAY_OPEN: True,
        appstate.TRAY_QUIT: True,
    }


def matrix_checks():
    print("[state x action matrix]")
    flag_combos = [appstate.FileFlags(*bits) for bits in
                   itertools.product([False, True], repeat=4)]
    walked = 0
    mismatches = []
    for job_state in appstate.JobState:
        for cur in flag_combos:
            for hist in flag_combos:
                for depth_ready, checking, downloading in itertools.product(
                        [False, True], repeat=3):
                    ctx = appstate.ActionContext(
                        job_state=job_state, current=cur, history=hist,
                        depth_ready=depth_ready, update_checking=checking,
                        update_downloading=downloading)
                    got = appstate.compute_actions(ctx)
                    if set(got) != set(appstate.ALL_ACTIONS):
                        mismatches.append(("action set", ctx))
                        continue
                    want = expected_actions(ctx)
                    if got != want:
                        mismatches.append((ctx, got, want))
                    walked += 1
    _check(f"matrix walked ({walked} contexts)", not mismatches,
           f"{walked} contexts, {len(mismatches)} mismatches")
    if mismatches:
        print("   first mismatch:", mismatches[0])

    # Spot rows a human would ask about by name:
    running = appstate.ActionContext(
        job_state=appstate.JobState.RUNNING,
        current=appstate.FileFlags(scene=True, folder=True))
    acts = appstate.compute_actions(running)
    _check("while running: cannot start another job",
           acts[appstate.PICK_VIDEO] is False
           and acts[appstate.PICK_DEPTH] is False)
    _check("while running: previous outputs still openable",
           acts[appstate.OPEN_SCENE] is True)
    _check("while running: jobs-folder change waits",
           acts[appstate.CHANGE_JOBS_FOLDER] is False)
    _check("while running: tray quit still offered (app confirms)",
           acts[appstate.TRAY_QUIT] is True)
    empty = appstate.compute_actions(appstate.ActionContext())
    _check("idle with no job: every output action calm-disabled",
           not any(empty[a] for a in (
               appstate.OPEN_SCENE, appstate.OPEN_FOLDER,
               appstate.OPEN_PREVIEW, appstate.OPEN_REPLAY,
               appstate.HIST_OPEN, appstate.HIST_FOLDER,
               appstate.HIST_PREVIEW, appstate.HIST_REPLAY,
               appstate.PLAY_DEPTH)))


def app_seam_checks():
    print("[app seams: history + crash recovery]")
    import app as app_module

    _check("app default root == settings default root",
           app_module.JOBS_ROOT == settings_mod.default_jobs_root())
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "ECHO Jobs"
        root.mkdir()

        def mk(name, files, age):
            d = root / name
            d.mkdir()
            for f in files:
                (d / f).write_bytes(b"x")
            import os
            os.utime(d, (age, age))
            return d

        finished = mk("job_done", ["scene.html", "job.json"], 3000)
        old = mk("job_old", ["viewer.html"], 2000)
        crashed_early = mk("job_crash1", ["job.json"], 4000)
        crashed_late = mk("job_crash2",
                          ["job.json", "points.bin", "poses.json"], 5000)
        mk("stray", ["random.txt"], 6000)   # not ours: no marker at all

        jobs = app_module.list_previous_jobs(root)
        names = [j.name for j in jobs]
        _check("history lists finished + crashed, skips stray",
               names == ["job_crash2", "job_crash1", "job_done", "job_old"],
               str(names))
        _check("finished detection",
               app_module.job_finished(finished)
               and app_module.job_finished(old)
               and not app_module.job_finished(crashed_early))
        _check("unfinished marked in display name",
               "未完成" in app_module._job_display_name(crashed_early)
               and "未完成" not in app_module._job_display_name(finished))
        flags = app_module.job_file_flags(crashed_late)
        _check("crashed-after-fuse: scene+replay still usable",
               flags.scene and flags.replay and flags.folder
               and not flags.preview)
        flags = app_module.job_file_flags(crashed_early)
        _check("crashed-early: only the folder opens",
               flags.folder and not flags.scene and not flags.replay)
        _check("no job selected -> all flags false",
               app_module.job_file_flags(None) == appstate.FileFlags())

    # run_pipeline writes the marker before the first stage runs, so
    # even a job whose very first stage blows up is recoverable.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "jobs"
        root.mkdir()
        try:
            app_module.run_pipeline(Path(td) / "missing.mp4", root,
                                    lambda _msg: None)
            raised = False
        except Exception:
            raised = True
        _check("pipeline on a missing video fails (as before)", raised)
        leftovers = list(root.iterdir())
        _check("the failed job folder remains, marked",
               len(leftovers) == 1
               and (leftovers[0] / "job.json").exists())
        if leftovers:
            marker = json.loads((leftovers[0] / "job.json").read_text())
            _check("marker says running/started",
                   marker.get("status") == "running"
                   and marker.get("stage") == "started")
            _check("and history picks it up as unfinished",
                   app_module.list_previous_jobs(root) == leftovers
                   and not app_module.job_finished(leftovers[0]))


def main():
    version_checks()
    settings_checks()
    autostart_checks()
    updater_checks()
    tray_checks()
    notify_checks()
    matrix_checks()
    app_seam_checks()
    print("\n" + ("ALL PASS" if not FAILURES else f"FAILURES: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
