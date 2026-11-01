"""One explicit state that every action's enabled-ness derives from.

As buttons multiply, the classic failure mode is action x state
combinations nobody walked through: starting a second job while one
runs, opening a scene file that was never produced, double-firing a
download. The structural fix kept deliberately at the UI edge (never
in the pipeline): the window holds ONE status -- idle / running /
done / failed, plus which output files actually exist -- and this
pure function maps (status, files) -> {action: enabled}. Handlers
additionally guard themselves (a disabled button is a hint, the
guard is the rule), and the state x action test matrix in
tests/test_desktop.py walks every combination so a future button
gets a row instead of a bug report.

File flags mean "the action can succeed right now": for the scene
and replay that includes "buildable on demand from the job's data",
matching how the app has always opened older jobs.
"""
from dataclasses import dataclass, field
from enum import Enum


class JobState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


# Action ids: every interactive element maps to exactly one.
PICK_VIDEO = "pick_video"
PICK_DEPTH = "pick_depth"
PLAY_DEPTH = "play_depth"
OPEN_SCENE = "open_scene"
OPEN_FOLDER = "open_folder"
OPEN_PREVIEW = "open_preview"
OPEN_REPLAY = "open_replay"
HIST_OPEN = "hist_open"
HIST_FOLDER = "hist_folder"
HIST_PREVIEW = "hist_preview"
HIST_REPLAY = "hist_replay"
OPEN_SETTINGS = "open_settings"
CHANGE_JOBS_FOLDER = "change_jobs_folder"
CHECK_UPDATE = "check_update"
DOWNLOAD_UPDATE = "download_update"
TRAY_OPEN = "tray_open"
TRAY_QUIT = "tray_quit"

ALL_ACTIONS = (
    PICK_VIDEO, PICK_DEPTH, PLAY_DEPTH,
    OPEN_SCENE, OPEN_FOLDER, OPEN_PREVIEW, OPEN_REPLAY,
    HIST_OPEN, HIST_FOLDER, HIST_PREVIEW, HIST_REPLAY,
    OPEN_SETTINGS, CHANGE_JOBS_FOLDER, CHECK_UPDATE, DOWNLOAD_UPDATE,
    TRAY_OPEN, TRAY_QUIT,
)


@dataclass
class FileFlags:
    """Which outputs a job offers right now."""
    scene: bool = False     # scene.html exists, or buildable from points.bin
    folder: bool = False    # the job folder itself exists
    preview: bool = False   # preview.html exists
    replay: bool = False    # replay.html exists, or buildable on demand


@dataclass
class ActionContext:
    job_state: JobState = JobState.IDLE
    current: FileFlags = field(default_factory=FileFlags)
    history: FileFlags = field(default_factory=FileFlags)
    depth_ready: bool = False        # a finished depth video can be played
    update_checking: bool = False
    update_downloading: bool = False


def compute_actions(ctx: ActionContext) -> dict:
    """The matrix itself: {action_id: enabled} for one context."""
    busy = ctx.job_state is JobState.RUNNING
    cur, hist = ctx.current, ctx.history
    return {
        # Starting work: never while a job runs (no second job, ever).
        PICK_VIDEO: not busy,
        PICK_DEPTH: not busy,
        PLAY_DEPTH: not busy and ctx.depth_ready,
        # Current job's outputs: exactly when their file is usable.
        OPEN_SCENE: cur.scene,
        OPEN_FOLDER: cur.folder,
        OPEN_PREVIEW: cur.preview,
        OPEN_REPLAY: cur.replay,
        # Selected history job: same rule, its own files.
        HIST_OPEN: hist.scene,
        HIST_FOLDER: hist.folder,
        HIST_PREVIEW: hist.preview,
        HIST_REPLAY: hist.replay,
        # Settings: always openable; the jobs-folder change waits for
        # the next job (the copy in the settings window says so).
        OPEN_SETTINGS: True,
        CHANGE_JOBS_FOLDER: not busy,
        # Update: one check at a time, one download at a time; the
        # download button only exists once an update is known, and
        # its guard is what makes a double-click harmless.
        CHECK_UPDATE: not ctx.update_checking and not ctx.update_downloading,
        DOWNLOAD_UPDATE: not ctx.update_downloading,
        # Tray: opening is always safe; quitting is always offered --
        # confirming (when a job runs) is the app's job, not the
        # menu's.
        TRAY_OPEN: True,
        TRAY_QUIT: True,
    }
