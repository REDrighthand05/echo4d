"""CLI activity events -- the on-disk channel between the headless
CLI and the web shell's silent toast strip.

The CLI is a separate process that may run while the app is closed,
so the only honest channel is a file: every finished CLI command
appends one JSON line to `cli_events.jsonl` in the settings folder
(the same per-user app-data folder settings.json lives in -- see
echo_app.settings). The web bridge drains the file by sequence
number and the page shows each event as a one-line silent toast
down the right side: the user's eyeball check that the CLI
interface is alive (user directive, 2026-11-04).

Layout in the settings folder:
  cli_events.jsonl  one JSON object per line --
                    {seq, ts, cmd, ok, summary} -- capped at the
                    newest MAX_EVENTS lines.
  cli_events_seq    the last sequence number handed out. A
                    separate counter file (not the line count) so
                    seq stays monotonic across processes and
                    across the cap trimming old lines away.
  cli_events_seen   the bridge's cursor: the highest seq already
                    handed to the page. It persists, so an app
                    restart never replays old news.

Containment: every function here is failure-proof by construction
-- append_event returns False instead of raising, readers return
what they could read. A broken event file must never touch a CLI
exit code or take the shell down.
"""
import json
import time
from pathlib import Path

from echo_app.settings import settings_dir

EVENTS_NAME = "cli_events.jsonl"
SEQ_NAME = "cli_events_seq"
SEEN_NAME = "cli_events_seen"
MAX_EVENTS = 200


def events_path() -> Path:
    return settings_dir() / EVENTS_NAME


def _seq_path() -> Path:
    return settings_dir() / SEQ_NAME


def seen_path() -> Path:
    return settings_dir() / SEEN_NAME


def _read_int(path: Path) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return 0


def _write_int(path: Path, value: int) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(str(value), encoding="utf-8")
    tmp.replace(path)


def append_event(cmd: str, ok: bool, summary: str, ts=None) -> bool:
    """Append one event line, keeping only the newest MAX_EVENTS.
    True when the line landed; False (never an exception) when the
    event could not be written -- the CLI command it reports on
    must not notice either way."""
    try:
        settings_dir().mkdir(parents=True, exist_ok=True)
        seq = _read_int(_seq_path()) + 1
        event = {
            "seq": seq,
            "ts": float(ts if ts is not None else time.time()),
            "cmd": str(cmd),
            "ok": bool(ok),
            "summary": str(summary),
        }
        lines = []
        try:
            lines = events_path().read_text(
                encoding="utf-8").splitlines()
        except OSError:
            lines = []
        lines = [ln for ln in lines if ln.strip()]
        lines.append(json.dumps(event, ensure_ascii=False))
        lines = lines[-MAX_EVENTS:]
        path = events_path()
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp.replace(path)
        _write_int(_seq_path(), seq)
        return True
    except Exception:
        return False


def read_events() -> list:
    """Every retained event, oldest first. Torn or foreign lines
    are skipped, never fatal."""
    try:
        raw = events_path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in raw:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and "seq" in event:
            out.append(event)
    return out


def read_since(cursor: int):
    """(events with seq > cursor, highest seq in the file). The
    second value lets the caller advance its cursor past events it
    chose not to present (e.g. stale ones) instead of re-reading
    them forever."""
    events = read_events()
    hi = max([int(e["seq"]) for e in events], default=int(cursor))
    return [e for e in events if int(e["seq"]) > int(cursor)], hi


def load_cursor() -> int:
    """The bridge's persisted cursor; 0 when none was ever saved."""
    return _read_int(seen_path())


def save_cursor(value: int) -> bool:
    try:
        settings_dir().mkdir(parents=True, exist_ok=True)
        _write_int(seen_path(), int(value))
        return True
    except Exception:
        return False
