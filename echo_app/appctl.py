"""Process control for the desktop app -- the CLI's hands on the
running EchoScene process (user directive, 2026-11-04: updates and
restarts must be drivable end-to-end from the CLI, because clicking
through the UI is not always reliable).

There is no IPC or control-file channel into the running app -- the
tray is a pystray menu only its own process can drive -- so control
is process-level, with process exit as the acceptance test:

  find     processes whose executable IS this install's
           EchoScene.exe (path equality, never a name-only match,
           so a stranger's same-named process is never touched).
           On a POSIX source checkout the fallback matches the
           python running this tree's launcher.py.
  close    ask politely first: WM_CLOSE to the process's windows
           (Windows) / SIGTERM (POSIX). The web shell quits on
           window close; the Tk shell may instead hide to the tray
           (close_to_tray) -- a hidden process is still a running
           process, so politeness gets a deadline, not the last
           word.
  kill     whatever is still alive after the grace period is
           terminated. A hot update cannot swap files under a live
           process, and a half-closed app is the one state this
           module exists to prevent.

Every OS touchpoint sits behind an injectable seam (procs_fn /
close_fn / kill_fn / sleep) so the state machine is unit-testable
with a fake process table; the real implementations are stdlib
only (ctypes Toolhelp on Windows, /proc on POSIX).

One honest edge: the frozen launcher's transient --echo-apply
helper shares the exe path, so it can appear in find results for
the few seconds an apply runs. Shutting down then would interrupt
the swap -- the install's own pending/backup repair
(hotapply.rollback_interrupted, run on every start) is the net for
exactly that, and callers should not shutdown blindly mid-apply.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

EXE_NAME = "EchoScene.exe"
_WM_CLOSE = 0x0010


class ProcInfo:
    """One row of the process table: pid, executable path ("" when
    the OS would not say), command line ("" on Windows)."""

    __slots__ = ("pid", "exe", "cmdline")

    def __init__(self, pid, exe="", cmdline=""):
        self.pid = int(pid)
        self.exe = exe or ""
        self.cmdline = cmdline or ""

    def __repr__(self):
        return f"ProcInfo({self.pid}, {self.exe!r})"


# -------------------------------------------------------------------
# the process table
# -------------------------------------------------------------------

def _list_processes_windows():
    import ctypes
    from ctypes import wintypes

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32 = ctypes.windll.kernel32
    snap = kernel32.CreateToolhelp32Snapshot(0x2, 0)  # SNAPPROCESS
    if snap == -1:
        return []
    out = []
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        ok = kernel32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            pid = int(entry.th32ProcessID)
            exe = ""
            handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED
            if handle:
                try:
                    buf = ctypes.create_unicode_buffer(1024)
                    size = wintypes.DWORD(len(buf))
                    if kernel32.QueryFullProcessImageNameW(
                            handle, 0, buf, ctypes.byref(size)):
                        exe = buf.value
                finally:
                    kernel32.CloseHandle(handle)
            out.append(ProcInfo(pid, exe))
            ok = kernel32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snap)
    return out


def _list_processes_posix():
    out = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return out
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            exe = os.readlink(entry / "exe")
        except OSError:
            exe = ""
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(
                b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            cmdline = ""
        out.append(ProcInfo(pid, exe, cmdline))
    return out


def list_processes():
    """The live process table. Never raises; an unreadable table
    reads as empty (callers treat that as 'not running', which the
    acceptance test -- the process must be gone -- still verifies
    honestly at the end of a shutdown)."""
    try:
        if sys.platform == "win32":
            return _list_processes_windows()
        return _list_processes_posix()
    except Exception:
        return []


# -------------------------------------------------------------------
# find / close / kill / launch
# -------------------------------------------------------------------

def app_executable(root) -> Path:
    """The installed executable of the install rooted at `root`."""
    return Path(root) / EXE_NAME


def _norm(path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def find_app_pids(root, procs=None) -> list:
    """Pids of THIS install's app processes. Path equality on the
    executable; on POSIX source checkouts, the python running this
    tree's launcher.py also counts (there is no exe there)."""
    root = Path(root)
    target = _norm(app_executable(root))
    launcher = str(root / "launcher.py")
    out = []
    for proc in (procs if procs is not None else list_processes()):
        if proc.exe and _norm(proc.exe) == target:
            out.append(proc.pid)
        elif (sys.platform != "win32" and proc.cmdline
                and launcher in proc.cmdline):
            out.append(proc.pid)
    return sorted(set(out))


def pid_alive(pid, procs=None) -> bool:
    table = procs if procs is not None else list_processes()
    return any(p.pid == pid for p in table)


def request_close(pid) -> bool:
    """The polite ask: WM_CLOSE to every top-level window the
    process owns (Windows), SIGTERM (POSIX). True when the ask was
    delivered to at least one window / the signal was sent."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        posted = []

        enum_proc_type = ctypes.WINFUNCTYPE(
            wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def _cb(hwnd, _lparam):
            owner = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if int(owner.value) == int(pid):
                user32.PostMessageW(hwnd, _WM_CLOSE, 0, 0)
                posted.append(hwnd)
            return True

        try:
            user32.EnumWindows(enum_proc_type(_cb), 0)
        except Exception:
            return False
        return bool(posted)
    try:
        os.kill(int(pid), 15)  # SIGTERM
        return True
    except OSError:
        return False


def terminate(pid) -> bool:
    """The unpolite certainty: TerminateProcess / SIGKILL."""
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1, False, int(pid))  # TERMINATE
        if not handle:
            return False
        try:
            return bool(kernel32.TerminateProcess(handle, 1))
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 9)  # SIGKILL
        return True
    except OSError:
        return False


def shutdown_app(root, *, grace=6.0, kill_grace=3.0, poll=0.2,
                 procs_fn=None, close_fn=None, kill_fn=None,
                 sleep=None) -> dict:
    """Close this install's app processes and prove they are gone.

    Returns {"was_running", "ok", "method", "pids"} where method is
    "not-running" | "graceful" | "forced" | "failed". The polite
    close gets `grace` seconds; survivors are terminated and get
    `kill_grace` more. ok is the acceptance test itself: no pid
    from the original set is alive at the end."""
    procs_fn = procs_fn or list_processes
    close_fn = close_fn or request_close
    kill_fn = kill_fn or terminate
    sleep = sleep or time.sleep

    def alive(pid):
        try:
            return any(p.pid == pid for p in procs_fn())
        except Exception:
            return False

    pids = find_app_pids(root, procs_fn())
    if not pids:
        return {"was_running": False, "ok": True,
                "method": "not-running", "pids": []}
    for pid in pids:
        try:
            close_fn(pid)
        except Exception:
            pass
    method = "graceful"
    waited = 0.0
    while waited < grace and any(alive(p) for p in pids):
        sleep(poll)
        waited += poll
    survivors = [p for p in pids if alive(p)]
    if survivors:
        method = "forced"
        for pid in survivors:
            try:
                kill_fn(pid)
            except Exception:
                pass
        waited = 0.0
        while waited < kill_grace and any(alive(p) for p in pids):
            sleep(poll)
            waited += poll
    ok = not any(alive(p) for p in pids)
    return {"was_running": True, "ok": ok,
            "method": method if ok else "failed", "pids": pids}


def launch_command(root) -> list:
    """How to start this install: the frozen exe when it exists,
    else the source-tree launcher under the current interpreter."""
    root = Path(root)
    exe = app_executable(root)
    if exe.exists():
        return [str(exe)]
    return [sys.executable, str(root / "launcher.py")]


def start_app(root, popen=None) -> int | None:
    """Launch the app detached (it must outlive this CLI process)
    and return the new pid, or None when the launch itself failed."""
    root = Path(root)
    cmd = launch_command(root)
    kwargs = {"cwd": str(root), "stdin": subprocess.DEVNULL,
              "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL,
              "close_fds": True}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x00000008 | 0x00000200  # DETACHED | NEW_GROUP
    else:
        kwargs["start_new_session"] = True
    try:
        proc = (popen or subprocess.Popen)(cmd, **kwargs)
    except Exception:
        return None
    return getattr(proc, "pid", None)
