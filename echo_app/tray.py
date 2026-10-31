"""System-tray residence (WeChat behaviour): closing the window
hides it to the tray and a running job CONTINUES; the tray menu
brings the window back or quits (the app confirms first when a job
is running -- that decision lives in app.py, on the UI thread).

Backend: pystray. Two hard lessons shape this module:

  * pystray's Icon construction touches the platform display backend
    immediately (on a headless Linux box it raises right there), so
    the Icon is only ever constructed inside start(), inside a
    try/except, on a best-effort basis. If the tray cannot start, the
    controller simply reports not-running and the app must fall back
    to plain close = quit -- an invisible, unrecoverable window is
    the one outcome that must never happen.
  * The menu is described by a pure spec (build_menu_spec) so the
    state x action tests can pin labels/enabled-ness headlessly with
    a fake backend, no display involved.

Containment contract: no method here raises. Tray events are posted
to the app's UI queue as ("tray_open", None) / ("tray_quit", None);
this module never touches Tk.
"""
import threading
from pathlib import Path

TITLE = "ECHO 场景生成器"
ICON_NAME = "echo_scene"

ACTION_OPEN = "open"
ACTION_STATUS = "status"
ACTION_QUIT = "quit"

_EVENT_FOR_ACTION = {ACTION_OPEN: "tray_open", ACTION_QUIT: "tray_quit"}


def tooltip_for(status_text: str) -> str:
    return f"{TITLE} — {status_text}"


def build_menu_spec(status_text: str):
    """The tray menu as data: (action, label, visible, enabled) rows
    plus ("separator",) markers. The progress row is a disabled label
    (WeChat-style at-a-glance state), 打开主窗口 is the default
    (double-click) action, 退出 always stays enabled -- quitting is
    confirmed by the app when a job runs, not by greying it out."""
    return (
        (ACTION_OPEN, "打开主窗口", True, True),
        (ACTION_STATUS, f"当前进度：{status_text}", True, False),
        ("separator",),
        (ACTION_QUIT, "退出", True, True),
    )


def _load_icon_image(icon_path):
    """The tray icon image: the shipped installer/echo.ico when it
    loads, else a plain blue square. Never raises."""
    try:
        from PIL import Image

        if icon_path and Path(icon_path).exists():
            img = Image.open(icon_path)
            img.load()
            return img.convert("RGBA")
    except Exception:
        pass
    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGBA", (64, 64), (0, 100, 209, 255))
        draw = ImageDraw.Draw(img)
        draw.ellipse((18, 18, 46, 46), fill=(255, 255, 255, 255))
        draw.ellipse((27, 27, 37, 37), fill=(0, 100, 209, 255))
        return img
    except Exception:
        return None


class _PystrayBackend:
    """The real backend. Constructed lazily inside TrayController.
    start() so a missing display/backend only costs the tray, never
    the app."""

    def __init__(self, icon_path):
        self._icon_path = icon_path
        self._icon = None

    def _build_menu(self, spec, on_action):
        import pystray

        items = []
        for row in spec:
            if row[0] == "separator":
                items.append(pystray.Menu.SEPARATOR)
                continue
            action, label, _visible, enabled = row

            def handler(_icon, _item, _action=action):
                on_action(_action)

            items.append(pystray.MenuItem(
                label, handler if enabled or action != ACTION_STATUS else None,
                enabled=enabled, default=(action == ACTION_OPEN)))
        return pystray.Menu(*items)

    def start(self, spec, tooltip, on_action):
        import pystray

        image = _load_icon_image(self._icon_path)
        if image is None:
            raise RuntimeError("no tray icon image")
        self._on_action = on_action
        self._icon = pystray.Icon(ICON_NAME, image, tooltip,
                                  self._build_menu(spec, on_action))
        thread = threading.Thread(target=self._run, daemon=True,
                                  name="echo-tray")
        thread.start()

    def _run(self):
        try:
            self._icon.run()
        except Exception:
            pass  # the loop dying must not take the app with it

    def set_status(self, spec, tooltip):
        if self._icon is None:
            return
        self._icon.menu = self._build_menu(spec, self._on_action)
        self._icon.title = tooltip
        self._icon.update_menu()

    def stop(self):
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:
                pass
            self._icon = None


class TrayController:
    """Owns the tray icon's lifecycle and status text.

    events: the app's UI queue; tray actions arrive there as
    ("tray_open", None) / ("tray_quit", None) so every reaction
    (showing the window, confirming a quit) happens on the UI thread.
    backend: injectable for tests; None = the pystray backend.
    """

    def __init__(self, events, icon_path=None, backend=None):
        self._events = events
        self._icon_path = icon_path
        self._backend = backend
        self._running = False
        self._status = "空闲"

    @property
    def running(self) -> bool:
        return self._running

    @property
    def status(self) -> str:
        return self._status

    def _on_action(self, action):
        kind = _EVENT_FOR_ACTION.get(action)
        if kind is not None:
            self._events.put((kind, None))

    def start(self) -> bool:
        """Bring the tray icon up. False (never an exception) when the
        platform/backend refuses -- the caller must then not hide the
        window to a tray that does not exist."""
        if self._running:
            return True
        try:
            if self._backend is None:
                self._backend = _PystrayBackend(self._icon_path)
            self._backend.start(build_menu_spec(self._status),
                                tooltip_for(self._status), self._on_action)
            self._running = True
        except Exception:
            self._running = False
        return self._running

    def set_status(self, status_text: str):
        """Refresh the progress row + tooltip. Safe to call when the
        tray never started (it just records the text)."""
        self._status = status_text
        if not self._running or self._backend is None:
            return
        try:
            self._backend.set_status(build_menu_spec(status_text),
                                     tooltip_for(status_text))
        except Exception:
            pass  # a stale tooltip is a nuisance, never a crash

    def stop(self):
        if self._backend is not None:
            try:
                self._backend.stop()
            except Exception:
                pass
        self._running = False
