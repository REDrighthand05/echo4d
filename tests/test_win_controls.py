"""Frameless window controls on the bridge (v0.5.8).

The native Win32 caption is gone; the page's own buttons call
win_minimize / win_toggle_max / win_close. Pin the bridge side: the
calls reach the window object, toggle alternates maximize/restore,
and a missing window is a no-op rather than a crash.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeWindow:
    def __init__(self):
        self.calls = []

    def minimize(self):
        self.calls.append("minimize")

    def maximize(self):
        self.calls.append("maximize")

    def restore(self):
        self.calls.append("restore")


def main() -> int:
    import web_app

    bridge = web_app.Bridge()
    bridge._window = None
    assert bridge.win_minimize() == {"ok": True}
    assert bridge.win_toggle_max() == {"ok": True}

    fake = FakeWindow()
    bridge._window = fake
    bridge.win_minimize()
    bridge.win_toggle_max()
    bridge.win_toggle_max()
    assert fake.calls == ["minimize", "maximize", "restore"], fake.calls
    print("PASS test_win_controls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
