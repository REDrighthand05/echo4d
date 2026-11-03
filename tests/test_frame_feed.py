"""Acceptance for the DS WO-B frame_feed kernel: order preserved,
per-item errors yielded in place as FrameError, buffer bounded,
early stop releases the producer."""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echo_scene.frame_feed import FrameError, frame_feed


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run():
    # Order + values.
    items = list(range(20))
    out = list(frame_feed(items, lambda i: i * 10))
    check(out == [i * 10 for i in items], f"order/values: {out[:5]}")

    # Errors in place, feed continues.
    def flaky(i):
        if i in (3, 7):
            raise ValueError(f"bad {i}")
        return i

    out = list(frame_feed(items[:10], flaky))
    check(len(out) == 10, "length preserved")
    for i, v in enumerate(out):
        if i in (3, 7):
            check(isinstance(v, FrameError), f"slot {i} is FrameError")
            check(v.index == i and v.path == i, "FrameError fields")
            check(isinstance(v.original, ValueError), "original kept")
        else:
            check(v == i, f"slot {i} value")

    # Empty input.
    check(list(frame_feed([], lambda i: i)) == [], "empty ok")

    # Bounded prefetch: decoded-but-unconsumed never exceeds depth+1
    # (the +1 is the item in the producer's hand).
    depth = 3
    decoded = [0]
    lock = threading.Lock()

    def counting(i):
        with lock:
            decoded[0] += 1
        return i

    consumed = 0
    worst = 0
    for _ in frame_feed(list(range(30)), counting, depth=depth):
        consumed += 1
        with lock:
            worst = max(worst, decoded[0] - consumed)
        time.sleep(0.002)
    check(worst <= depth + 1, f"prefetch bound: worst lead {worst}")

    # Early stop: producer thread must wind down, not hang.
    slow = frame_feed(list(range(1000)), lambda i: time.sleep(0.01) or i)
    next(slow)
    slow.close()
    deadline = time.time() + 3
    extra = [t for t in threading.enumerate()
             if t.name == "frame_feed-producer"]
    while extra and time.time() < deadline:
        time.sleep(0.1)
        extra = [t for t in threading.enumerate()
                 if t.name == "frame_feed-producer" and t.is_alive()]
    check(not extra, "producer exited after early stop")
    print("PASS test_frame_feed")


if __name__ == "__main__":
    run()
