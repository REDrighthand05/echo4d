"""Prefetching producer/consumer iterator.

Values are yielded in the original input order. If decode_fn raises an
Exception subclass for an item, a FrameError instance is yielded at that
item's position; exceptions are never raised to the consumer for normal
decode failures.

A daemon producer thread decodes ahead with a bounded buffer of at most
`depth` decoded items. On early consumer stop (generator close or garbage
collection), the stop flag is set and the producer uses timeout-based queue
operations so it can terminate promptly.

BaseException handling: if decode_fn raises a BaseException subclass such as
KeyboardInterrupt or SystemExit, it is not converted to FrameError and does
not propagate to the consumer. Instead, the producer thread terminates and
sets the stop flag, ending the feed.
"""

import queue
import threading


class FrameError(Exception):
    """Represents a decode failure for one input item."""

    def __init__(self, index, path, original):
        self.index = index
        self.path = path
        self.original = original
        super().__init__(index, path, original)


def frame_feed(items, decode_fn, depth=3):
    """Yield decoded items or FrameError instances in input order.

    The producer thread decodes ahead while the consumer computes. At most
    `depth` decoded-but-unconsumed results are buffered. If the consumer stops
    early, the producer is signalled to stop and will not block indefinitely.
    """
    if depth < 1:
        depth = 1

    q = queue.Queue(maxsize=depth)
    stop = threading.Event()
    sentinel = object()

    def wait_for_space():
        while not stop.is_set():
            if not q.full():
                return True
            stop.wait(0.05)
        return False

    def put_with_timeout(value):
        while not stop.is_set():
            try:
                q.put(value, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def producer():
        try:
            for index, item in enumerate(items):
                if stop.is_set():
                    return
                if not wait_for_space():
                    return

                try:
                    value = decode_fn(item)
                except Exception as exc:
                    value = FrameError(index, item, exc)

                if not put_with_timeout(value):
                    return

            if not stop.is_set() and wait_for_space():
                put_with_timeout(sentinel)
        except BaseException:
            stop.set()
            return

    thread = threading.Thread(
        target=producer,
        name="frame_feed-producer",
        daemon=True,
    )
    thread.start()

    try:
        while True:
            try:
                value = q.get(timeout=0.1)
            except queue.Empty:
                if stop.is_set():
                    break
                if not thread.is_alive() and q.empty():
                    break
                continue

            if value is sentinel:
                break

            yield value
    finally:
        stop.set()
