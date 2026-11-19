"""Batch depth estimation utilities for a single-frame ONNX depth model."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import numpy as np
import onnxruntime as ort
from PIL import Image

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _preprocess_image(path):
    with Image.open(path) as img:
        img = img.convert("RGB")
        width, height = img.size
        resized = img.resize((518, 518), Image.BILINEAR)
        pixels = np.asarray(resized)

    tensor = np.array(pixels.transpose(2, 0, 1), dtype=np.float32, order="C")
    tensor /= 255.0
    tensor -= _MEAN
    tensor /= _STD
    return tensor, width, height


def _postprocess_frame(raw_frame, width, height):
    arr = np.asarray(raw_frame, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError("expected a 2-D depth map")
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    depth_img = Image.fromarray(arr, mode="F")
    resized = depth_img.resize((int(width), int(height)), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def _extract_frame_output(output, index, batch_len):
    arr = np.asarray(output)
    if batch_len == 1:
        if arr.ndim >= 1 and arr.shape[0] == 1:
            return arr[index]
        return arr
    if arr.ndim >= 1 and arr.shape[0] == batch_len:
        return arr[index]
    raise ValueError("unexpected batch output shape")


def _infer_prepared(tensor, width, height, session, timings=None):
    try:
        outputs = _run_ort(session, tensor[None], timings)
        frame = _extract_frame_output(outputs[0], 0, 1)
        return _postprocess_frame(frame, width, height)
    except Exception:
        return None


def _infer_single(path, session):
    try:
        prepared = _preprocess_image(path)
    except Exception:
        return None
    return _infer_prepared(*prepared, session)


def make_session(model_path, providers=None):
    if providers is not None:
        return ort.InferenceSession(str(model_path), providers=providers)
    return ort.InferenceSession(str(model_path))


# How many worker sessions a parallel depth run may open. One
# session per worker thread, each pinned to a single ORT thread: four
# default sessions in four threads fight over the same cores and run
# SLOWER than serial (measured 0.81x), while four 1-thread sessions
# run 1.34x and two run 1.52x on this 4-core VM (40 real frames, DAV2
# Small, outputs bit-identical to serial single-frame). The Base
# model is 389 MB, so it caps at 2 sessions instead of 4.
_LARGE_MODEL_BYTES = 200_000_000
_MIN_FRAMES_PARALLEL = 8


def parallel_workers(model_path, session, n_frames):
    """Worker count for the thread-parallel depth path, 0 = stay serial.

    Parallel only when the session really runs on CPU (a GPU session
    already parallelises inside the runtime; extra CPU sessions would
    only fight it), the machine has more than one core, and there are
    enough frames to pay back the per-session load (~0.5 s each).
    ECHO_DEPTH_WORKERS overrides the count (0 forces serial); the CPU
    gate still applies.
    """
    try:
        providers = session.get_providers()
    except Exception:
        return 0
    if not providers or providers[0] != "CPUExecutionProvider":
        return 0
    try:
        cores = os.cpu_count() or 1
    except Exception:
        cores = 1
    if cores < 2 or n_frames < _MIN_FRAMES_PARALLEL:
        return 0
    forced = os.environ.get("ECHO_DEPTH_WORKERS")
    if forced is not None:
        try:
            workers = int(forced)
        except ValueError:
            return 0
        return max(0, min(8, workers))
    try:
        big = os.path.getsize(model_path) >= _LARGE_MODEL_BYTES
    except Exception:
        big = False
    # Base's replica and activation workspace make the default two
    # one-thread sessions slower on the desktop reconstruction A/B.
    # An explicit override can still opt into multiple CPU sessions.
    return 0 if big else min(4, cores)


def _worker_session(model_path):
    """One ORT session pinned to a single thread, for one worker thread."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(model_path), opts,
                                providers=["CPUExecutionProvider"])


def iter_depth_estimates_parallel(image_paths, model_path, workers,
                                  chunk_size: int = 16):
    """Yield (index, depth_or_None) with one session per worker thread.

    Same per-frame math as _infer_single (bit-identical to serial
    single-frame; within the ~1e-4 epsilon the batched path already
    has against single-frame). Order is the input order. A failed
    frame yields None, exactly like the serial path, so the caller
    falls back per frame. Worked in chunks of chunk_size so at most
    one chunk of depth maps is ever in flight (the un-chunked serial
    path once held ~6.4 GB and was OOM-killed). Any session or pool
    failure raises out of the iteration and the caller falls back to
    the serial streaming path.
    """
    paths = list(image_paths)
    if not paths:
        return
    if workers < 1:
        raise ValueError("workers must be >= 1")
    local = threading.local()

    def _init():
        local.session = _worker_session(model_path)

    def _one(path):
        return _infer_single(path, local.session)

    with ThreadPoolExecutor(max_workers=workers,
                            initializer=_init) as ex:
        for start in range(0, len(paths), chunk_size):
            chunk = paths[start:start + chunk_size]
            for offset, depth in enumerate(ex.map(_one, chunk)):
                yield start + offset, depth


def _pixel_batch_dimension(session):
    for model_input in session.get_inputs():
        if model_input.name == "pixel_values" and model_input.shape:
            return model_input.shape[0]
    return None


def _prepare_group(paths):
    prepared = []
    for path in paths:
        try:
            prepared.append(_preprocess_image(path))
        except Exception:
            prepared.append(None)
    return prepared


def _run_ort(session, batch, timings):
    started = perf_counter()
    try:
        return session.run(["predicted_depth"], {"pixel_values": batch})
    finally:
        if timings is not None:
            timings["ort_run_calls"] = timings.get("ort_run_calls", 0) + 1
            timings["ort_run_seconds"] = timings.get("ort_run_seconds", 0) + perf_counter() - started


def _infer_group(prepared, session, dimension, batch_failed, timings):
    valid = [(i, item) for i, item in enumerate(prepared) if item is not None]
    fixed = isinstance(dimension, int) and dimension > 1
    if valid and (fixed or (not batch_failed and len(valid) == len(prepared) > 1)):
        tensors = [item[0] for _, item in valid]
        if fixed:
            tensors.extend([tensors[-1]] * (dimension - len(tensors)))
        try:
            batch = np.stack(tensors)
            outputs = _run_ort(session, batch, timings)
            raw = [_extract_frame_output(outputs[0], i, len(batch)) for i in range(len(valid))]
        except Exception:
            if fixed:
                raise
            # Retry with the already prepared inputs, and avoid hitting the
            # same batch error (for example GPU memory pressure) next time.
            batch_failed = True
        else:
            results = [None] * len(prepared)
            for frame, (index, (_, width, height)) in zip(raw, valid):
                try:
                    results[index] = _postprocess_frame(frame, width, height)
                except Exception:
                    pass
            return results, batch_failed
    results = [_infer_prepared(*item, session, timings) if item is not None else None
               for item in prepared]
    return results, batch_failed


def iter_depth_estimates(image_paths, session, batch_size: int = 4,
                         chunk_size: int = 16, *, prefetch: bool = False, timings=None):
    """Stream ordered depth maps with at most one prepared group ahead.

    chunk_size is retained for callers of the previous chunked iterator.
    A group now yields as soon as its own depths are ready, with no change
    to frame grouping. Prefetch only overlaps CPU preparation with the
    caller and ORT; it never runs two inferences or uploads at once.
    """
    paths = list(image_paths)
    if not paths:
        return
    dimension = _pixel_batch_dimension(session)
    size = max(1, batch_size)
    if dimension == 1:
        size = 1
    elif isinstance(dimension, int) and dimension > 1:
        size = dimension
    groups = iter(range(0, len(paths), size))
    start = next(groups)
    executor = ThreadPoolExecutor(max_workers=1) if prefetch else None
    batch_failed = False
    pending = None
    try:
        prep_start = perf_counter()
        prepared = _prepare_group(paths[start:start + size])
        if timings is not None:
            timings["cpu_input_wait_seconds"] = perf_counter() - prep_start
        while True:
            next_start = next(groups, None)
            if executor is not None and next_start is not None:
                pending = executor.submit(_prepare_group, paths[next_start:next_start + size])
            infer_started = perf_counter()
            depths, batch_failed = _infer_group(prepared, session, dimension, batch_failed, timings)
            if timings is not None:
                elapsed = perf_counter() - infer_started
                timings["group_inference_and_postprocess_seconds"] = (
                    timings.get("group_inference_and_postprocess_seconds", 0.0) + elapsed)
                timings["dynamic_batch_fell_back"] = batch_failed
            for offset, depth in enumerate(depths):
                yield start + offset, depth
            if next_start is None:
                return
            start = next_start
            prep_start = perf_counter()
            prepared = (pending.result() if pending is not None
                        else _prepare_group(paths[start:start + size]))
            if timings is not None:
                timings["cpu_input_wait_seconds"] += perf_counter() - prep_start
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


def estimate_depth_batch(image_paths, session, batch_size: int = 4) -> list:
    return [depth for _, depth in iter_depth_estimates(image_paths, session, batch_size)]
