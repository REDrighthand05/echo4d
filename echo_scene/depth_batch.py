"""Batch depth estimation utilities for a single-frame ONNX depth model."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
from PIL import Image


def _preprocess_image(path):
    with Image.open(path) as img:
        img = img.convert("RGB")
        width, height = img.size
        resized = img.resize((518, 518), Image.BILINEAR)
        pixels = np.asarray(resized, dtype=np.float32) / 255.0

    tensor = pixels.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    tensor = (tensor - mean) / std
    return np.ascontiguousarray(tensor, dtype=np.float32), width, height


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


def _infer_single(path, session):
    try:
        tensor, width, height = _preprocess_image(path)
        input_array = np.ascontiguousarray(tensor[np.newaxis, :, :, :], dtype=np.float32)
        outputs = session.run(["predicted_depth"], {"pixel_values": input_array})
        frame = _extract_frame_output(outputs[0], 0, 1)
        return _postprocess_frame(frame, width, height)
    except Exception:
        return None


def make_session(model_path, providers=None):
    if providers is None:
        try:
            from echo_engine.compute_policy import select_providers
            providers = select_providers()
        except Exception:
            providers = list(ort.get_available_providers())
    return ort.InferenceSession(str(model_path), providers=providers)


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
    return 2 if big else min(4, cores)


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


def iter_depth_estimates(image_paths, session, batch_size: int = 4,
                         chunk_size: int = 16):
    """Yield (index, depth_or_None) streaming over bounded chunks.

    estimate_depth_batch holds every result until it returns; at full
    frame resolution one depth map is ~8MB, so a job with several
    hundred registered frames kept ~6.4GB resident and was OOM-killed
    before writing anything (room2 recheck, 2026-11-04). This walks
    the list in chunks and yields each frame as its chunk finishes,
    so the caller can save it and let it go. chunk_size is rounded
    down to a whole number of batches, so the batch groupings -- and
    therefore every result -- are exactly the ones a single
    estimate_depth_batch call over the whole list would produce.
    """
    paths = list(image_paths)
    if batch_size > 1:
        chunk_size = max(batch_size,
                         chunk_size // batch_size * batch_size)
    for start in range(0, len(paths), chunk_size):
        chunk = paths[start:start + chunk_size]
        depths = estimate_depth_batch(chunk, session,
                                      batch_size=batch_size)
        for offset, depth in enumerate(depths):
            yield start + offset, depth
        del depths


def estimate_depth_batch(image_paths, session, batch_size: int = 4) -> list:
    paths = list(image_paths)
    if not paths:
        return []

    if batch_size <= 1:
        return [_infer_single(path, session) for path in paths]

    results = [None] * len(paths)

    for start in range(0, len(paths), batch_size):
        chunk_paths = paths[start:start + batch_size]
        chunk_tensors = []
        chunk_meta = []
        chunk_indices = []
        preprocess_ok = True

        for offset, path in enumerate(chunk_paths):
            try:
                tensor, width, height = _preprocess_image(path)
            except Exception:
                preprocess_ok = False
                break
            chunk_tensors.append(tensor)
            chunk_meta.append((width, height))
            chunk_indices.append(start + offset)

        if not preprocess_ok:
            for offset, path in enumerate(chunk_paths):
                results[start + offset] = _infer_single(path, session)
            continue

        try:
            batch = np.ascontiguousarray(np.stack(chunk_tensors, axis=0), dtype=np.float32)
            outputs = session.run(["predicted_depth"], {"pixel_values": batch})
            batch_output = outputs[0]
            raw_frames = [
                _extract_frame_output(batch_output, i, len(chunk_tensors))
                for i in range(len(chunk_tensors))
            ]
        except Exception:
            for offset, path in enumerate(chunk_paths):
                results[start + offset] = _infer_single(path, session)
            continue

        for i, (width, height) in enumerate(chunk_meta):
            idx = chunk_indices[i]
            try:
                results[idx] = _postprocess_frame(raw_frames[i], width, height)
            except Exception:
                results[idx] = None

    return results
