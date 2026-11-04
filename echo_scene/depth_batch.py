"""Batch depth estimation utilities for a single-frame ONNX depth model."""

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
    if providers is not None:
        return ort.InferenceSession(str(model_path), providers=providers)
    return ort.InferenceSession(str(model_path))


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
