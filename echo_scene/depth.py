"""Monocular depth estimation with an ONNX Depth Anything V2 model."""

import os
from pathlib import Path
from typing import Dict, Union

import numpy as np
import onnxruntime as ort
from PIL import Image

_INPUT_NAME = "pixel_values"
_OUTPUT_NAME = "predicted_depth"
_INPUT_SIZE = 518

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

_SESSIONS: Dict[str, ort.InferenceSession] = {}


def _get_session(model_path: Union[str, os.PathLike]) -> ort.InferenceSession:
    key = os.fspath(model_path)
    session = _SESSIONS.get(key)
    if session is None:
        if not os.path.isfile(key):
            raise FileNotFoundError(f"ONNX model file not found: {key}")
        session = ort.InferenceSession(
            key,
            providers=["CPUExecutionProvider"],
        )
        _SESSIONS[key] = session
    return session


def estimate_depth(
    image_path: Union[str, os.PathLike],
    model_path: Union[str, os.PathLike],
) -> np.ndarray:
    """Estimate relative inverse depth for an image using an ONNX model.

    The returned array has the original image height and width. Larger values
    indicate surfaces nearer to the camera. Values are not converted to meters.
    """
    image_path_str = os.fspath(image_path)
    if not os.path.isfile(image_path_str):
        raise FileNotFoundError(f"Image file not found: {image_path_str}")

    with Image.open(image_path_str) as img:
        rgb = img.convert("RGB")
        original_width, original_height = rgb.size
        resized = rgb.resize(
            (_INPUT_SIZE, _INPUT_SIZE),
            resample=Image.BILINEAR,
        )
        pixels = np.asarray(resized, dtype=np.float32) / 255.0

    tensor = np.transpose(pixels, (2, 0, 1))[None, ...]
    tensor = (tensor - _MEAN) / _STD
    tensor = np.ascontiguousarray(tensor, dtype=np.float32)

    session = _get_session(model_path)
    outputs = session.run([_OUTPUT_NAME], {_INPUT_NAME: tensor})
    depth = np.asarray(outputs[0], dtype=np.float32)

    if depth.ndim == 3:
        depth = depth[0]
    elif depth.ndim > 3:
        depth = depth.reshape(depth.shape[-2], depth.shape[-1])

    depth_image = Image.fromarray(np.ascontiguousarray(depth), mode="F")
    depth_resized = depth_image.resize(
        (original_width, original_height),
        resample=Image.BILINEAR,
    )

    return np.asarray(depth_resized, dtype=np.float32)


def save_depth_preview(
    depth: np.ndarray,
    out_png: Union[str, os.PathLike],
) -> None:
    """Save a normalized 8-bit grayscale preview of a depth array."""
    depth_array = np.asarray(depth, dtype=np.float32)

    low, high = np.percentile(depth_array, [2.0, 98.0])
    if high > low:
        normalized = (depth_array - low) / (high - low)
    else:
        normalized = np.zeros_like(depth_array, dtype=np.float32)

    display = np.clip(normalized, 0.0, 1.0) * 255.0
    display = display.astype(np.uint8)

    Image.fromarray(display, mode="L").save(os.fspath(out_png), format="PNG")
