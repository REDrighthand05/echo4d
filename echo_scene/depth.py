"""Monocular depth estimation with an ONNX Depth Anything V2 model."""

import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Union

import numpy as np
import onnxruntime as ort
from PIL import Image

_INPUT_NAME = "pixel_values"
_OUTPUT_NAME = "predicted_depth"
_INPUT_SIZE = 518

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

_SESSIONS: dict[tuple[str, tuple[str, ...]], ort.InferenceSession] = {}

# The active execution provider, recorded on first session creation
# ("CPUExecutionProvider" until a GPU runtime is installed).
ACTIVE_PROVIDER: str | None = None


def _provider_chain() -> list:
    """Best-first providers the installed onnxruntime actually has.

    CUDA first when installed, then DirectML, then CPU.
    ECHO_DEPTH_PROVIDER selects an available provider for comparison runs.
    """
    available = list(ort.get_available_providers())
    forced = os.environ.get("ECHO_DEPTH_PROVIDER")
    if forced and forced in available:
        return [forced]
    chain = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider")
             if p in available]
    chain.append("CPUExecutionProvider")
    return chain


_GPU_PROVIDERS = {"CUDAExecutionProvider", "DmlExecutionProvider"}


def depth_runtime_inventory():
    tool = shutil.which("nvidia-smi")
    names = []
    if tool:
        try:
            result = subprocess.run(
                [tool, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, errors="replace", timeout=3,
                check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        except (OSError, subprocess.SubprocessError):
            pass
    cards = ", ".join(names) if names else "未由 nvidia-smi 确认"
    providers = ", ".join(ort.get_available_providers())
    return [f"NVIDIA 显卡检测：{cards}。", f"已安装的深度运行时 provider：{providers}。"]


def _get_session(model_path: Union[str, os.PathLike], *, providers=None,
                 profile_prefix=None) -> ort.InferenceSession:
    global ACTIVE_PROVIDER
    model = os.fspath(model_path)
    chain = _provider_chain() if providers is None else providers
    key = (model, tuple(chain))
    session = _SESSIONS.get(key) if profile_prefix is None else None
    if session is None:
        if not os.path.isfile(model):
            raise FileNotFoundError(f"ONNX model file not found: {model}")
        options = ort.SessionOptions()
        if "DmlExecutionProvider" in chain:
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if profile_prefix is not None:
            options.enable_profiling = True
            options.profile_file_prefix = os.fspath(profile_prefix)
        try:
            session = ort.InferenceSession(model, options, providers=chain)
        except (RuntimeError, ValueError):
            if not _GPU_PROVIDERS.intersection(chain):
                raise
            return _get_session(model, providers=["CPUExecutionProvider"],
                                profile_prefix=profile_prefix)
        ACTIVE_PROVIDER = session.get_providers()[0]
        if profile_prefix is None:
            _SESSIONS[key] = session
    return session


def end_provider_profile(session, *, keep_file=False):
    """Read operator execution events, excluding data transfers."""
    path = None
    try:
        filename = session.end_profiling()
        if not filename:
            return {}
        path = Path(filename)
        events = json.loads(path.read_text(encoding="utf-8"))
        counts = Counter()
        for event in events:
            args = event.get("args", {})
            if (event.get("cat") == "Node"
                    and event.get("name", "").endswith("_kernel_time")
                    and args.get("op_name")
                    and not str(args["op_name"]).lower().startswith("memcpy")
                    and args.get("provider")):
                counts[args["provider"]] += 1
        return dict(counts)
    except (OSError, ValueError, RuntimeError):
        return {}
    finally:
        if path is not None and not keep_file:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def estimate_depth(
    image_path: Union[str, os.PathLike],
    model_path: Union[str, os.PathLike],
    *, session=None,
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

    if session is None:
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
