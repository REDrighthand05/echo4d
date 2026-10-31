"""Frame quality selection for a video-to-scene pipeline."""

from __future__ import annotations

import pathlib
from typing import List, Union

import numpy as np
from PIL import Image


def frame_sharpness(image_path: Union[str, pathlib.Path]) -> float:
    """Return the variance of the discrete Laplacian of a grayscale image."""
    with Image.open(image_path) as img:
        gray = img.convert("L")
        gray.thumbnail((640, 640))
        arr = np.asarray(gray, dtype=np.float64)

    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 3:
        return 0.0

    center = arr[1:-1, 1:-1]
    up = arr[:-2, 1:-1]
    down = arr[2:, 1:-1]
    left = arr[1:-1, :-2]
    right = arr[1:-1, 2:]

    laplacian = (-4.0 * center) + up + down + left + right
    return float(np.var(laplacian))


def select_frames(
    frames_dir: Union[str, pathlib.Path],
    max_frames: int,
    min_sharpness: float,
) -> List[pathlib.Path]:
    """Select sharp frames from a directory of ordered JPG frames."""
    directory = pathlib.Path(frames_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {directory}")

    frames = sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"
    )
    if not frames:
        return []

    scores = [frame_sharpness(p) for p in frames]

    kept = [p for p, s in zip(frames, scores) if s >= min_sharpness]
    if not kept:
        best_idx = max(range(len(frames)), key=lambda i: scores[i])
        kept = [frames[best_idx]]

    if max_frames <= 0:
        return []
    if max_frames == 1:
        return [kept[0]]
    if len(kept) <= max_frames:
        return kept

    n = len(kept)
    k = max_frames
    indices = [0]
    for i in range(1, k - 1):
        idx = int(round(i * (n - 1) / (k - 1)))
        indices.append(idx)
    indices.append(n - 1)

    return [kept[i] for i in indices]
