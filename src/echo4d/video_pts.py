"""Optional decoded-frame PTS sampling for RGB-only COLMAP video input.

Keep this module separate from the SfM solver. PTS is a time measurement, NOT
a physical-distance measurement or a substitute for sensor calibration.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

from .model import DataError


def decoded_frame_pts(raw: str) -> list[float]:
    """Decode bounded ffprobe output and reject ambiguous decoded-frame times."""
    if len(raw.encode('utf-8')) > 8_000_000:
        raise DataError('Decoded-frame PTS metadata exceeds 8MB')
    try:
        info = json.loads(raw)
        rows = info['frames']
        if not isinstance(rows, list) or not 3 <= len(rows) <= 100_000:
            raise DataError('Need 3-100000 decoded video frames for PTS mode')
        times = [float(row['best_effort_timestamp_time']) for row in rows]
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise DataError('Missing or invalid decoded-frame PTS metadata') from exc
    if not all(math.isfinite(t) and t >= 0 for t in times):
        raise DataError('Decoded-frame PTS must be finite and nonnegative')
    if any(b <= a for a, b in zip(times, times[1:])):
        raise DataError('Decoded-frame PTS must be strictly increasing')
    return times


def select_pts_frames(times: list[float], max_frames: int) -> tuple[int, list[int]]:
    """Deterministic decoded-frame stride; always respect the frame budget."""
    if not 3 <= max_frames <= 240:
        raise DataError('max_frames must be in [3,240]')
    if not 3 <= len(times) <= 100_000:
        raise DataError('Need 3-100000 decoded frames')
    step = math.ceil(len(times) / max_frames)
    indices = list(range(0, len(times), step))
    if len(indices) < 3:
        raise DataError('PTS sampling yielded fewer than 3 frames')
    return step, indices


def extract_source_pts(video: Path, image_dir: Path, max_frames: int,
                       receipt: dict, run_step: Callable) -> dict[str, float]:
    """Select decoded frames without CFR duplication or PTS resampling.

    This maps the *same* decoded indices from ffprobe to ffmpeg's select(n)
    filter, verifies all generated names, and returns relative source PTS
    under the existing frame_XXXXX.png -> seconds interface.
    """
    raw = run_step(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'frame=best_effort_timestamp_time', '-of',
         'json', str(video)], receipt, 'probe-decoded-frame-pts', timeout=240)
    times = decoded_frame_pts(raw)
    step, indices = select_pts_frames(times, max_frames)
    image_dir.mkdir(parents=True)
    run_step(
        ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
         '-i', str(video), '-map', '0:v:0', '-an', '-sn',
         '-vf', f'select=not(mod(n\,{step}))',
         '-fps_mode', 'passthrough', '-frames:v', str(len(indices)),
         '-start_number', '0', '-f', 'image2',
         str(image_dir / 'frame_%05d.png')],
        receipt, 'extract-frames-source-pts', timeout=240)
    expected = [f'frame_{i:05d}.png' for i in range(len(indices))]
    actual = sorted(p.name for p in image_dir.iterdir() if p.is_file())
    if actual != expected:
        raise DataError('Source PTS extraction frame count/names mismatch')
    start = times[0]
    sampled = {name: times[index] - start
               for name, index in zip(expected, indices)}
    receipt['source_pts_sampling'] = {
        'mode': 'source-pts',
        'decoded_frames': len(times),
        'stride': step,
        'sampled_frames': len(indices),
        'source_pts_first_seconds': start,
        'frame_map': [
            {'name': name, 'decoded_frame_index': index,
             'source_pts_seconds': times[index],
             'relative_pts_seconds': sampled[name]}
            for name, index in zip(expected, indices)],
    }
    return sampled
