"""Plan desk source identities from timestamp indexes, never measured pixels or GT.

V1 is the unchanged first 75 stride-eight RGB frames. V2 is a NEW experiment:
filter the original RGB index by the original 30 ms depth window, then take
every eighth eligible entry, up to exactly 75 unique source associations.
No pose, measured pixel value or score selects a frame.
"""
from __future__ import annotations

import math
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np

from echo4d.model import DataError, file_sha256
from echo4d.tum import nearest_timestamp, read_rows
from tum_desk_sfm import FRAME_COUNT, ROOT, SOURCE_STRIDE, validate_archive_name

MAX_GAP_S = .03
MODES = ('v1-frozen', 'v2-source-paired')


def extract_time_indexes(archive: Path, output: Path) -> dict:
    """Extract only original rgb.txt/depth.txt, using bounded safe TAR handling."""
    output.mkdir(parents=True, exist_ok=False)
    wanted = {'rgb.txt', 'depth.txt'}
    seen = set()
    count = total = 0
    with tarfile.open(archive, 'r:gz') as source:
        for member in source:
            parts = validate_archive_name(member.name, 'dir' if member.isdir()
                                          else 'file' if member.isfile() else 'other',
                                          member.size)
            if member.isdir():
                continue
            count += 1
            total += member.size
            if count > 25000 or total > 3 * 1024**3:
                raise DataError('Desk index scan exceeds archive resource bounds')
            if len(parts) != 2 or parts[1] not in wanted:
                continue
            name = parts[1]
            if name in seen or member.size > 1024**2:
                raise DataError('Duplicate or oversized original desk index')
            stream = source.extractfile(member)
            if stream is None:
                raise DataError('Missing original desk index bytes')
            with stream:
                raw = stream.read(1024**2 + 1)
            if len(raw) != member.size:
                raise DataError('Truncated original desk index')
            (output / name).write_bytes(raw)
            seen.add(name)
    if seen != wanted:
        raise DataError('Original archive must contain RGB and depth time indexes')
    return {name: file_sha256(output / name) for name in sorted(wanted)}


def _index(path: Path, kind: str) -> list[tuple[float, str]]:
    rows = read_rows(path, 2)
    result = []
    seen_paths = set()
    previous = -math.inf
    for stamp, name in rows:
        t = float(stamp)
        p = PurePosixPath(name)
        if (not math.isfinite(t) or t <= previous or p.is_absolute()
                or len(p.parts) != 2 or p.parts[0] != kind
                or '..' in p.parts or '\\' in name or p.suffix.lower() != '.png'
                or name in seen_paths):
            raise DataError(f'Invalid or reused {kind} source identity/time')
        result.append((t, name))
        seen_paths.add(name)
        previous = t
    if not result:
        raise DataError(f'Empty {kind} source time index')
    return result


def plan_cohort(index_root: Path, mode: str) -> dict:
    if mode not in MODES:
        raise DataError('Unknown desk source cohort protocol')
    rgb = _index(index_root / 'rgb.txt', 'rgb')
    depth = _index(index_root / 'depth.txt', 'depth')
    times = np.array([row[0] for row in depth], dtype=np.float64)
    selected = []
    rejected = []
    used = set()
    candidates = list(range(0, len(rgb), SOURCE_STRIDE))
    if mode == 'v2-source-paired':
        eligible = []
        for rgb_index, (timestamp, rgb_path) in enumerate(rgb):
            if nearest_timestamp(times, timestamp, MAX_GAP_S) is None:
                rejected.append({'source_rgb_index': rgb_index, 'rgb': rgb_path,
                                 'original_rgb_time_s': timestamp,
                                 'reason': 'no_depth_within_30ms'})
            else:
                eligible.append(rgb_index)
        candidates = eligible[::SOURCE_STRIDE]
    for rgb_index in candidates:
        timestamp, rgb_path = rgb[rgb_index]
        at = nearest_timestamp(times, timestamp, MAX_GAP_S)
        reason = 'no_depth_within_30ms' if at is None else (
            'reused_depth_source' if depth[at][1] in used else None)
        if mode == 'v2-source-paired' and reason:
            rejected.append({'source_rgb_index': rgb_index, 'rgb': rgb_path,
                             'original_rgb_time_s': timestamp, 'reason': reason})
            continue
        item = {'name': f'frame_{len(selected):05d}.png',
                'source_rgb_index': rgb_index, 'original_tum_path': rgb_path,
                't_tum_rgb_s': timestamp,
                'source_depth_path': depth[at][1] if at is not None else None,
                'source_depth_time_s': depth[at][0] if at is not None else None,
                'timestamp_delta_seconds': float(times[at] - timestamp)
                if at is not None else None}
        selected.append(item)
        if at is not None:
            used.add(depth[at][1])
        if len(selected) == FRAME_COUNT:
            break
    return {'version': 'desk-source-cohort-plan-1', 'mode': mode,
            'selection_complete': len(selected) == FRAME_COUNT,
            'source_stride': SOURCE_STRIDE, 'required_frames': FRAME_COUNT,
            'stride_domain': 'eligible_rgb_index' if mode == 'v2-source-paired' else 'original_rgb_index',
            'max_rgb_depth_gap_s': MAX_GAP_S,
            'source_index_sha256': {name: file_sha256(index_root / name)
                                    for name in ('rgb.txt', 'depth.txt')},
            'selection_inputs': 'original index timestamps and source identities only',
            'depth_pixels_or_GT_read_for_selection': False,
            'selected_frames': selected, 'rejected_before_selection': rejected}
