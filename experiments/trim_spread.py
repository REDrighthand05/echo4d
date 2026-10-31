"""Spread yardstick applied to a DELIVERED cloud (points.bin + frame ids).

experiments/measure_spread.py re-derives points from depth maps, so it
cannot see point selection. This applies the identical statistic to the
final fused cloud: voxelize at the yardstick voxel size (extent/110,
origin at the per-axis 1st percentile, same hash as measure_spread),
find voxels containing >= 2 distinct frames, and report the RMS distance
of their points to the voxel centroid.

Usage: python experiments/trim_spread.py <points.bin> <frame_ids.npy> <voxel>
"""
import json
import sys
from pathlib import Path

import numpy as np


def load_cloud(path: Path):
    raw = np.fromfile(path, dtype=np.uint8)
    n = len(raw) // 15
    raw = raw[: n * 15].reshape(n, 15)
    pts = raw[:, :12].copy().view("<f4").reshape(n, 3).astype(np.float64)
    return pts


def delivered_spread(points: np.ndarray, fids: np.ndarray, voxel: float) -> dict:
    lo = np.percentile(points, 1, axis=0)
    keys = np.floor((points - lo) / voxel).astype(np.int64)
    lin = (keys[:, 0] * 73856093) ^ (keys[:, 1] * 19349663) ^ (keys[:, 2] * 83492791)
    vox_id, inverse = np.unique(lin, return_inverse=True)
    counts = np.bincount(inverse)
    n_frames_code = 1024
    pair_code = inverse.astype(np.int64) * n_frames_code + fids.astype(np.int64)
    uniq_pairs = np.unique(pair_code)
    distinct = np.bincount(uniq_pairs // n_frames_code, minlength=len(vox_id))
    multi = distinct >= 2
    in_multi = multi[inverse]
    cent = np.zeros((len(vox_id), 3))
    for k in range(3):
        cent[:, k] = np.bincount(inverse, weights=points[:, k]) / counts
    d = points[in_multi] - cent[inverse[in_multi]]
    spread = float(np.sqrt(np.mean(np.sum(d * d, axis=1)))) if in_multi.any() else float("nan")
    return {
        "points": int(len(points)),
        "voxels": int(len(vox_id)),
        "cross_frame_voxels": int(multi.sum()),
        "points_in_cross_frame_voxels": int(in_multi.sum()),
        "delivered_spread_rms": spread,
    }


if __name__ == "__main__":
    pts = load_cloud(Path(sys.argv[1]))
    fids = np.load(sys.argv[2])
    assert len(fids) == len(pts), (len(fids), len(pts))
    result = delivered_spread(pts, fids, float(sys.argv[3]))
    result["points_bin"] = sys.argv[1]
    print(json.dumps(result, indent=2))
