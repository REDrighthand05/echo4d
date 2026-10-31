"""Continuity cleanup: keep the connected main body of a fused cloud.

Promoted from experiments/continuity_clean.py (validated 2026-10-31 on
the room acceptance cloud: 208,666 -> 183,691 points at strictness 0.80;
the 24,975 removed points were 5,877 tiny fragments, median 2 points,
floating off the room's real surfaces; the room and furniture stayed
intact).

The rule is a pure keep/drop selection -- the user's own idea: give
every point the volume of its grid cell; points whose cells touch
(26-neighbourhood) form one body; only the largest body survives. No
point is moved, blended or averaged, and the outcome is deterministic.

The cell size is not fixed: it is swept over a deterministic candidate
ladder and the first size whose largest body covers at least
``strictness`` of the cloud wins (the same pick rule the experiment
validated). The ladder mirrors the experiment candidate for candidate:
a coarse geometric ladder over the cloud extent, plus a fine ladder
across the percolation jump zone. The fine ladder's upper bound is a
fixed fraction of the extent, calibrated so that on the room validation
cloud the production ladder is identical to the experiment's
(``--fine-below 0.095`` at that cloud's extent).
"""
import json
from pathlib import Path

import numpy as np

__all__ = ["DEFAULT_STRICTNESS", "clean_points", "stage_clean"]

# The user's chosen strictness (keep-target 80%): the single knob of
# this stage, in one place.
DEFAULT_STRICTNESS = 0.80

# Candidate ladder (fractions of the cloud extent / grid geometry).
_COARSE_LO_FRAC = 1.0 / 1500.0
_COARSE_HI_FRAC = 1.0 / 25.0
_COARSE_STEPS = 18
_FINE_STEPS = 46
_FINE_BELOW_FRACTION = 0.004060450244401552  # of extent; see docstring

_MIN_POINTS = 1000  # below this there is no body to speak of; skip

# 13 forward neighbour offsets: the other 13 are their negatives.
_OFFSETS_26 = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) > (0, 0, 0)
]


def _same_row(uniq, shifted, dims, dx, dy, dz):
    """Reject shifted keys that wrapped around a row/plane boundary."""
    z = uniq % dims[2]
    y = (uniq // dims[2]) % dims[1]
    x = uniq // (dims[1] * dims[2])
    nz = z + dz
    ny = y + dy
    nx = x + dx
    return (
        (nz >= 0) & (nz < dims[2]) & (ny >= 0) & (ny < dims[1]) & (nx >= 0) & (nx < dims[0])
    )


def _components_at(points: np.ndarray, s: float):
    """Connected components of the occupied cells (26-connectivity).

    Verbatim the validated experiment's construction: cells are
    floor(points / s) re-based at their minimum, packed into one int64
    key; neighbours are probed by shifting the packed key, with the
    _same_row guard rejecting row/plane wrap-arounds. Returns (labels
    per point, component point-counts sorted descending, label order).
    """
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    vox = np.floor(points / s).astype(np.int64)
    vox -= vox.min(axis=0)  # keep packed keys small/non-negative
    dims = vox.max(axis=0) + 1
    key = (vox[:, 0] * dims[1] + vox[:, 1]) * dims[2] + vox[:, 2]
    uniq, inverse = np.unique(key, return_inverse=True)
    nv = len(uniq)
    rows, cols_idx = [], []
    base = uniq  # sorted by construction of np.unique
    for dx, dy, dz in _OFFSETS_26:
        shifted = uniq + (dx * dims[1] + dy) * dims[2] + dz
        pos = np.searchsorted(base, shifted)
        pos_c = np.minimum(pos, nv - 1)
        hit = base[pos_c] == shifted
        hit &= _same_row(uniq, shifted, dims, dx, dy, dz)
        src = np.nonzero(hit)[0]
        rows.append(src)
        cols_idx.append(pos_c[hit])
    if rows:
        r = np.concatenate(rows)
        c = np.concatenate(cols_idx)
        graph = sparse.coo_matrix(
            (np.ones(len(r), dtype=np.int8), (r, c)), shape=(nv, nv)
        )
    else:
        graph = sparse.coo_matrix((nv, nv), dtype=np.int8)
    _, vlab = connected_components(graph, directed=False)
    plab = vlab[inverse]
    sizes = np.bincount(plab, minlength=plab.max() + 1)
    order = np.argsort(-sizes, kind="stable")
    return plab, sizes[order], order


def _candidate_sizes(extent: float) -> np.ndarray:
    sizes = extent * np.geomspace(_COARSE_LO_FRAC, _COARSE_HI_FRAC, _COARSE_STEPS)
    fine_below = extent * _FINE_BELOW_FRACTION
    if fine_below > sizes[0]:
        fine = np.geomspace(sizes[0], fine_below, _FINE_STEPS)
        sizes = np.unique(np.concatenate([sizes, fine]))
    return sizes


def clean_points(points: np.ndarray, strictness: float = DEFAULT_STRICTNESS):
    """Largest-connected-body keep mask for a point cloud.

    Sweeps the candidate cell sizes ascending and stops at the first
    size whose largest body covers at least ``strictness`` of the
    cloud -- the experiment's pick rule. If no candidate reaches the
    target, the largest candidate (the most-merged reading) is used.
    Returns (keep_mask, stats).
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=bool), {"points": 0}
    extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    chosen = None
    for s in _candidate_sizes(extent):
        labels, comp_sizes, order = _components_at(pts, float(s))
        coverage = float(comp_sizes[0]) / n
        chosen = (float(s), labels == order[0], coverage, int(len(comp_sizes)))
        if coverage >= strictness:
            break
    s, keep, coverage, n_components = chosen
    stats = {
        "cell_size": s,
        "coverage": coverage,
        "components": n_components,
        "points_before": int(n),
        "points_after": int(keep.sum()),
        "points_dropped": int(n - keep.sum()),
        "extent": extent,
    }
    return keep, stats


def stage_clean(job_dir, strictness: float = DEFAULT_STRICTNESS) -> dict:
    """Filter the job's fused cloud down to its connected main body.

    Reads points.bin (and points_frame_ids.npy when present), rewrites
    points.bin / scene_points.ply / the ids with the kept points only,
    so the mesh and voxel stages downstream consume the cleaned cloud.
    """
    from .fuse import _write_binary_ply, _write_points_bin

    job_dir = Path(job_dir)
    points_path = job_dir / "points.bin"
    if not points_path.exists():
        info = {"stage": "clean", "ok": False, "reason": "points.bin missing"}
        (job_dir / "job.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
        return info
    raw = np.fromfile(points_path, dtype=np.uint8)
    if raw.size % 15:
        info = {"stage": "clean", "ok": False, "reason": "points.bin malformed"}
        (job_dir / "job.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
        return info
    raw = raw.reshape(-1, 15)
    points = raw[:, :12].copy().view(np.float32).reshape(-1, 3).astype(np.float64)
    colors = raw[:, 12:15].copy()

    ids_path = job_dir / "points_frame_ids.npy"
    frame_ids = None
    if ids_path.exists():
        loaded = np.load(ids_path)
        if len(loaded) == len(points):
            frame_ids = loaded

    if len(points) < _MIN_POINTS:
        info = {
            "stage": "clean", "ok": True, "skipped": "too few points",
            "points": int(len(points)), "strictness": strictness,
        }
        (job_dir / "job.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
        return info

    keep, stats = clean_points(points, strictness)
    if not keep.all():
        _write_points_bin(points[keep], colors[keep], points_path)
        _write_binary_ply(points[keep], colors[keep], job_dir / "scene_points.ply")
        if frame_ids is not None:
            np.save(ids_path, frame_ids[keep].astype(np.int32))
    info = {
        "stage": "clean",
        "ok": True,
        "strictness": strictness,
        "cell_size": stats["cell_size"],
        "coverage": stats["coverage"],
        "components": stats["components"],
        "points_before": stats["points_before"],
        "points_after": stats["points_after"],
        "points_dropped": stats["points_dropped"],
    }
    (job_dir / "job.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return info
