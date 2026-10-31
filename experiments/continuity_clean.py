"""Continuity cleanup: keep the connected body, drop isolated points.

The user's idea (2026-10-31): a point cloud file itself has no notion of
continuity -- points are just coordinates. Give every point an entity
volume (here: the voxel cell it falls in), declare two points "in
contact" when their occupied voxels are 26-neighbours (face/edge/corner),
and keep only the largest connected body. Enlarge the cells until the
main body covers a chosen fraction q of all points -- q is the
STRICTNESS parameter (default 0.80): the kept set is the largest
connected component at the smallest cell size whose coverage reaches q;
everything outside that body is dropped as discontinuous noise.

Deterministic selection only: points are kept or dropped, never moved
or averaged (same family as the corroboration trim).

Usage:
    python experiments/continuity_clean.py <job_dir> <out_dir> [--strictness 0.80]

Reads <job_dir>/points.bin (15 bytes/point: float32 xyz + uint8 rgb).
Writes into <out_dir>:
    points_q90.bin / points_q80.bin / points_q70.bin  (same 15-byte format)
    continuity_summary.json  (sweep table + per-variant stats)
The headline variant is the one at --strictness; all three are written.
--fine-below S adds a second, denser sweep over the percolation zone
(s <= S) so the q-crossing cell size is located sharply.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

# 13 forward neighbour offsets: the other 13 are their negatives.
OFFSETS = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) > (0, 0, 0)
]


def load_points(path: Path):
    raw = np.fromfile(path, dtype=np.uint8)
    n = len(raw) // 15
    raw = raw[: n * 15].reshape(n, 15)
    pts = raw[:, :12].copy().view("<f4").reshape(n, 3).astype(np.float64)
    cols = raw[:, 12:15].copy()
    return pts, cols


def save_points(path: Path, pts_f32_bytes_view, cols, keep_mask):
    """Write kept points in the original 15-byte layout, original order."""
    n = keep_mask.shape[0]
    out = np.empty((n, 15), dtype=np.uint8)
    out[:, :12] = pts_f32_bytes_view
    out[:, 12:15] = cols
    out[keep_mask].tofile(path)


def components_at(pts: np.ndarray, s: float):
    """Connected components of occupied voxels (26-connectivity).

    Returns (labels per point, sizes per component sorted desc,
    n_occupied_voxels)."""
    vox = np.floor(pts / s).astype(np.int64)
    vox -= vox.min(axis=0)  # keep packed keys small/non-negative
    dims = vox.max(axis=0) + 1
    key = (vox[:, 0] * dims[1] + vox[:, 1]) * dims[2] + vox[:, 2]
    uniq, inverse = np.unique(key, return_inverse=True)
    nv = len(uniq)
    # Neighbour lookup: for each offset, shift keys and searchsorted.
    rows, cols_idx = [], []
    base = uniq  # sorted by construction of np.unique
    for dx, dy, dz in OFFSETS:
        shifted = uniq + (dx * dims[1] + dy) * dims[2] + dz
        pos = np.searchsorted(base, shifted)
        pos_c = np.minimum(pos, nv - 1)
        hit = base[pos_c] == shifted
        # guard against key wrap-around across rows/planes: verify via coords
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
    return plab, sizes[order], nv, order


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


def sweep(pts: np.ndarray, n_steps: int = 18, fine_below: float | None = None):
    extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    sizes = list(extent * np.geomspace(1 / 1500, 1 / 25, n_steps))
    if fine_below is not None:
        sizes += list(np.geomspace(sizes[0], fine_below, 46))
    sizes = sorted(set(float(s) for s in sizes))
    table = []
    cache = {}
    for s in sizes:
        plab, comp_sizes, nv, order = components_at(pts, float(s))
        cov = comp_sizes[0] / len(pts)
        table.append(
            {
                "s": float(s),
                "occupied_voxels": int(nv),
                "n_components": int(len(comp_sizes)),
                "largest_points": int(comp_sizes[0]),
                "coverage": float(cov),
                "second_points": int(comp_sizes[1]) if len(comp_sizes) > 1 else 0,
            }
        )
        cache[float(s)] = (plab, order)
        print(
            f"s={s:.4f} vox={nv:7d} comps={len(comp_sizes):6d} "
            f"largest={comp_sizes[0]:7d} coverage={cov:.4f}"
        )
    return table, cache


def pick_for_q(table, q):
    """Smallest s whose largest-component coverage >= q (else the max)."""
    for row in table:
        if row["coverage"] >= q:
            return row
    return table[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--strictness", type=float, default=0.80)
    ap.add_argument(
        "--fine-below",
        type=float,
        default=None,
        help="add a dense second sweep for cell sizes up to this value",
    )
    args = ap.parse_args()

    job = Path(args.job_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = np.fromfile(job / "points.bin", dtype=np.uint8)
    pts, cols = load_points(job / "points.bin")
    n = len(pts)
    f32_bytes = raw[: n * 15].reshape(n, 15)[:, :12].copy()
    print(f"{n} points from {job}/points.bin")

    table, cache = sweep(pts, fine_below=args.fine_below)
    summary = {"n_points": n, "sweep": table, "variants": {}}
    for q in (0.90, 0.80, 0.70):
        row = pick_for_q(table, q)
        plab, order = cache[row["s"]]
        keep = plab == order[0]
        tag = f"q{int(round(q * 100))}"
        save_points(out / f"points_{tag}.bin", f32_bytes, cols, keep)
        summary["variants"][tag] = {
            "q": q,
            "s": row["s"],
            "kept": int(keep.sum()),
            "kept_pct": float(keep.mean() * 100),
            "dropped": int((~keep).sum()),
            "second_component_points": row["second_points"],
        }
        print(
            f"{tag}: s={row['s']:.4f} kept={int(keep.sum())} "
            f"({keep.mean() * 100:.2f}%) dropped={int((~keep).sum())}"
        )
    summary["headline_strictness"] = args.strictness
    (out / "continuity_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"summary -> {out}/continuity_summary.json")


if __name__ == "__main__":
    main()
