"""Camera-up-aligned top views of fused clouds, same framing for all jobs.

World up = mean camera -Y axis from poses.json. Both clouds are
projected onto the plane perpendicular to their up axis and rasterized
with a shared scale (pixels per scene unit) so the pair is comparable.

Usage: python experiments/render_top.py <job_dir> <out.png> [<job_dir> <out.png> ...]
Framing/scale are computed jointly over all given jobs.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echo_scene.fuse import _quat_xyzw_to_matrix  # noqa: E402

SIZE = 1100
MARGIN = 30


def load_cloud(job: Path):
    raw = np.fromfile(job / "points.bin", dtype=np.uint8)
    n = len(raw) // 15
    raw = raw[: n * 15].reshape(n, 15)
    pts = raw[:, :12].copy().view("<f4").reshape(n, 3).astype(np.float64)
    cols = raw[:, 12:15].astype(np.int32)
    return pts, cols


def project(job: Path, pts: np.ndarray):
    poses = json.loads((job / "poses.json").read_text())
    ups = []
    for p in poses.values():
        rot = _quat_xyzw_to_matrix(p["cam_from_world_quat_xyzw"])
        ups.append(rot.T @ np.array([0.0, -1.0, 0.0]))
    up = np.mean(ups, axis=0)
    up /= np.linalg.norm(up)
    a = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    e1 = np.cross(up, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    x = pts @ e1
    y = pts @ e2
    h = pts @ up
    return x, y, h


def main() -> None:
    args = sys.argv[1:]
    jobs = [(Path(args[i]), args[i + 1]) for i in range(0, len(args), 2)]
    projected = []
    all_x, all_y = [], []
    for job, _ in jobs:
        pts, cols = load_cloud(job)
        x, y, h = project(job, pts)
        projected.append((x, y, h, cols))
        all_x.append(x)
        all_y.append(y)
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    cx, cy = (all_x.min() + all_x.max()) / 2, (all_y.min() + all_y.max()) / 2
    span = max(all_x.max() - all_x.min(), all_y.max() - all_y.min())
    scale = (SIZE - 2 * MARGIN) / span
    for (job, out), (x, y, h, cols) in zip(jobs, projected):
        img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        px = ((x - cx) * scale + SIZE / 2).astype(np.int32)
        py = ((y - cy) * scale + SIZE / 2).astype(np.int32)
        order = np.argsort(h)  # low first, high points drawn on top
        px, py, cols_o = px[order], py[order], cols[order]
        ok = (px >= 0) & (px < SIZE) & (py >= 0) & (py < SIZE)
        img[py[ok], px[ok]] = cols_o[ok]
        Image.fromarray(img).save(out)
        print(f"{job}: {len(x)} points -> {out}")


if __name__ == "__main__":
    main()
