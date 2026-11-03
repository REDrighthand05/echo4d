"""Orbit still of the room cloud via MUS — ECHO-side glue.

Real-pose stills prove projection; this one proves the room reads as
a room: an elevated orbit camera built from the cloud bbox, with the
up direction taken from the capture poses themselves (mean camera
up — the same signal the pipeline's up-alignment uses).

Run:  experiments/render_op/.venv-mus/bin/python experiments/render_op/orbit_still.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_and_still import JOB, OUT, load_points_bin, quat_xyzw_to_R  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from echo_super.kernels.mus_render import render_frame  # noqa: E402


def main():
    xyz, rgb = load_points_bin(JOB / "points.bin")
    poses = json.loads((JOB / "poses.json").read_text())
    names = sorted(poses)

    ups, fwds = [], []
    for name in names:
        R = quat_xyzw_to_R(poses[name]["cam_from_world_quat_xyzw"])
        ups.append(-R[1])   # image-up direction, in world coords
        fwds.append(R[2])   # viewing direction, in world coords
    up = np.mean(ups, axis=0)
    up /= np.linalg.norm(up)
    fwd = np.mean(fwds, axis=0)
    fwd /= np.linalg.norm(fwd)

    centre = xyz.mean(axis=0)
    extent = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
    horiz = fwd - up * float(fwd @ up)
    horiz /= np.linalg.norm(horiz)
    cam_pos = centre - horiz * (1.35 * extent) + up * (0.80 * extent)

    f = centre - cam_pos
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])
    t = -R @ cam_pos

    W, H = 1600, 1200
    fx = (W / 2) / np.tan(np.radians(27.5))
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1.0]])
    cam = {"K": K, "R": R, "t": t}
    img = render_frame(xyz, rgb, camera=cam, width=W, height=H,
                       point_radius_px=2.2, glow_strength=0.35,
                       background_rgb=(5, 7, 14))
    out_path = OUT / "room_still_orbit.png"
    from PIL import Image
    Image.fromarray(img).save(out_path)
    lit = float((np.asarray(img).sum(axis=2) > 30).mean())
    print(f"orbit still: {out_path} centre={centre.round(2)} "
          f"extent={extent:.2f} up={up.round(2)} lit_frac={lit:.2f}")


if __name__ == "__main__":
    main()
