"""MUS bench + real-data still — ECHO-side glue (NOT part of mus_render).

Loads the room job's fused cloud (jobs/room2_trim2/points.bin, the
208,666-point K=2 production cloud), takes a REAL camera pose from
the job's poses.json (cam_from_world -> MUS camera dict directly),
and:
  * renders candidate stills (room + camera-route lines) to PNG,
  * benchmarks render_frame at 1280x720 and 1920x1080 (median of 10
    after 2 warmups), writing bench_results.json.

Run:  experiments/render_op/.venv-mus/bin/python experiments/render_op/bench_and_still.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from echo_super.kernels.mus_render import render_frame  # noqa: E402

JOB = ROOT / "jobs" / "room2_trim2"
OUT = Path(__file__).resolve().parent


def load_points_bin(path):
    raw = np.fromfile(path, dtype=np.uint8)
    rec = raw.reshape(-1, 15)
    xyz = rec[:, :12].copy().view(np.float32).reshape(-1, 3)
    rgb = rec[:, 12:15].copy()
    return np.ascontiguousarray(xyz), rgb


def quat_xyzw_to_R(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def camera_from_pose(pose, out_w, out_h):
    cam = pose["camera"]
    params = cam["params"]
    model = cam["model"]
    if "SIMPLE" in model:
        # SIMPLE_PINHOLE / SIMPLE_RADIAL: params are [f, cx, cy, ...]
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    else:  # PINHOLE / OPENCV / FULL_OPENCV: [fx, fy, cx, cy, ...]
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    sx = out_w / cam["width"]
    sy = out_h / cam["height"]
    K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1.0]])
    R = quat_xyzw_to_R(pose["cam_from_world_quat_xyzw"])
    t = np.array(pose["cam_from_world_t"], dtype=float)
    return {"K": K, "R": R, "t": t}, cam["width"], cam["height"]


def main():
    xyz, rgb = load_points_bin(JOB / "points.bin")
    print(f"cloud: {xyz.shape[0]} points")
    poses = json.loads((JOB / "poses.json").read_text())
    names = sorted(poses)
    print(f"poses: {len(names)} frames; model: "
          f"{poses[names[0]]['camera']['model']} "
          f"{poses[names[0]]['camera']['width']}x"
          f"{poses[names[0]]['camera']['height']}")

    # Camera route as line segments (camera centres in world frame).
    centres = []
    for name in names:
        p = poses[name]
        R = quat_xyzw_to_R(p["cam_from_world_quat_xyzw"])
        t = np.array(p["cam_from_world_t"], dtype=float)
        centres.append(-R.T @ t)
    centres = np.array(centres)
    route = np.stack([centres[:-1], centres[1:]], axis=1).astype(np.float32)

    # Candidate stills from real viewpoints at 25% / 50% / 75%.
    for frac in (0.25, 0.5, 0.75):
        name = names[int(frac * (len(names) - 1))]
        cam, iw, ih = camera_from_pose(poses[name], 1280, round(1280 * ih_safe(poses[name])))
        for tag, lines_arg in (("route", route), ("clean", None)):
            img = render_frame(xyz, rgb, lines_arg, camera=cam, width=1280,
                               height=img_h(poses[name]),
                               point_radius_px=2.0, glow_strength=0.45,
                               background_rgb=(5, 7, 14))
            out_path = OUT / f"room_still_{int(frac * 100)}_{tag}.png"
            from PIL import Image
            Image.fromarray(img).save(out_path)
            print(f"still {out_path.name}: pose {name}, "
                  f"{img.shape[1]}x{img.shape[0]}")

    # Benchmark at exact 720p / 1080p with the middle pose.
    mid = poses[names[len(names) // 2]]
    results = {}
    for (w, h) in ((1280, 720), (1920, 1080)):
        cam, _, _ = camera_from_pose(mid, w, h)
        for _ in range(2):
            render_frame(xyz, rgb, route, camera=cam, width=w, height=h,
                         point_radius_px=1.7, glow_strength=0.45,
                         background_rgb=(5, 7, 14))
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            render_frame(xyz, rgb, route, camera=cam, width=w, height=h,
                         point_radius_px=1.7, glow_strength=0.45,
                         background_rgb=(5, 7, 14))
            times.append(time.perf_counter() - t0)
        times.sort()
        med = times[len(times) // 2]
        results[f"{w}x{h}"] = {
            "median_ms": round(med * 1000, 1),
            "min_ms": round(times[0] * 1000, 1),
            "max_ms": round(times[-1] * 1000, 1),
            "fps": round(1.0 / med, 2),
            "points": int(xyz.shape[0]),
            "route_segments": int(route.shape[0]),
        }
        print(f"bench {w}x{h}: median {med * 1000:.1f} ms "
              f"({1.0 / med:.2f} fps) over {xyz.shape[0]} points")
    (OUT / "bench_results.json").write_text(json.dumps(results, indent=2))


def ih_safe(pose):
    cam = pose["camera"]
    return cam["height"] / cam["width"]


def img_h(pose):
    return int(round(1280 * ih_safe(pose)))


if __name__ == "__main__":
    main()
