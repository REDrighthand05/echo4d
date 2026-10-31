"""Scratch driver: rerun the depth stage of a job with Depth Anything
V2 Large, leaving production code untouched.

Mirrors echo_scene.pipeline.stage_depth's contract exactly (frames =
poses.json keys, output = depth/<stem>.npy via echo_scene.depth.
estimate_depth), with two deliberate differences:

  * the model is models/depth_anything_v2_large.onnx instead of the
    production Base pick;
  * the frame set is poses ∩ existing depth files — the exact set the
    room2_sc baseline (spread 0.0751171) was measured/fused on. The
    historical quirk is preserved on purpose so the model is the ONLY
    changed variable: frame_00163 is registered but has no baseline
    depth map, frame_00271 has a depth map but no pose.

Usage: python experiments/large_depth.py <job_dir> [model_path]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echo_scene.depth import estimate_depth  # noqa: E402


def main() -> None:
    job = Path(sys.argv[1])
    root = Path(__file__).resolve().parent.parent
    model = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        root / "models" / "depth_anything_v2_large.onnx")
    poses = json.loads((job / "poses.json").read_text())
    depth_dir = job / "depth"
    targets = []
    for name in sorted(poses.keys()):
        stem = Path(name).stem
        if (depth_dir / f"{stem}.npy").exists() and (job / "frames" / name).exists():
            targets.append(name)
    print(f"[large-depth] model={model.name} frames={len(targets)}", flush=True)
    t0 = time.time()
    for i, name in enumerate(targets, 1):
        depth = estimate_depth(job / "frames" / name, model)
        np.save(depth_dir / f"{Path(name).stem}.npy", depth)
        if i % 10 == 0 or i == len(targets):
            el = time.time() - t0
            print(f"[large-depth] {i}/{len(targets)} "
                  f"elapsed={el:.0f}s eta={el / i * (len(targets) - i):.0f}s",
                  flush=True)
    print(f"[large-depth] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
