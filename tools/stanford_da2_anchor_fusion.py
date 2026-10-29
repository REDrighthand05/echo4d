"""Stanford Lounge: replace 120/150 measured Z maps with pinned DA2-Small CPU.

All 150 RGB frames and ALL 3000-frame RGB-D-estimated camera poses are retained.
30 sensor-depth frames calibrate relative RGB prediction to metres. This is NOT
an independent sparse-capture or RGB-only reconstruction. No Mesh is an input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from official_room_depth_ladder import (UPSTREAM_SHA, even_indices, integrate_subset,
                                        load_trajectory, native_pairs, render_fixed_views)

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
MODEL_REVISION = "870a35c76c2bc1d82fbde922d95015496cb7dd6c"
MODEL_SHA256 = "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"


def normalized_relative(relative: np.ndarray) -> np.ndarray:
    r = np.asarray(relative, dtype=np.float32)
    if r.ndim != 2 or not np.isfinite(r).all():
        raise ValueError("Nonfinite relative depth")
    lo, hi = np.percentile(r, [2, 98])
    if hi - lo < 1e-5:
        raise ValueError("Flat relative depth")
    return np.clip((r - lo) / (hi - lo), -0.3, 1.3)


def fit_inverse_depth(samples: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float, dict]:
    """Robust global inverse-depth scale from 30 actual sensor frames only."""
    from scipy.optimize import least_squares
    xs, ys = [], []
    rng = np.random.default_rng(20261029)
    for relative, sensor_m in samples:
        if relative.shape != sensor_m.shape:
            raise ValueError("Anchor dimensions differ")
        good = np.isfinite(sensor_m) & (sensor_m >= 0.45) & (sensor_m <= 3.)
        ids = np.flatnonzero(good)
        if len(ids) < 200:
            continue
        ids = rng.choice(ids, size=min(1200, len(ids)), replace=False)
        xs.append(relative.ravel()[ids])
        ys.append(1. / sensor_m.ravel()[ids])
    if len(xs) < 10:
        raise RuntimeError("Too few valid measured-depth anchors")
    x, y = np.concatenate(xs), np.concatenate(ys)
    solution = least_squares(lambda v: v[0]*x + v[1] - y,
                             x0=[0.6, 0.35], bounds=([0.0001, 0.0001], [10., 4.]),
                             loss="soft_l1", f_scale=0.12, max_nfev=100)
    a, b = map(float, solution.x)
    if not solution.success or a <= .005 or not np.isfinite([a, b]).all():
        raise RuntimeError("Unstable DA2 inverse-depth calibration")
    return a, b, {"anchor_frames_fit": len(xs), "anchor_pixels_fit": len(x),
                  "metric_source": "30 actual measured depth PNGs, never inferred by DA2",
                  "fit_inverse_depth": [a, b]}


def predicted_metric(relative: np.ndarray, a: float, b: float) -> np.ndarray:
    inv = a * relative + b
    depth = np.zeros(relative.shape, np.uint16)
    valid = np.isfinite(inv) & (inv >= 1/3.) & (inv <= 1/0.45)
    depth[valid] = np.round(1000. / inv[valid]).astype(np.uint16)
    return depth


def compose_views(output: Path) -> Path:
    names = ["recorded-00000.png", "recorded-01500.png", "recorded-02999.png",
             "free-1.png", "free-2.png"]
    columns = ["real source", "150 measured", "30 measured", "30 measured + 120 DA2"]
    board = Image.new("RGB", (1280, 5 * 264 + 34), (19, 19, 19))
    draw = ImageDraw.Draw(board)
    for col, title in enumerate(columns):
        draw.text((col * 320 + 6, 10), title, fill="white")
    for row, name in enumerate(names):
        y = 34 + row * 264
        draw.text((5, y), name[:-4], fill="white")
        for col, key in enumerate(["measured150", "measured150", "measured30", "hybrid"]):
            with Image.open(output / key / "views" / name) as im:
                im = im.convert("RGB")
                left = 0 if col == 0 and row < 3 else im.width - 320
                panel = im.crop((left, 28, left + 320, 268))
                if col == 0 and row >= 3:
                    panel = Image.new("RGB", (320, 240), (19, 19, 19))
                board.paste(panel, (col * 320, y + 24))
    target = output / "source-measured150-measured30-da2hybrid-3plus2.png"
    board.save(target)
    return target


def run(official: Path, output: Path) -> None:
    import open3d as o3d
    import torch
    import trimesh
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    if output.exists():
        raise FileExistsError("Use fresh output directory")
    output.mkdir(parents=True)
    report = {"status": "RUNNING", "scene": "real Stanford Lounge",
              "input_rgb_frames": 150, "known_pose_prior": "ALL 3000 RGB-D frames",
              "measured_depth_frames": 30, "predicted_depth_frames": 120,
              "existing_mesh_prior": False, "model": MODEL_ID,
              "model_revision": MODEL_REVISION, "model_license": "Apache-2.0 Small",
              "model_checkpoint_sha256_expected": MODEL_SHA256,
              "prediction_device": "cpu", "visual_acceptance": "NOT REVIEWED"}
    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    receipt = json.loads((official / "receipt.json").read_text())
    if receipt.get("upstream_commit") != UPSTREAM_SHA:
        raise ValueError("Wrong original Open3D official source")
    poses = load_trajectory(official / "scene" / "trajectory.log")
    if len(poses) != 3000:
        raise ValueError("Expected all 3000 frozen upstream poses")
    original = o3d.data.LoungeRGBDImages()
    pairs = native_pairs(Path(original.color_paths[0]).parent,
                         Path(original.depth_paths[0]).parent, poses)
    if len(pairs) != 3000:
        raise ValueError("Expected all 3000 original paired frames")
    chosen = even_indices(len(pairs), 150)
    anchors = set(even_indices(150, 30))
    selected = [pairs[i] for i in chosen]
    report["selected_original_ordinals"] = chosen
    report["measured_anchor_ordinals"] = [chosen[i] for i in sorted(anchors)]
    save()

    torch.set_num_threads(2)
    checkpoint = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION,
                          allow_patterns=["model.safetensors", "config.json", "preprocessor_config.json"]))
    actual_sha = hashlib.sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest()
    if actual_sha != MODEL_SHA256:
        raise ValueError("Pretrained DA2 checkpoint bytes do not match pinned SHA256")
    processor = AutoImageProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(checkpoint,
                                                       local_files_only=True).to("cpu").eval()
    relative_dir = output / "relative"
    relative_dir.mkdir()
    measured_samples = []
    inference_order = sorted(anchors) + [i for i in range(150) if i not in anchors]
    for n, i in enumerate(inference_order):
        _, color_path, depth_path, _ = selected[i]
        with Image.open(color_path) as im:
            rgb = im.convert("RGB")
            inputs = processor(images=rgb, return_tensors="pt")
        with torch.inference_mode():
            predicted = model(**inputs).predicted_depth
            resized = torch.nn.functional.interpolate(predicted.unsqueeze(1),
                size=(480, 640), mode="bicubic", align_corners=False).squeeze()
        relative = normalized_relative(resized.cpu().numpy())
        np.save(relative_dir / f"{i:03d}.npy", relative.astype(np.float16), allow_pickle=False)
        if i in anchors:
            with Image.open(depth_path) as im:
                measured = np.asarray(im, dtype=np.uint16).astype(np.float32) / 1000.
            measured_samples.append((relative, measured))
        if (n + 1) % 15 == 0:
            print(f"REAL_CPU_DA2 {n+1}/150", flush=True)
    a, b, calibration = fit_inverse_depth(measured_samples)
    report["calibration"] = calibration
    report["checkpoint_sha256_verified"] = actual_sha
    save()

    predicted_dir = output / "predicted_mm"
    predicted_dir.mkdir()
    hybrid = []
    for i, (ordinal, rgb_path, depth_path, pose) in enumerate(selected):
        if i in anchors:
            hybrid.append((ordinal, rgb_path, depth_path, pose))
        else:
            rel = np.load(relative_dir / f"{i:03d}.npy", allow_pickle=False).astype(np.float32)
            z = predicted_metric(rel, a, b)
            path = predicted_dir / f"{i:03d}.png"
            Image.fromarray(z).save(path)
            hybrid.append((ordinal, rgb_path, path, pose))
    candidates = {
        "measured150": selected,
        "measured30": [selected[i] for i in sorted(anchors)],
        "hybrid": hybrid,
    }
    full = o3d.io.read_triangle_mesh(str(official / "scene" / "integrated.ply"))
    bounds = np.array([full.get_min_bound(), full.get_max_bound()])
    models = {}
    for key, frames in candidates.items():
        mesh, stat = integrate_subset(o3d, trimesh, frames, len(frames), output,
                                      voxel=3.0 / 512.0)
        glb = output / f"measured-{len(frames)}.glb"
        renamed = output / f"{key}.glb"
        glb.rename(renamed)
        models[key] = stat | {"glb": renamed.name}
        (output / key).mkdir()
        render_fixed_views(o3d, mesh, pairs, bounds, output / key)
        report["models"] = models
        save()
    image = compose_views(output)
    report["status"] = "REAL_MESH_GENERATED_VISUAL_REVIEW_REQUIRED"
    report["visual"] = image.name
    report["visual_acceptance"] = "FAIL until actual five-view image reviewed"
    save()
    print(f"REAL_RESULT {image}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.official.resolve(), args.output.resolve())
