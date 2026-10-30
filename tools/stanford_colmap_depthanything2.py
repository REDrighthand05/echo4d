"""Stanford Lounge: eight measured depths plus two CPU learned relative depths.

300 original RGB, 287 COLMAP-estimated poses, 4 shared measured metric anchors.
Old eight-measured GLB is comparison only, never a fusion prior.
"""
import argparse
import json
import math
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from echo4d.model import Camera
from stanford_colmap_metric300 import original_pairs, read_model, depth_scale
from stanford_colmap_nested_depth import render_rays
from stanford_colmap_pose_cover10 import UPSTREAM_RUN, sha256, select_pose_coverage, five_rays
from stanford_colmap_stereo6 import CAMERA, camera_to_world, fuse_hybrid
from stanford_colmap_eight_measured_control import eight_only_names, old_mesh_to_open3d

CONTROL_RUN = "https://github.com/REDrighthand05/echo4d/actions/runs/41000309095"


MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
MODEL_REVISION = "32d03942121d29edb49de4e2cc15831558af3f36"
WEIGHTS_SHA256 = "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"


class DepthAnythingCPU:
    """Licensed Apache-2.0 small checkpoint, pinned weights, actual CPU inference."""

    def __init__(self):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        torch.set_num_threads(2)
        local = Path(snapshot_download(
            repo_id=MODEL_ID, revision=MODEL_REVISION,
            allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"]))
        if sha256(local / "model.safetensors") != WEIGHTS_SHA256:
            raise RuntimeError("Depth Anything V2 model weights SHA256 mismatch")
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(str(local))
        self.model = AutoModelForDepthEstimation.from_pretrained(
            str(local), use_safetensors=True).to("cpu").eval()

    def infer(self, rgb):
        import torch.nn.functional as F
        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt")
        with self.torch.inference_mode():
            relative = self.model(**inputs).predicted_depth.unsqueeze(1)
            relative = F.interpolate(
                relative, size=rgb.shape[:2], mode="bicubic", align_corners=False)
        return relative[0, 0].cpu().numpy().astype(np.float32)


def calibrate_sparse_metric_depth(relative, observations, w2c, scale):
    """Fit *per-target* relative inverse depth using RGB-COLMAP 3D tracks.

    Four measured depth anchors only establish global SfM-to-metre scale.
    No sensor depth from inferred target images, no old Mesh, no GT poses.
    A failed or unsupported fit raises instead of inventing metric depth.
    """
    if relative.ndim != 2 or not np.isfinite(relative).all():
        raise ValueError("Invalid relative depth prediction")
    h, w = relative.shape
    rows = []
    for x, y, xyz in observations:
        ix, iy = int(round(x)), int(round(y))
        if not (0 <= ix < w and 0 <= iy < h):
            continue
        z = float((w2c[:3, :3] @ xyz + w2c[:3, 3])[2] * scale)
        p = float(relative[iy, ix])
        if 0.55 < z < 2.9 and np.isfinite(p) and p > 0:
            rows.append((ix, iy, p, z))
    if len(rows) < 35:
        raise RuntimeError(f"Too few independent RGB-SfM tracks: {len(rows)}")
    a = np.asarray(rows, dtype=np.float64)
    x, z = a[:, 2], a[:, 3]
    A = np.stack((x, np.ones_like(x)), axis=1)
    target = 1.0 / z
    weights = np.ones_like(target)
    for _ in range(8):
        coef = np.linalg.lstsq(A * weights[:, None], target * weights, rcond=None)[0]
        residual = A @ coef - target
        robust = max(float(np.median(np.abs(residual))) * 1.4826, 0.015)
        weights = np.clip(1.5 * robust / np.maximum(np.abs(residual), 1e-9), 0.05, 1.0)
    slope, intercept = map(float, coef)
    if slope <= 0 or not np.isfinite(coef).all():
        raise RuntimeError(f"Relative depth has no positive inverse-depth calibration: {coef}")
    denom_sparse = A @ coef
    estimated = 1.0 / np.maximum(denom_sparse, 1e-6)
    residual_m = np.abs(estimated - z)
    inliers = residual_m < np.maximum(0.32, 0.22 * z)
    if np.count_nonzero(inliers) < 30 or np.mean(inliers) < 0.45:
        raise RuntimeError(
            f"Unreliable metric fit: {np.count_nonzero(inliers)}/{len(rows)} sparse inliers")
    # Refit on geometrically supported RGB-only tracks; no hidden sensor depth.
    coef = np.linalg.lstsq(A[inliers], target[inliers], rcond=None)[0]
    slope, intercept = map(float, coef)
    if slope <= 0:
        raise RuntimeError("Refitted inverse-depth slope is nonpositive")
    denom = relative.astype(np.float64) * slope + intercept
    metric = (1.0 / np.maximum(denom, 1e-6)).astype(np.float32)
    support = np.ones((h, w), dtype=np.uint8)
    for ix, iy, _, _ in a[inliers]:
        support[int(iy), int(ix)] = 0
    distance = cv2.distanceTransform(support, cv2.DIST_L2, 3)
    # Do not hallucinate unsupported entire walls: only predict within
    # 90 pixels of independently triangulated RGB observations.
    valid = ((distance <= 90) & (denom > 1.0 / 2.9) &
             (denom < 1.0 / 0.55) & np.isfinite(metric))
    depth = np.where(valid, metric, 0).astype(np.float32)
    info = dict(
        method="Depth Anything V2 Small CPU relative depth + RGB-SfM inverse-depth fit",
        model_id=MODEL_ID, model_revision=MODEL_REVISION,
        model_weights_sha256=WEIGHTS_SHA256,
        sparse_tracks=len(rows), sparse_inliers=int(inliers.sum()),
        sparse_median_residual_m=float(np.median(residual_m[inliers])),
        inverse_depth_slope=slope, inverse_depth_intercept=intercept,
        valid_fraction=float(np.mean(valid)), is_sensor_depth=False,
        metric_calibration="RGB-SfM triangulated tracks scaled by four measured depth anchors")
    return depth, info


def self_test():
    rng = np.random.default_rng(17)
    h, w = 120, 160
    yy, xx = np.indices((h, w))
    true_z = 1.3 + 0.003 * xx + 0.001 * yy
    relative = ((1.0 / true_z - 0.08) / 0.16).astype(np.float32)
    observations = []
    for _ in range(300):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        observations.append((x, y, np.array([0.0, 0.0, true_z[y, x]])))
    depth, info = calibrate_sparse_metric_depth(
        relative, observations, np.eye(4), 1.0)
    assert info["sparse_inliers"] >= 200, info
    assert np.mean(depth > 0) > 0.5, info
    assert np.median(np.abs(depth[depth > 0] - true_z[depth > 0])) < 0.03, info
    print("PASS: CPU relative-depth metric calibration on synthetic SfM tracks")


def run(upstream, control, output):
    import open3d as o3d
    import trimesh
    upstream, control, output = upstream.resolve(), control.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Refuse to overwrite a prior real experiment")
    output.mkdir(parents=True)
    started = time.time()
    receipt = dict(status="RUNNING", scene="Stanford Lounge", source_run=UPSTREAM_RUN,
                   control_run=CONTROL_RUN, input_rgb=300, registered_rgb=287,
                   measured_depth_maps=8, inferred_depth_maps=2,
                   prior_mesh=False, ground_truth_poses=False, gpu=False,
                   predicted_depth_source="Depth Anything V2 Small CPU, RGB-SfM metricized per image",
                   metric_scale_source="four real sensor depth anchors",
                   full_room_acceptance="FAIL", visual_acceptance="PENDING")
    def save():
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    save()
    try:
        parent = json.loads((upstream / "receipt.json").read_text())
        ledger = json.loads((upstream / "inputs.json").read_text())
        baseline = json.loads((control / "receipt.json").read_text())
        if (parent.get("selected_rgb"), parent.get("registered_frames"),
                parent.get("rgb_sampling")) != (300, 287, "early-overlap"):
            raise ValueError("Frozen independent RGB COLMAP input changed")
        if len(ledger) != 300 or baseline.get("measured_depth_maps") != 8:
            raise ValueError("Frozen RGB ledger or eight-depth control missing")
        views = read_model(upstream / parent["model"])
        if len(views) != 287:
            raise ValueError("Expected 287 registered RGB-only camera poses")
        lookup = {int(Path(n).stem): n for n, _, _ in views}
        base30 = [lookup[int(i)] for i in parent["measured_depth_rgb_samples"]]
        base15 = base30[::2]
        base10 = [base15[int(i)] for i in np.linspace(0, 14, 10, dtype=int)]
        shared = [base10[i] for i in (0, 3, 6, 9)]
        source = original_pairs(o3d.data.LoungeRGBDImages())
        rgb_paths, depth_paths = {}, {}
        for row in ledger:
            sample = int(row["sample"])
            rgb, dep = source[int(row["original_index"])]
            if sha256(rgb) != row["rgb_sha256"]:
                raise ValueError(f"Original RGB changed: {sample}")
            name = f"{sample:06d}{rgb.suffix.lower()}"
            rgb_paths[name], depth_paths[name] = rgb, dep
        for name in shared:
            i = int(Path(name).stem)
            if sha256(depth_paths[name]) != ledger[i]["depth_sha256"]:
                raise ValueError(f"Measured metric anchor changed: {i}")
        scale, witnesses = depth_scale(
            [v for v in views if v[0] in set(shared)],
            {n: depth_paths[n] for n in shared})
        if not np.isclose(scale, baseline["metric_scale_m_per_colmap_unit"],
                          rtol=1e-7, atol=1e-10):
            raise ValueError("Frozen measured metric scale changed")
        ten = select_pose_coverage(views, shared, 10, scale)
        eight = eight_only_names(ten, baseline["stereo_replacement_samples"])
        if sorted(int(Path(n).stem) for n in eight) != baseline["measured_depth_samples"]:
            raise ValueError("Eight measured input frames changed")
        old_glb = control / "measured8_only.glb"
        if sha256(old_glb) != baseline["measured8_only_sha256"]:
            raise ValueError("Comparison GLB hash mismatch")
        old = old_mesh_to_open3d(o3d, trimesh, old_glb)
        poses = {name: camera_to_world(w2c, scale) for name, w2c, _ in views}
        cache = {}
        def rgb(name):
            if name not in cache:
                with Image.open(rgb_paths[name]) as im:
                    cache[name] = np.asarray(im.convert("RGB"), dtype=np.uint8)
            return cache[name]
        inferred = {}
        ledger_mvs = []
        predictor = DepthAnythingCPU()  # loads real, SHA-verified weights on CPU
        view_by_name = {name: (w2c, matches) for name, w2c, matches in views}
        for sample in baseline["stereo_replacement_samples"]:
            name = f"{int(sample):06d}.png"
            relative = predictor.infer(rgb(name))
            w2c, matches = view_by_name[name]
            depth, info = calibrate_sparse_metric_depth(relative, matches, w2c, scale)
            if info["valid_fraction"] < .002:
                raise RuntimeError(f"Learned depth has no supported pixels: {name}: {info}")
            inferred[name] = (rgb(name), depth, CAMERA, poses[name])
            Image.fromarray(np.clip(depth*1000, 0, 65535).astype(np.uint16)).save(
                output / f"inferred-{Path(name).stem}-mm.png")
            ledger_mvs.append(dict(target=name, **info))
            receipt["inferred"] = ledger_mvs
            save()
        mesh = fuse_hybrid(o3d, trimesh, views, eight, inferred, scale,
                           rgb_paths, depth_paths, output / "measured8_da2.glb")
        recorded, rays = five_rays(o3d, views, scale)
        panel = Image.new("RGB", (960, 5*270+30), (18, 18, 18))
        draw = ImageDraw.Draw(panel)
        for col, title in enumerate(("Original RGB", "8 measured ONLY", "8 measured + 2 DA2 CPU")):
            draw.text((col*320+5, 8), title, fill="white")
        hits = {"measured8_only": [], "measured8_da2": []}
        for i, ray in enumerate(rays):
            y = 30+i*270
            draw.text((5, y+3), f"recorded {Path(recorded[i][0]).stem}" if i<3
                      else f"free orbit {i-2}", fill="white")
            if i < 3:
                with Image.open(rgb_paths[recorded[i][0]]) as im:
                    panel.paste(im.convert("RGB").resize((320,240)), (0,y+26))
            for col, (key, obj) in enumerate((
                ("measured8_only", old), ("measured8_da2", mesh)), 1):
                rendered, hit = render_rays(o3d, obj, ray)
                panel.paste(rendered, (col*320,y+26))
                hits[key].append(hit)
        panel.save(output / "source-eight-only-vs-eight-da2-3plus2.png")
        receipt.update(status="REAL_MESH_GENERATED_VISUAL_REVIEW_REQUIRED",
                       measured_depth_samples=baseline["measured_depth_samples"],
                       inferred_depth_samples=baseline["stereo_replacement_samples"],
                       shared_scale_samples=baseline["shared_scale_samples"],
                       metric_scale_m_per_colmap_unit=scale,
                       metric_scale_witnesses=witnesses,
                       baseline_glb_sha256=sha256(old_glb),
                       da2_glb_sha256=sha256(output/"measured8_da2.glb"),
                       same_ray_hit_percent=hits,
                       visual_acceptance="PENDING five-view inspection; full room FAIL")
    except Exception as exc:
        receipt.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["elapsed_seconds"] = round(time.time()-started,2)
        save()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path)
    p.add_argument("--control", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        self_test()
    else:
        if not all((a.upstream, a.control, a.output)):
            p.error("--upstream, --control and --output required")
        run(a.upstream, a.control, a.output)
