"""Stanford Lounge: eight real depths plus two CPU RGB multi-view inferred depths.

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


def select_sources(target, poses, count=5):
    ref = poses[target]
    candidates = []
    for name, pose in poses.items():
        if name == target:
            continue
        delta = ref[:3, :3].T @ (pose[:3, 3] - ref[:3, 3])
        baseline = float(np.linalg.norm(delta))
        lateral = float(np.linalg.norm(delta[:2]))
        rotation = pose[:3, :3].T @ ref[:3, :3]
        angle = math.degrees(math.acos(float(np.clip((np.trace(rotation)-1)/2, -1, 1))))
        if not (0.055 <= baseline <= 0.45 and lateral > 0.045 and angle < 30):
            continue
        time_gap = abs(int(Path(name).stem) - int(Path(target).stem))
        score = abs(baseline - 0.16) + 0.003 * angle + 0.0005 * time_gap
        candidates.append((score, name, float(delta[0]), baseline, angle))
    candidates.sort()
    chosen = []
    for sign in (False, True):
        match = next((x for x in candidates if (x[2] > 0) == sign), None)
        if match is not None:
            chosen.append(match)
    for item in candidates:
        if item[1] not in {x[1] for x in chosen}:
            chosen.append(item)
        if len(chosen) >= count:
            break
    if len(chosen) < 2:
        raise RuntimeError(f"Insufficient RGB multi-view neighbours: {target}")
    return chosen[:count]


def plane_sweep(target, pose, sources, cam=CAMERA, *, planes=56):
    """CPU calibrated RGB-only plane sweep; low-confidence depth remains zero."""
    if len(sources) < 2 or target.shape != (cam.height, cam.width, 3):
        raise ValueError("Require calibrated target and two RGB neighbours")
    factor = 2 if cam.width > 400 else 1
    w, h = cam.width // factor, cam.height // factor
    fx, fy = cam.fx / factor, cam.fy / factor
    cx, cy = (cam.cx + .5) / factor - .5, (cam.cy + .5) / factor - .5
    def gray(image):
        small = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY),
                                (3, 3), 0).astype(np.float32)
    reference = gray(target)
    others = []
    for name, rgb, other_pose in sources:
        if rgb.shape != target.shape:
            raise ValueError(f"Source RGB dimensions differ: {name}")
        rotation = other_pose[:3, :3].T @ pose[:3, :3]
        translation = other_pose[:3, :3].T @ (pose[:3, 3] - other_pose[:3, 3])
        others.append((gray(rgb), rotation, translation))
    yy, xx = np.indices((h, w), dtype=np.float32)
    rays = np.stack(((xx - cx) / fx, (yy - cy) / fy, np.ones_like(xx)))
    depth_planes = 1 / np.linspace(1 / 2.9, 1 / .55, planes, dtype=np.float32)
    costs = np.full((planes, h, w), 999, np.float32)
    for index, depth in enumerate(depth_planes):
        xyz = rays * depth
        per_view = []
        for image, rotation, translation in others:
            q = np.einsum("ij,jhw->ihw", rotation, xyz) + translation[:, None, None]
            z = np.maximum(q[2], 1e-4)
            u = (fx * q[0] / z + cx).astype(np.float32)
            v = (fy * q[1] / z + cy).astype(np.float32)
            inside = (q[2] > .2) & (u >= 2) & (u < w-2) & (v >= 2) & (v < h-2)
            warped = cv2.remap(image, u, v, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)
            cost = cv2.boxFilter(np.abs(warped - reference), -1, (3, 3))
            per_view.append(np.where(inside, cost, 999))
        ordered = np.sort(np.stack(per_view), axis=0)
        costs[index] = np.where(ordered[1] < 999,
                                (ordered[0] + ordered[1]) / 2, 999)
    best = costs.argmin(axis=0)
    best_cost = np.take_along_axis(costs, best[None], axis=0)[0]
    indices = np.arange(planes)[:, None, None]
    runner_up = np.where(np.abs(indices - best[None]) <= 2, 999, costs).min(axis=0)
    gradient = (np.abs(cv2.Sobel(reference, cv2.CV_32F, 1, 0, ksize=3)) +
                np.abs(cv2.Sobel(reference, cv2.CV_32F, 0, 1, ksize=3)))
    valid = (best_cost < 20) & (runner_up - best_cost > 2) & (gradient > 12)
    low_res = np.where(valid, depth_planes[best], 0).astype(np.float32)
    full_res = cv2.resize(low_res, (cam.width, cam.height),
                          interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(full_res), {
        "method": "calibrated CPU multi-view RGB plane sweep",
        "rgb_sources": [s[0] for s in sources],
        "depth_planes": planes,
        "valid_fraction": float(np.mean(valid)),
        "depth_m": [0.55, 2.9],
        "is_sensor_depth": False,
    }


def self_test():
    rng = np.random.default_rng(12)
    h, w, true_z = 120, 160, 1.5
    cam = Camera(w, h, 110, 110, 79.5, 59.5)
    texture = cv2.GaussianBlur(rng.integers(0, 256, (h, w), dtype=np.uint8),
                               (3, 3), 0)
    rgb = cv2.cvtColor(texture, cv2.COLOR_GRAY2RGB)
    sources = []
    for i, baseline in enumerate((.10, -.12, .18)):
        pose = np.eye(4)
        pose[0, 3] = baseline
        shift = -cam.fx * baseline / true_z
        warped = cv2.warpAffine(rgb, np.float32([[1, 0, shift], [0, 1, 0]]),
                                (w, h), flags=cv2.INTER_LINEAR)
        sources.append((str(i), warped, pose))
    depth, info = plane_sweep(rgb, np.eye(4), sources, cam, planes=48)
    inner = depth[12:-12, 12:-12]
    values = inner[inner > 0]
    assert len(values) > inner.size * .05, info
    assert abs(float(np.median(values)) - true_z) < .18, np.median(values)
    print("PASS synthetic RGB plane sweep; median depth", float(np.median(values)))


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
                   predicted_depth_source="CPU RGB multi-view plane sweep",
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
        for sample in baseline["stereo_replacement_samples"]:
            name = f"{int(sample):06d}.png"
            choices = select_sources(name, poses)
            neighbours = [(n, rgb(n), poses[n]) for _, n, _, _, _ in choices]
            depth, info = plane_sweep(rgb(name), poses[name], neighbours)
            if info["valid_fraction"] < .002:
                raise RuntimeError(f"RGB MVS produced insufficient depth for {name}: {info}")
            inferred[name] = (rgb(name), depth, CAMERA, poses[name])
            Image.fromarray(np.clip(depth*1000, 0, 65535).astype(np.uint16)).save(
                output / f"inferred-{Path(name).stem}-mm.png")
            ledger_mvs.append(dict(target=name, neighbours=[
                dict(name=n, baseline_m=float(b), angle_deg=float(a))
                for _, n, _, b, a in choices], **info))
            receipt["inferred"] = ledger_mvs
            save()
        mesh = fuse_hybrid(o3d, trimesh, views, eight, inferred, scale,
                           rgb_paths, depth_paths, output / "measured8_mvs2.glb")
        recorded, rays = five_rays(o3d, views, scale)
        panel = Image.new("RGB", (960, 5*270+30), (18, 18, 18))
        draw = ImageDraw.Draw(panel)
        for col, title in enumerate(("Original RGB", "8 measured ONLY", "8 measured + 2 RGB MVS")):
            draw.text((col*320+5, 8), title, fill="white")
        hits = {"measured8_only": [], "measured8_mvs2": []}
        for i, ray in enumerate(rays):
            y = 30+i*270
            draw.text((5, y+3), f"recorded {Path(recorded[i][0]).stem}" if i<3
                      else f"free orbit {i-2}", fill="white")
            if i < 3:
                with Image.open(rgb_paths[recorded[i][0]]) as im:
                    panel.paste(im.convert("RGB").resize((320,240)), (0,y+26))
            for col, (key, obj) in enumerate((
                ("measured8_only", old), ("measured8_mvs2", mesh)), 1):
                rendered, hit = render_rays(o3d, obj, ray)
                panel.paste(rendered, (col*320,y+26))
                hits[key].append(hit)
        panel.save(output / "source-eight-only-vs-eight-mvs2-3plus2.png")
        receipt.update(status="REAL_MESH_GENERATED_VISUAL_REVIEW_REQUIRED",
                       measured_depth_samples=baseline["measured_depth_samples"],
                       inferred_depth_samples=baseline["stereo_replacement_samples"],
                       shared_scale_samples=baseline["shared_scale_samples"],
                       metric_scale_m_per_colmap_unit=scale,
                       metric_scale_witnesses=witnesses,
                       baseline_glb_sha256=sha256(old_glb),
                       mvs_glb_sha256=sha256(output/"measured8_mvs2.glb"),
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
