"""Use real RGB-SfM observed landmarks to choose ten measured depth frames.

A/B against the previous RGB-pose-frustum selection at exactly the same 300
RGBs, 287 independently recovered COLMAP poses, ten original measured depths,
and four shared metric-scale depth anchors. No old Mesh, GT poses, predicted
depth, or GPU. SfM landmarks are RGB-derived sparse observations, not dense
measured surfaces; a textureless wall may remain absent.
"""
import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from stanford_colmap_metric300 import original_pairs, read_model, depth_scale
from stanford_colmap_nested_depth import render_rays
from stanford_colmap_pose_cover10 import (
    sha256, select_pose_coverage, fuse, five_rays,
)

UPSTREAM_RUN = "https://github.com/REDrighthand05/echo4d/actions/runs/40970565394"
VOXEL_METERS = 0.12
BALANCE_METERS = 0.60


def select_sfm_coverage(views, fixed, budget, scale):
    """Greedy observed RGB-SfM voxel coverage, with real multi-view support.

    Uses ONLY image keypoint -> COLMAP 3D point associations and RGB poses.
    Counts a cell only if observed by >=2 different RGB cameras. Candidate
    cells must lie within the measured-depth fusion's 0.3-3m range and actual
    RGB image bounds. Equalize coarse spatial regions to prevent textured
    furniture from drowning out less-textured room areas. Re-observation is
    useful but has diminishing returns. No depth maps are opened here.
    """
    ordered = sorted(views, key=lambda v: v[0])
    names = [v[0] for v in ordered]
    if len(names) != len(set(names)) or len(fixed) != len(set(fixed)):
        raise ValueError("Duplicate image names or fixed anchors")
    if not set(fixed).issubset(names) or not len(fixed) < budget <= len(names):
        raise ValueError("Invalid anchors or ten-frame budget")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid COLMAP-to-meters scale")

    seen = []
    support = Counter()
    for name, w2c, matches in ordered:
        keys = set()
        for x, y, point in matches:
            if not (0 <= x < 640 and 0 <= y < 480):
                continue
            p = np.asarray(point, dtype=np.float64)
            if p.shape != (3,) or not np.all(np.isfinite(p)):
                continue
            z = scale * float(w2c[2, :3] @ p + w2c[2, 3])
            if not (0.3 < z < 3.0):
                continue
            keys.add(tuple(np.floor(p * scale / VOXEL_METERS).astype(np.int64)))
        seen.append(keys)
        support.update(keys)

    cells = sorted(k for k, count in support.items() if count >= 2)
    if len(cells) < 100:
        raise RuntimeError("Too few real multi-view SfM voxels for coverage selection")
    ids = {key: i for i, key in enumerate(cells)}
    coarse = [tuple(np.floor(np.asarray(key) * VOXEL_METERS /
                              BALANCE_METERS).astype(np.int64)) for key in cells]
    population = Counter(coarse)
    # Each occupied coarse room region contributes roughly equally, while
    # a cell supported by more RGB cameras is mildly more trustworthy.
    weights = np.asarray([
        min(support[key], 12) ** 0.25 / population[group]
        for key, group in zip(cells, coarse)], dtype=np.float64)
    view_cells = [np.asarray(sorted(ids[k] for k in keys if k in ids),
                             dtype=np.int32) for keys in seen]
    if not all(len(view_cells[names.index(n)]) for n in fixed):
        raise RuntimeError("A fixed depth anchor has no supported SfM cells")
    chosen = [names.index(n) for n in fixed]
    counts = np.zeros(len(cells), dtype=np.int16)
    for i in chosen:
        counts[view_cells[i]] += 1
    while len(chosen) < budget:
        gain_per_cell = weights / np.square(1.0 + counts)
        scores = np.asarray([gain_per_cell[ix].sum() if len(ix) else -1.0
                             for ix in view_cells])
        scores[chosen] = -np.inf
        best = int(np.argmax(scores))
        if not np.isfinite(scores[best]) or scores[best] <= 0:
            raise RuntimeError("No remaining SfM-supported depth candidate")
        chosen.append(best)
        counts[view_cells[best]] += 1
    return sorted(names[i] for i in chosen), {
        "supported_sfm_voxels": len(cells),
        "candidate_views_with_sfm": sum(bool(len(v)) for v in view_cells),
        "covered_voxels": int(np.count_nonzero(counts)),
        "algorithm": "observed RGB-SfM points; >=2 RGB views; 12cm voxels; "
                     "60cm equalized regions; diminishing overlap",
    }


def self_test():
    views = []
    # Two differently textured synthetic walls: the result must be invariant
    # to input order, use exactly the budget, and preserve four fixed anchors.
    for i in range(24):
        pose = np.eye(4)
        pose[0, 3] = -0.04 * i
        matches = []
        for j in range(80):
            p = np.array([(j % 16) * .09, (j // 16) * .12,
                          1.0 + (i // 8) * .20])
            matches.append((320.0, 240.0, p))
        views.append((f"{i:06d}.png", pose, matches))
    fixed = [views[i][0] for i in (0, 7, 14, 21)]
    a, details = select_sfm_coverage(views, fixed, 10, 1.0)
    b, _ = select_sfm_coverage(list(reversed(views)), fixed, 10, 1.0)
    assert a == b and len(a) == len(set(a)) == 10
    assert set(fixed).issubset(a) and details["supported_sfm_voxels"] >= 100
    try:
        select_sfm_coverage(views, fixed, 4, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid depth budget was accepted")
    print("PASS: deterministic real-SfM voxel selection, fixed anchors, strict budget")


def run(upstream, output):
    import open3d as o3d
    import trimesh

    upstream, output = upstream.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Fresh output directory required")
    output.mkdir(parents=True)
    start = time.time()
    receipt = dict(
        status="RUNNING", scene="Stanford Lounge measured RGB-D",
        upstream_run=UPSTREAM_RUN, upstream_artifact_id=12939354226,
        prior_mesh=False, prior_gt_poses=False, predicted_depth=False,
        pose_source="300 original RGB COLMAP; 287 registered cameras",
        selection_source="actual RGB-SfM multi-view point observations",
        visual_acceptance="FAIL until five-view human review",
        free_orbit_acceptance="FAIL until five-view human review")
    def save():
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    save()
    try:
        parent = json.loads((upstream / "receipt.json").read_text())
        ledger = json.loads((upstream / "inputs.json").read_text())
        if (parent["selected_rgb"], parent["registered_frames"],
                parent["rgb_sampling"]) != (300, 287, "early-overlap"):
            raise ValueError("Unexpected frozen RGB-only COLMAP source")
        if len(ledger) != 300:
            raise ValueError("Original 300 RGB input ledger missing")
        views = read_model(upstream / parent["model"])
        if len(views) != 287:
            raise ValueError("Wrong number of independently registered cameras")
        registered = sorted(name for name, _, _ in views)
        lookup = {int(Path(name).stem): name for name in registered}
        base30 = [lookup[i] for i in parent["measured_depth_rgb_samples"]]
        base15 = base30[::2]
        base10 = [base15[int(i)] for i in np.linspace(0, 14, 10, dtype=int)]
        if len(set(base10)) != 10:
            raise ValueError("Temporal ten-depth baseline changed")
        shared = [base10[i] for i in (0, 3, 6, 9)]
        source = original_pairs(o3d.data.LoungeRGBDImages())
        colors, depths = {}, {}
        for row in ledger:
            i = int(row["sample"])
            rgb, dep = source[int(row["original_index"])]
            if sha256(rgb) != row["rgb_sha256"]:
                raise ValueError(f"Changed RGB source sample {i}")
            name = f"{i:06d}{rgb.suffix.lower()}"
            colors[name], depths[name] = rgb, dep
        for name in shared:
            i = int(Path(name).stem)
            if sha256(depths[name]) != ledger[i]["depth_sha256"]:
                raise ValueError(f"Changed shared depth scale anchor {i}")
        scale, witnesses = depth_scale(
            [v for v in views if v[0] in set(shared)],
            {name: depths[name] for name in shared})
        pose_names = select_pose_coverage(views, shared, 10, scale)
        sfm_names, sfm_stats = select_sfm_coverage(views, shared, 10, scale)
        if set(pose_names) == set(sfm_names):
            raise RuntimeError("SfM selected identical cameras to pose-only baseline")
        for name in set(pose_names) | set(sfm_names):
            i = int(Path(name).stem)
            if sha256(depths[name]) != ledger[i]["depth_sha256"]:
                raise ValueError(f"Changed measured depth sample {i}")
        receipt.update(
            selected_rgb=300, registered_rgb=287, measured_depth_per_mesh=10,
            common_scale_depth_samples=[int(Path(n).stem) for n in shared],
            pose_cover_depth_samples=[int(Path(n).stem) for n in pose_names],
            sfm_cover_depth_samples=[int(Path(n).stem) for n in sfm_names],
            metric_scale_source="same four real measured depth maps in both meshes",
            metric_scale_m_per_colmap_unit=scale,
            metric_scale_witnesses=witnesses, sfm_coverage=sfm_stats)
        save()
        old = fuse(o3d, trimesh, views, pose_names, scale, colors, depths,
                   output / "posecover10.glb")
        new = fuse(o3d, trimesh, views, sfm_names, scale, colors, depths,
                   output / "sfmcover10.glb")
        receipt.update(
            posecover10_glb_sha256=sha256(output / "posecover10.glb"),
            sfmcover10_glb_sha256=sha256(output / "sfmcover10.glb"))
        save()
        recorded, rays = five_rays(o3d, views, scale)
        labels = [f"recorded source sample {Path(v[0]).stem}" for v in recorded]
        labels += ["free room orbit A (RGB poses)", "free room orbit B (RGB poses)"]
        panel = Image.new("RGB", (960, 5 * 270 + 30), (18, 18, 18))
        draw = ImageDraw.Draw(panel)
        for col, label in enumerate(("Original RGB", "pose-frustum 10", "real-SfM 10")):
            draw.text((col * 320 + 7, 8), label, fill="white")
        receipt["same_ray_hit_percent"] = {}
        for i, ray in enumerate(rays):
            y = 30 + 270 * i
            draw.text((6, y + 3), labels[i], fill="white")
            if i < 3:
                with Image.open(colors[recorded[i][0]]) as im:
                    panel.paste(im.convert("RGB").resize((320, 240)), (0, y + 26))
            for col, (name, mesh) in enumerate(
                    (("posecover10", old), ("sfmcover10", new)), 1):
                rendered, coverage = render_rays(o3d, mesh, ray)
                panel.paste(rendered, (col * 320, y + 26))
                receipt["same_ray_hit_percent"].setdefault(name, []).append(coverage)
        filename = "source-posecover10-vs-sfmcover10-3plus2.png"
        panel.save(output / filename)
        receipt.update(
            status="REAL_MESH_GENERATED_VISUAL_REVIEW_REQUIRED",
            image=filename,
            recorded_samples=[int(Path(v[0]).stem) for v in recorded],
            free_camera_reference="RGB-COLMAP pose-only room orbit; identical rays",
            visual_acceptance="PENDING; complete room NOT established")
    except Exception as exc:
        receipt.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["elapsed_seconds"] = round(time.time() - start)
        save()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
    else:
        if args.upstream is None or args.output is None:
            p.error("--upstream and --output required")
        run(args.upstream, args.output)
