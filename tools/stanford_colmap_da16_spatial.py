"""Nested Stanford Lounge RGB-depth expansion: eight measured + sixteen CPU DA2.

Uses the existing #159 RGB-COLMAP pose, sparse metric calibration, CPU model,
and TSDF implementation. Six extra RGB predictions are added to the previous
8+8 depth budget; the old GLB is for visual comparison ONLY, not a prior.
No held-out target sensor depth is used before fusion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import stanford_colmap_da8_spatial as base
from stanford_colmap_nested_depth import render_rays
from stanford_colmap_pose_cover10 import five_rays, sha256
from stanford_colmap_metric300 import read_model
from stanford_colmap_eight_measured_control import old_mesh_to_open3d

# Two frozen DA2 targets + fourteen additional, instead of six in #159.
ADDITIONAL_RGB_PREDICTIONS = 14
EXPECTED_TOTAL_PREDICTIONS = 16


def self_test():
    assert base.EXTRA_PREDICTIONS == 6, "Upstream DA8 selector unexpectedly changed"
    assert ADDITIONAL_RGB_PREDICTIONS - base.EXTRA_PREDICTIONS == 8
    base.self_test()
    print("PASS: nested 8 measured + 8 learned -> 8 measured + 16 learned")


def run(upstream: Path, control: Path, da2: Path, da8: Path, output: Path):
    import open3d as o3d
    import trimesh

    upstream, control, da2, da8, output = (
        Path(p).resolve() for p in (upstream, control, da2, da8, output))
    old_receipt = json.loads((da8 / "receipt.json").read_text())
    old_glb = da8 / "measured8_da8.glb"
    if (old_receipt.get("measured_depth_maps"),
            old_receipt.get("inferred_depth_maps"),
            old_receipt.get("prior_mesh")) != (8, 8, False):
        raise ValueError("Frozen 8+8 comparison provenance changed")
    if sha256(old_glb) != old_receipt["new_glb_sha256"]:
        raise ValueError("Frozen 8+8 comparison GLB hash mismatch")

    if base.EXTRA_PREDICTIONS != 6:
        raise ValueError("Base DA8 selection algorithm changed")
    # The initial six new targets must stay nested. Only the depth budget
    # changes: the original candidate shortlist and greedy ranking are reused.
    base.EXTRA_PREDICTIONS = ADDITIONAL_RGB_PREDICTIONS
    try:
        base.run(upstream, control, da2, output)
    finally:
        base.EXTRA_PREDICTIONS = 6

    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    try:
        old_extra = old_receipt["new_predicted_samples"]
        new_extra = receipt["new_predicted_samples"]
        if new_extra[:len(old_extra)] != old_extra:
            raise ValueError("Non-nested new RGB prediction order: cannot claim controlled expansion")
        if len(new_extra) != ADDITIONAL_RGB_PREDICTIONS or receipt["inferred_depth_maps"] != 16:
            raise ValueError("Did not generate all sixteen RGB-predicted depths")
        if receipt["measured_depth_samples"] != old_receipt["measured_depth_samples"]:
            raise ValueError("Eight sensor-depth inputs differ from comparison")

        generated = output / "measured8_da8.glb"
        new_glb = output / "measured8_da16.glb"
        generated.rename(new_glb)
        receipt.update(
            target_inferred_depth_maps=16,
            previous_inferred_depth_maps=8,
            added_rgb_only_depth_maps=8,
            nested_first_eight_prediction_samples=[
                int(x["target"].split(".")[0]) for x in receipt["inferred"][:8]],
            old_da8_glb_sha256=sha256(old_glb),
            new_glb_sha256=sha256(new_glb),
            new_glb=new_glb.name,
            old_glb_used_as_reconstruction_prior=False,
            high_information_baseline="TUM XYZ measured RGB-D #161 is a separate scene, not used in this Stanford A/B",
            full_room_acceptance="FAIL",
            visual_acceptance="PENDING real five-view visual review; not a complete room",
        )
        parent = json.loads((upstream / "receipt.json").read_text())
        views = read_model(upstream / parent["model"])
        recorded, rays = five_rays(o3d, views, receipt["metric_scale_m_per_colmap_unit"])
        old = old_mesh_to_open3d(o3d, trimesh, old_glb)
        new = old_mesh_to_open3d(o3d, trimesh, new_glb)
        panel = Image.new("RGB", (960, 5 * 270 + 30), (18, 18, 18))
        draw = ImageDraw.Draw(panel)
        for col, title in enumerate(("Original RGB", "8 measured + 8 DA2", "8 measured + 16 DA2")):
            draw.text((col * 320 + 5, 8), title, fill="white")
        hits = {"old_da8": [], "new_da16": []}
        for i, ray in enumerate(rays):
            y = 30 + i * 270
            draw.text((5, y + 3),
                      f"recorded {Path(recorded[i][0]).stem}" if i < 3 else f"free orbit {i-2}",
                      fill="white")
            if i < 3:
                src = output / f"source-native-{Path(recorded[i][0]).stem}.png"
                with Image.open(src) as im:
                    panel.paste(im.convert("RGB").resize((320, 240)), (0, y + 26))
            for col, (label, mesh) in enumerate((("old_da8", old), ("new_da16", new)), 1):
                picture, hit = render_rays(o3d, mesh, ray)
                panel.paste(picture, (col * 320, y + 26))
                hits[label].append(hit)
        name = "source-da8-vs-da16-3plus2.png"
        panel.save(output / name)
        receipt.update(image=name, same_ray_hit_percent=hits,
                       status="REAL_MESH_GENERATED_VISUAL_REVIEW_REQUIRED")
    except Exception as exc:
        receipt.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("upstream", "control", "da2", "da8", "output"):
        parser.add_argument("--" + flag, type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not all((args.upstream, args.control, args.da2, args.da8, args.output)):
            parser.error("--upstream --control --da2 --da8 --output required")
        run(args.upstream, args.control, args.da2, args.da8, args.output)
