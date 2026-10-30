"""TUM XYZ: retain six measured sensor depths; add six spatially distinct RGB AI views.

Compare exactly the old 6 measured + 6 local-calibrated AI depths with
6 measured + 12 local-calibrated AI depths. Same reference poses, same
Open3D TSDF, same five camera views. No old Mesh prior. The twelve target
sensor depths are opened ONLY after both meshes have been exported.
Stacked on PR #167 (which supplies the verified CPU Depth Anything adapter).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
from PIL import Image

from echo4d.model import DataError, Frame, rigid_matrix, safe_path
from tum_public_roundtrip import ROOT, URL
from tum_xyz_depth_budget_48_24_12 import make_recording, diverse_pose_order
from tum_xyz_open3d_320 import (
    TRAIN_INDICES, RECORDED_INDICES, export_glb, free_view, fuse, panel, render,
)
from tum_xyz_six_da6_local import (
    CPUDepthAnything, VOXEL_M, calibrate_from_six, compare_withheld,
    correct_local_bias, measured_projection, sha256,
)


def experiment(args, receipt):
    import open3d as o3d
    import trimesh

    record, imported = make_recording(args, receipt)
    entries = record.entries
    order = diverse_pose_order(entries, TRAIN_INDICES)
    measured_ids = tuple(sorted(order[:6]))
    old_ai_ids = tuple(sorted(order[6:12]))
    extra_ai_ids = tuple(sorted(order[12:18]))
    ai_ids = old_ai_ids + extra_ai_ids
    if len(set(measured_ids + ai_ids)) != 18:
        raise RuntimeError("Need eighteen distinct measured/RGB-only frames")
    if not set(measured_ids + ai_ids).isdisjoint(set(range(0, 60, 5))):
        raise RuntimeError("Reserved recorded-view holdouts overlap reconstruction")
    receipt.update(
        source_rgb_frames=60, measured_indices=measured_ids,
        old_rgb_only_indices=old_ai_ids, added_rgb_only_indices=extra_ai_ids,
        sensor_depth_used_for_reconstruction=6,
        target_sensor_depth_read=False, mesh_prior=False, gpu=False,
        reference_poses="TUM provided; not estimated from RGB",
        scale="six TUM measured depth frames + TUM reference poses",
        prediction="Depth Anything V2 Small CPU; relative depth calibrated from six measured frames",
        comparison="same 6 measured + 6 AI local correction versus same 6 measured + 12 AI local correction",
    )

    record.entries = [entries[i] for i in measured_ids]
    measured = [f for _, f, _ in record.iter_frames()]
    record.entries = entries
    if len(measured) != 6:
        raise RuntimeError("Exactly six measured sensor frames required")
    args.output.mkdir(parents=True, exist_ok=True)

    model = CPUDepthAnything()
    inferred = {}
    global_predictions = {}
    for index in ai_ids:
        spec = entries[index]
        # Deliberately open only the RGB, never the target's depth path.
        with Image.open(safe_path(record.root, spec["color"])) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        if rgb.shape != (record.camera.height, record.camera.width, 3):
            raise DataError("RGB calibration and camera shape differ")
        pose = rigid_matrix(spec["cameraToWorld"])
        relative = model.infer(rgb)
        anchors = measured_projection(measured, record.camera, pose)
        metric, calibration = calibrate_from_six(relative, anchors)
        corrected, correction = correct_local_bias(metric, anchors)
        inferred[index] = Frame(float(spec["t"]), pose, corrected, rgb).validate(record.camera)
        global_predictions[index] = metric
        Image.fromarray(np.clip(corrected * 1000, 0, 65535).astype(np.uint16)).save(
            args.output / f"ai-local-{index:02d}-mm.png"
        )
        receipt.setdefault("predictions", []).append({
            "index": index, "old_six": index in old_ai_ids,
            "calibration": calibration, "correction": correction,
        })
        print(f"CPU AI RGB {index}: {correction['valid_fraction']:.1%} local valid", flush=True)
    del model
    gc.collect()

    baseline, _ = fuse(
        o3d, measured + [inferred[i] for i in old_ai_ids],
        record.camera, pixel_stride=1, voxel=VOXEL_M,
    )
    export_glb(baseline, args.output / "tum-six-measured-six-ai-local.glb")
    expanded, _ = fuse(
        o3d, measured + [inferred[i] for i in ai_ids],
        record.camera, pixel_stride=1, voxel=VOXEL_M,
    )
    export_glb(expanded, args.output / "tum-six-measured-twelve-ai-local.glb")

    for name in ("tum-six-measured-six-ai-local.glb",
                 "tum-six-measured-twelve-ai-local.glb"):
        mesh = trimesh.load(str(args.output / name), force="mesh", process=False)
        if len(mesh.faces) < 100 or mesh.visual.kind != "vertex":
            raise RuntimeError(f"Invalid real color GLB: {name}")
        receipt.setdefault("meshes", {})[name] = {
            "faces": len(mesh.faces), "sha256": sha256(args.output / name),
        }

    # No withheld depth is loaded until BOTH independent GLBs exist.
    record.entries = [entries[i] for i in ai_ids]
    withheld = [f for _, f, _ in record.iter_frames()]
    record.entries = entries
    if len(withheld) != 12:
        raise RuntimeError("Expected twelve withheld sensor-depth labels")
    receipt["withheld_sensor_evaluation_only"] = {
        str(i): compare_withheld(global_predictions[i], inferred[i].depth, truth.depth)
        for i, truth in zip(ai_ids, withheld)
    }
    receipt["target_sensor_depth_read_for_evaluation_only_after_mesh_export"] = True

    association = json.loads((imported / "association.json").read_text())
    source_root = args.work / "dataset" / ROOT
    rows = []
    views = []
    for index in RECORDED_INDICES:
        pose = rigid_matrix(entries[index]["cameraToWorld"])
        with Image.open(source_root / association["frames"][index]["rgb_path"]) as im:
            photo = im.convert("RGB").copy()
        photo.save(args.output / f"real-original-{index:02d}-640x480.png")
        small = np.asarray(photo.resize((320, 240), Image.Resampling.LANCZOS))
        old = render(o3d, baseline, record.camera, pose)
        new = render(o3d, expanded, record.camera, pose)
        filename = f"recorded-{index:02d}-six-vs-twelve.png"
        rows.append(panel(args.output / filename, [small, old, new],
                          ["REAL RGB", "6 measured + 6 AI", "6 measured + 12 AI"]))
        views.append({"kind": "recorded", "index": index, "file": filename})
    for side, index in ((-1., 19), (1., 39)):
        pose = free_view(rigid_matrix(entries[index]["cameraToWorld"]), side)
        old = render(o3d, baseline, record.camera, pose)
        new = render(o3d, expanded, record.camera, pose)
        filename = f"free-{'left' if side < 0 else 'right'}-six-vs-twelve.png"
        rows.append(panel(args.output / filename,
                          [np.full_like(old, 20), old, new],
                          ["FREE VIEW", "6 measured + 6 AI", "6 measured + 12 AI"]))
        views.append({"kind": "free", "reference_index": index,
                      "offset_m": .22, "yaw_deg": side * 12, "file": filename})
    board = Image.new("RGB", (960, 267 * len(rows)))
    for n, row in enumerate(rows):
        board.paste(row, (0, n * 267))
    board.save(args.output / "tum-six-vs-twelve-ai-five-views.png")
    receipt.update(
        views=views, visual_verdict="PENDING human inspection of actual board",
        full_room="FAIL until real continuous freely observable space is shown",
        historical_fail="#74 and Desk 16.7779cm > 15cm remain FAIL",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    receipt = {
        "status": "running", "scene": "TUM Freiburg1 XYZ", "source": URL,
        "base": "PR #167 -> #165 -> #164 -> high-info #161",
        "commands": [],
        "run_url": f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/{os.getenv('GITHUB_RUN_ID')}"
        if os.getenv("GITHUB_RUN_ID") else None,
    }
    try:
        experiment(args, receipt)
        receipt["status"] = "TWO_REAL_MESHES_AND_FIVE_VIEWS_EXPORTED"
        return 0
    except Exception as exc:
        receipt.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "result.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps({"status": receipt["status"],
                          "error": receipt.get("error"),
                          "meshes": receipt.get("meshes")}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
