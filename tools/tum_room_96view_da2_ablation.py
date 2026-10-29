"""Real TUM fr1_room 96-view sensor->DA2 substitution on fixed cameras.

All 96 original RGB images, measured metric poses and intrinsics are fixed.
The 30 DA2 Small predictions are replayed from a SHA-verified RGB-only run;
0/8/16/30 of these replace measured Z at the same source frame identity.
The other 66 views retain measured Z. No existing Mesh enters any fusion.
This is NOT a 96-view RGB-only experiment: 66 sensor views remain at level 30.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from echo4d.model import Camera, DataError, file_sha256
from echo4d.holdout import read_echo_ply
from sparse_open3d_fusion import fuse_sparse
from tum_highres_tsdf import face_components
from tum_room_da2_global_scale_replay import (
    FROZEN_ARCHIVE_SHA256, checked_replay, five_view_comparison, mesh_glb)
from tum_room_sensor_anchor_ladder import anchor_indices, verified_sensor_depths
from tum_room_sparse_quality import get_archive, select_training

SOURCE_RUN = 40790769632
LEVELS = (0, 8, 16, 30)


def validate_cohort(frozen: dict, replay: Path, inferred: dict) -> tuple[dict, list[int], dict[int, dict]]:
    """Match all predicted depths to the same 96 high-information frame IDs/poses."""
    original = json.loads((replay / "source/recording.json").read_text())
    _, selected, receipt = select_training(original)
    source_ids = receipt["expanded_indices"]
    if (source_ids != frozen["selection"]["expanded_indices"]
            or len(source_ids) != 96 or len(inferred["frames"]) != 30):
        raise DataError("96 measured and 30 predicted cohorts differ")
    if (selected["intrinsics"]["width"] != 640
            or selected["intrinsics"]["height"] != 480):
        raise DataError("Measured RGB source must be original 640x480")
    expected_camera = Camera.from_dict(selected["intrinsics"]).scaled(2)
    if expected_camera.to_dict() != inferred["intrinsics"]:
        raise DataError("Predicted and measured intrinsics differ")
    by_id = {}
    for f in inferred["frames"]:
        idx = f["source_anchor_index"]
        if idx in by_id or idx not in source_ids or idx in receipt["holdout_indices"]:
            raise DataError("Predicted frame duplicates or leaks outside 96 train frames")
        source = original["frames"][idx]
        a = np.asarray(source["cameraToWorld"], dtype=np.float64)
        b = np.asarray(f["cameraToWorld"], dtype=np.float64)
        if a.shape != (16,) or b.shape != (16,) or not np.allclose(a, b, atol=1e-7):
            raise DataError("RGB predicted depth camera pose changed")
        by_id[idx] = f
    return selected, source_ids, by_id


def prepare_measured_input(replay: Path, selected: dict, source_ids: list[int],
                           sensor: dict[int, np.ndarray], output: Path) -> Path:
    """Materialize a 320x240 96-frame measured control, using only training IDs."""
    output.mkdir(parents=True, exist_ok=False)
    (output / "color").mkdir()
    (output / "depth").mkdir()
    assoc = json.loads((replay / "source/association.json").read_text())
    camera = Camera.from_dict(selected["intrinsics"]).scaled(2)
    frames = []
    for j, idx in enumerate(source_ids):
        entry = assoc["frames"][idx]
        rgb_path = replay / "source" / entry["rgb_path"]
        if file_sha256(rgb_path) != entry["rgb_sha256"]:
            raise DataError("Frozen 640x480 RGB bytes differ from TUM association")
        with Image.open(rgb_path) as im:
            if im.size != (640, 480):
                raise DataError("Unexpected original TUM RGB dimensions")
            rgb = im.convert("RGB").resize((320, 240), Image.Resampling.BILINEAR)
            rgb.save(output / "color" / f"{j:03d}.png")
        z = sensor[idx]
        if z.shape != (240, 320) or z.dtype != np.float32:
            raise DataError("Measured depth is not 320x240 float32 metres")
        np.save(output / "depth" / f"{j:03d}.npy", z, allow_pickle=False)
        f = selected["frames"][j]
        frames.append({"t": float(f["t"]), "cameraToWorld": f["cameraToWorld"],
                       "color": f"color/{j:03d}.png", "depth": f"depth/{j:03d}.npy",
                       "source_anchor_index": idx})
    manifest = {"version": "echo-dataset-1", "source": "recorded", "units": "meters",
                "name": "TUM fr1_room 96 fixed views / 96 sensor depth",
                "duration": float(frames[-1]["t"] - frames[0]["t"]),
                "depth_scale": 1, "intrinsics": camera.to_dict(), "frames": frames,
                "notes": "96 real metric poses; measured sensor depth at all frames; no prior Mesh"}
    path = output / "recording.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return path


def replace_with_predicted(control: Path, inferred_root: Path,
                           predicted: dict[int, dict], chosen: list[int], target: Path) -> Path:
    """Copy control, replacing exactly chosen sensor Z arrays, never changing RGB/pose."""
    if len(set(chosen)) != len(chosen) or not set(chosen).issubset(predicted):
        raise DataError("Invalid or duplicate RGB-predicted replacement IDs")
    shutil.copytree(control, target)
    record_path = target / "recording.json"
    record = json.loads(record_path.read_text())
    replaced = set()
    for frame in record["frames"]:
        idx = frame["source_anchor_index"]
        if idx not in chosen:
            continue
        z = np.load(inferred_root / predicted[idx]["depth"], allow_pickle=False)
        if z.shape != (240, 320) or z.dtype != np.float32 or not np.isfinite(z).all() or (z < 0).any():
            raise DataError("Frozen RGB-predicted depth is not finite metric Z")
        np.save(target / frame["depth"], z, allow_pickle=False)
        replaced.add(idx)
    if replaced != set(chosen):
        raise DataError("Not all declared RGB predictions were substituted")
    record["name"] = f"TUM room 96 views / {96-len(chosen)} sensor + {len(chosen)} DA2 predicted Z"
    record["notes"] = (f"Known metric poses; {96-len(chosen)} measured sensor Z, "
                       f"{len(chosen)} frozen RGB-predicted DA2 Z; no Mesh prior")
    record_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record_path


def relabel(mesh: Path, receipt: Path, predicted_count: int) -> dict:
    old = b"comment Open3D sparse TSDF; measured RGB-D; known poses; no inferred depth"
    new = (f"comment Open3D TSDF; {96-predicted_count}/96 measured sensor Z, "
           f"{predicted_count}/96 RGB-predicted DA2 Z; metric poses; no prior Mesh").encode()
    raw = mesh.read_bytes()
    if raw.count(old) != 1:
        raise DataError("Cannot truthfully label Open3D Mesh provenance")
    mesh.write_bytes(raw.replace(old, new, 1))
    result = json.loads(receipt.read_text())
    result.update({"geometry_source": f"{96-predicted_count} measured sensor + {predicted_count} DA2 RGB-predicted Z frames",
                   "prior_mesh_used": False, "measured_sensor_frames": 96-predicted_count,
                   "predicted_depth_frames": predicted_count, "mesh_sha256": file_sha256(mesh)})
    receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Real immutable #99 artifact")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="Optional local verified official TUM archive")
    args = parser.parse_args()
    frozen, replay, inferred = checked_replay(args.source.resolve())
    selected, source_ids, predicted = validate_cohort(frozen, replay, inferred)
    output, work = args.output.resolve(), args.work.resolve()
    if output.exists() or work.exists():
        raise DataError("Use fresh output/work directories; no silent overwrite")
    output.mkdir(parents=True)
    work.mkdir(parents=True)
    archive = work / "official-room.tgz"
    archive_receipt = {}
    get_archive(archive, archive_receipt, args.archive.resolve() if args.archive else None)
    if file_sha256(archive) != FROZEN_ARCHIVE_SHA256:
        raise DataError("Official TUM room archive hash differs from frozen source")
    sensor, hashes = verified_sensor_depths(
        archive, json.loads((replay / "source/association.json").read_text()), source_ids)
    base = prepare_measured_input(replay, selected, source_ids, sensor, work / "input-0")
    predicted_ids = [f["source_anchor_index"] for f in inferred["frames"]]
    report = {"experiment": "TUM fr1_room 96 fixed RGB/poses: measured Z -> frozen DA2 predicted Z",
              "source_run": SOURCE_RUN, "source_archive_sha256": FROZEN_ARCHIVE_SHA256,
              "source_ledger_sha256": file_sha256(replay / "ledger.json"),
              "source_inferred_manifest_sha256": file_sha256(replay / "inferred/recording.json"),
              "original_rgb": "96 original 640x480 images downsampled to 320x240 for identical fusion",
              "known_metric_poses": 96, "same_96_frame_ids": source_ids,
              "no_mesh_prior": True, "sensor_holdout_used_for_fusion": False,
              "metric_scale": "TUM known metric poses; DA2 predictions pre-calibrated from known-pose RGB",
              "predicted_source": "frozen DA2 Small CPU run #99; model NOT rerun here",
              "official_archive": archive_receipt,
              "sensor_png_sha256": {str(i): hashes[i] for i in source_ids},
              "fusion": {"voxel_m": .04, "truncation_m": .12, "max_depth_m": 4., "pixel_stride": 2},
              "levels": {}, "issue_74": "FAIL / NOT ACCEPTED; requires visual inspection"}
    previous_mesh = None
    for count in LEVELS:
        chosen = [predicted_ids[i] for i in anchor_indices(30, count)]
        manifest = base if count == 0 else replace_with_predicted(
            base.parent, replay / "inferred", predicted, chosen, work / f"input-{count}")
        mesh_folder = output / f"mesh-{count}"
        fusion = fuse_sparse(manifest, mesh_folder, voxel_m=.04,
                             truncation_m=.12, max_depth_m=4., pixel_stride=2)
        mesh = mesh_folder / "mesh.ply"
        fusion = relabel(mesh, mesh_folder / "report.json", count)
        vertices, faces = read_echo_ply(mesh)
        mesh_glb(mesh, output / f"sensor-{96-count}-da2-{count}.glb")
        report["levels"][str(count)] = {
            "measured_frames": 96-count, "predicted_frames": count,
            "predicted_source_indices": chosen, "mesh_sha256": file_sha256(mesh),
            "faces": len(faces), "components": face_components(len(vertices), faces),
            "fusion": fusion, "glb_sha256": file_sha256(output / f"sensor-{96-count}-da2-{count}.glb")}
        if count == 0:
            previous_mesh = mesh
        else:
            visual = output / f"visual-{count}"
            visual.mkdir()
            views = five_view_comparison(args.source.resolve(), replay, previous_mesh,
                                         mesh, visual, frozen)
            board = visual / "source-old-new-3plus2.png"
            with Image.open(board) as im:
                img = im.copy()
            draw = ImageDraw.Draw(img)
            for row in range(5):
                y = row * 480
                draw.rectangle((640, y, 1919, y+28), fill=(12,12,12))
                draw.text((650, y+7), "96 sensor Z frames / measured Mesh", fill="white")
                draw.text((1290, y+7), f"{96-count} sensor + {count} DA2 Z / new Mesh", fill="white")
            img.save(board)
            report["levels"][str(count)]["five_views"] = views
            report["levels"][str(count)]["visual_sha256"] = file_sha256(board)
    report["visual_conclusion"] = "UNREVIEWED: real 3+2 geometry needs human inspection; coverage is not accuracy"
    report["independent_surface_truth"] = False
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "real_mesh_generated_visual_review_required",
                      "levels": {k: {"faces": v["faces"], "components": v["components"]}
                                 for k, v in report["levels"].items()},
                      "issue_74": report["issue_74"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
