"""TUM fr1_room: 96 fixed RGB/poses, full DA2-Small CPU depth and 0/32/64/96 sensor-Z ladder.

Independent adapter stacked on #103. The 30 frozen CPU DA2 predictions are
reused; 66 missing views are inferred from original RGB with known-pose RGB
ZNCC metric anchors. No measured depth or existing Mesh enters the 0-sensor
candidate. Only after its fusion are verified TUM sensor PNGs read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from echo4d.model import Camera, DataError, file_sha256
from echo4d.holdout import read_echo_ply
from sparse_open3d_fusion import fuse_sparse
from tum_highres_tsdf import face_components
from tum_room_96view_da2_ablation import (
    validate_cohort, prepare_measured_input, replace_with_predicted, relabel)
from tum_room_da2_global_scale_replay import (
    FROZEN_ARCHIVE_SHA256, checked_replay, five_view_comparison, mesh_glb)
from tum_room_sensor_anchor_ladder import anchor_indices, verified_sensor_depths
from tum_room_sparse_quality import get_archive
from tum_room_knownpose_multiview import choose_neighbors
from tum_room_multiview_zncc import sweep_zncc
from tum_room_depthanything_small import (
    MODEL_ID, MODEL_REVISION, MODEL_SHA256, infer_relative, calibrate_relative_depth)

LEVELS = (0, 32, 64, 96)


def sensor_positions(n: int, level: int) -> list[int]:
    """Same fixed, nested time-only sensor schedule for all candidate levels."""
    if n != 96 or level not in LEVELS:
        raise DataError("Only frozen 96-view, 0/32/64/96 sensor ladder supported")
    return anchor_indices(n, level)


def build_complete_rgb_prediction(replay: Path, selected: dict, source_ids: list[int],
                                  frozen_predicted: dict[int, dict], output: Path
                                  ) -> tuple[Path, dict]:
    """Construct 96 RGB-predicted Z frames without opening sensor depth PNGs."""
    if output.exists() or len(source_ids) != 96 or len(frozen_predicted) != 30:
        raise DataError("Unexpected frozen prediction cohort or existing output")
    output.mkdir(parents=True)
    (output / "color").mkdir()
    (output / "depth").mkdir()
    assoc = json.loads((replay / "source/association.json").read_text())
    camera = Camera.from_dict(selected["intrinsics"]).scaled(2)
    K = np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy],
                  [0, 0, 1]], dtype=np.float64)
    poses = [np.asarray(f["cameraToWorld"], dtype=np.float64).reshape(4, 4, order="F")
             for f in selected["frames"]]
    cache = {}

    def rgb(i: int) -> np.ndarray:
        if i not in cache:
            idx = source_ids[i]
            entry = assoc["frames"][idx]
            path = replay / "source" / entry["rgb_path"]
            if file_sha256(path) != entry["rgb_sha256"]:
                raise DataError("Frozen original RGB checksum changed")
            with Image.open(path) as im:
                if im.size != (640, 480):
                    raise DataError("Original TUM RGB is not 640x480")
                cache[i] = np.asarray(im.convert("RGB").resize(
                    (320, 240), Image.Resampling.BILINEAR))
        return cache[i]

    frames, receipts = [], []
    for i, idx in enumerate(source_ids):
        color = rgb(i)
        frozen = frozen_predicted.get(idx)
        if frozen is not None:
            z = np.load(replay / "inferred" / frozen["depth"], allow_pickle=False)
            detail = {"status": "frozen_DA2_RGB_prediction", "source_run": 40790769632}
        else:
            neighbors = choose_neighbors(poses, i)
            if len(neighbors) < 2:
                z = np.zeros((240, 320), dtype=np.float32)
                detail = {"status": "no_two_independent_RGB_neighbors",
                          "neighbor_count": len(neighbors)}
            else:
                seed, seed_report = sweep_zncc(
                    color, [rgb(j) for j in neighbors], K, poses[i],
                    [poses[j] for j in neighbors])
                relative = infer_relative(color)
                z, fit = calibrate_relative_depth(relative, seed)
                detail = {"status": fit["status"], "neighbors": neighbors,
                          "rgb_seed_valid_pixels": seed_report["valid_pixels"],
                          "metric_fit": fit}
        if (z.shape != (240, 320) or z.dtype != np.float32
                or not np.isfinite(z).all() or (z < 0).any()):
            raise DataError("DA2 metric prediction invalid: " + str(idx))
        Image.fromarray(color).save(output / "color" / f"{i:03d}.png")
        np.save(output / "depth" / f"{i:03d}.npy", z, allow_pickle=False)
        f = selected["frames"][i]
        frames.append({"t": float(f["t"]), "cameraToWorld": f["cameraToWorld"],
                       "color": f"color/{i:03d}.png", "depth": f"depth/{i:03d}.npy",
                       "source_anchor_index": idx})
        receipts.append({"train_position": i, "source_index": idx,
                         "predicted_valid_pixels": int(np.count_nonzero(z)), **detail})
        print(f"RGB_DA2_VIEW {i+1}/96 source={idx} valid={np.count_nonzero(z)} "
              f"status={detail['status']}", flush=True)
    manifest = {"version": "echo-dataset-1", "source": "recorded", "units": "meters",
                "name": "TUM fr1_room 96 RGB-predicted DA2 Small depth maps",
                "duration": float(frames[-1]["t"]-frames[0]["t"]),
                "depth_scale": 1, "intrinsics": camera.to_dict(), "frames": frames,
                "notes": "Recorded RGB/known metric poses, DA2 Small relative depth + "
                         "known-pose RGB ZNCC scale; 0 measured sensor Z, no prior Mesh"}
    path = output / "recording.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return path, {"receipts": receipts, "new_cpu_inference_frames": 66,
                  "frozen_prediction_frames": 30,
                  "predicted_nonempty_frames": sum(r["predicted_valid_pixels"] > 0
                                                  for r in receipts)}


def render_against_measured(source: Path, replay: Path, frozen: dict,
                            measured_mesh: Path, mesh: Path, output: Path,
                            measured_count: int) -> list[dict]:
    output.mkdir(parents=True)
    views = five_view_comparison(source, replay, measured_mesh, mesh, output, frozen)
    board = output / "source-old-new-3plus2.png"
    with Image.open(board) as im:
        image = im.copy()
    draw = ImageDraw.Draw(image)
    for row in range(5):
        y = row * 480
        draw.rectangle((640, y, 1919, y+28), fill=(12, 12, 12))
        draw.text((650, y+7), "96 measured sensor Z / control Mesh", fill="white")
        draw.text((1290, y+7),
                  f"{measured_count} sensor + {96-measured_count} DA2 / candidate Mesh",
                  fill="white")
    image.save(board)
    return views


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="Frozen real #99 artifact")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--archive", type=Path, help="Optional SHA-verified TUM archive")
    args = p.parse_args()
    source, work, output = args.source.resolve(), args.work.resolve(), args.output.resolve()
    if work.exists() or output.exists():
        raise DataError("Use fresh work/output directories")
    frozen, replay, inferred = checked_replay(source)
    selected, source_ids, frozen_by_id = validate_cohort(frozen, replay, inferred)
    work.mkdir(parents=True)
    output.mkdir(parents=True)
    report = {"experiment": "TUM fr1_room 96 fixed views: 0/32/64/96 sensor-Z",
              "source_run": 40790769632,
              "source_archive_sha256": FROZEN_ARCHIVE_SHA256,
              "source_ledger_sha256": file_sha256(replay / "ledger.json"),
              "source_inferred_sha256": file_sha256(replay / "inferred/recording.json"),
              "source_indices": source_ids, "original_rgb": "96 real 640x480, inference 320x240",
              "metric_poses": 96, "prior_mesh_used": False,
              "model": MODEL_ID, "revision": MODEL_REVISION,
              "checkpoint_sha256": MODEL_SHA256, "model_license": "Apache-2.0 Small",
              "metric_scale": "known TUM metric poses + independent RGB ZNCC matches",
              "sensor_Z_used_in_zero_candidate": False,
              "fusion": {"voxel_m": .04, "truncation_m": .12,
                         "max_depth_m": 4., "pixel_stride": 2},
              "levels": {}, "issue_74": "FAIL / NOT ACCEPTED"}

    predicted_manifest, predictions = build_complete_rgb_prediction(
        replay, selected, source_ids, frozen_by_id, work / "predicted-96")
    report["prediction"] = predictions
    # Critical separation: 0-sensor candidate is fused BEFORE the sensor
    # archive is downloaded/opened. No prior Mesh enters this reconstruction.
    candidates = {}
    def fuse_level(level: int, manifest: Path):
        folder = output / f"mesh-sensor-{level}"
        try:
            receipt = fuse_sparse(manifest, folder, voxel_m=.04,
                                  truncation_m=.12, max_depth_m=4., pixel_stride=2)
            mesh = folder / "mesh.ply"
            receipt = relabel(mesh, folder / "report.json", 96-level)
            vertices, faces = read_echo_ply(mesh)
            glb = output / f"sensor-{level}-da2-{96-level}.glb"
            mesh_glb(mesh, glb)
            report["levels"][str(level)] = {
                "status": "mesh_generated_visual_review_required",
                "measured_frames": level, "predicted_frames": 96-level,
                "faces": len(faces), "components": face_components(len(vertices), faces),
                "mesh_sha256": file_sha256(mesh), "glb_sha256": file_sha256(glb),
                "fusion": receipt}
            candidates[level] = mesh
        except Exception as exc:
            report["levels"][str(level)] = {
                "status": "FAIL", "error": type(exc).__name__ + ": " + str(exc),
                "measured_frames": level, "predicted_frames": 96-level}
        (output / "report.json").write_text(json.dumps(
            report, indent=2, allow_nan=False) + "\n")

    fuse_level(0, predicted_manifest)
    # Measured sensor pixels are accessed only after the pure RGB candidate.
    archive = work / "official-room.tgz"
    archive_receipt = {}
    get_archive(archive, archive_receipt, args.archive.resolve() if args.archive else None)
    if file_sha256(archive) != FROZEN_ARCHIVE_SHA256:
        raise DataError("TUM official archive SHA mismatch")
    assoc = json.loads((replay / "source/association.json").read_text())
    sensor, sensor_hashes = verified_sensor_depths(archive, assoc, source_ids)
    measured = prepare_measured_input(
        replay, selected, source_ids, sensor, work / "measured-96")
    report["archive"] = archive_receipt
    report["sensor_png_sha256"] = {str(i): sensor_hashes[i] for i in source_ids}
    prediction_by_id = {f["source_anchor_index"]: f
                        for f in json.loads(predicted_manifest.read_text())["frames"]}
    for level in (32, 64, 96):
        if level == 96:
            candidate_manifest = measured
        else:
            sensor_selected = {source_ids[j] for j in sensor_positions(96, level)}
            predict_selected = [idx for idx in source_ids if idx not in sensor_selected]
            candidate_manifest = replace_with_predicted(
                measured.parent, predicted_manifest.parent, prediction_by_id,
                predict_selected, work / f"hybrid-{level}")
        fuse_level(level, candidate_manifest)
    if 96 in candidates:
        for level in (0, 32, 64):
            if level not in candidates:
                continue
            visual = output / f"visual-sensor-{level}"
            try:
                views = render_against_measured(
                    source, replay, frozen, candidates[96], candidates[level],
                    visual, level)
                report["levels"][str(level)]["five_views"] = views
                report["levels"][str(level)]["visual_sha256"] = file_sha256(
                    visual / "source-old-new-3plus2.png")
            except Exception as exc:
                report["levels"][str(level)]["visual_error"] = str(exc)
    report["visual_conclusion"] = "UNREVIEWED; geometry and 3+2 views require inspection"
    report["independent_surface_truth"] = False
    (output / "report.json").write_text(json.dumps(
        report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"levels": {k: v["status"] for k, v in report["levels"].items()},
                      "predicted_nonempty_frames": predictions["predicted_nonempty_frames"],
                      "issue_74": report["issue_74"]}, ensure_ascii=False))
    if len(candidates) < 2 or 96 not in candidates:
        raise DataError("Too few real Mesh levels produced")


if __name__ == "__main__":
    main()
