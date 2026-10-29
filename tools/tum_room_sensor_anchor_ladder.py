"""Real TUM fr1_room input ablation: 0 -> 4 -> 8 -> 30 measured-Z frames.

All 30 camera poses, RGB images, and DA2 RGB-predicted depths stay frozen.
Only selected frames replace predicted Z with independently recorded TUM sensor Z.
This is a deliberately *intermediate* input rung, not RGB-only success.
No measured Mesh enters candidate fusion; 30-sensor is a separate control.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import shutil
import tarfile

import cv2
import numpy as np
from PIL import Image, ImageDraw

from echo4d.model import DataError, file_sha256
from echo4d.holdout import read_echo_ply
from sparse_open3d_fusion import fuse_sparse
from tum_highres_tsdf import face_components
from tum_room_da2_global_scale_replay import (
    FROZEN_ARCHIVE_SHA256, checked_replay, five_view_comparison, mesh_glb)
from tum_room_sparse_quality import ROOT, get_archive

SOURCE_RUN = 40790769632
LEVELS = (4, 8, 30)
MAX_DEPTH_M = 4.0


def anchor_indices(n: int, count: int) -> list[int]:
    """Nested time-spread selection; no RGB, sensor or Mesh scoring."""
    if not isinstance(n, int) or not isinstance(count, int) or n < 2 or not 0 <= count <= n:
        raise DataError("Invalid frame or sensor anchor count")
    if count == 0:
        return []
    if count == 1:
        return [n // 2]
    chosen = {0, n - 1}
    while len(chosen) < count:
        intervals = sorted(zip(sorted(chosen)[:-1], sorted(chosen)[1:]),
                           key=lambda pair: (-(pair[1] - pair[0]), pair[0]))
        for lo, hi in intervals:
            mid = (lo + hi) // 2
            if mid not in chosen:
                chosen.add(mid)
                break
        else:
            raise DataError("Unable to select nested sensor anchors")
    return sorted(chosen)


def verified_sensor_depths(archive: Path, association: dict, source_indices: list[int]
                           ) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    """Read only the requested official sensor PNGs; verify frozen RGB-D association hashes.

    This function is called first with the 8 anchor frames. The other 22
    sensor images are not read until the 4/8-frame candidates are fused.
    """
    if len(set(source_indices)) != len(source_indices):
        raise DataError("Duplicate source sensor indices")
    expected = {}
    for idx in source_indices:
        if not 0 <= idx < len(association["frames"]):
            raise DataError("Source index outside frozen TUM association")
        entry = association["frames"][idx]
        path = entry["depth_path"]
        if not path.startswith("depth/") or not path.endswith(".png"):
            raise DataError("Unexpected TUM sensor depth path")
        if path in expected:
            raise DataError("Two source frames reference one sensor PNG")
        expected[ROOT + "/" + path] = (idx, entry["depth_sha256"])
    depths, hashes = {}, {}
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            if member.name not in expected:
                continue
            idx, digest = expected[member.name]
            if not member.isfile() or member.size > 2_000_000:
                raise DataError("Invalid TUM depth archive member")
            stream = handle.extractfile(member)
            if stream is None:
                raise DataError("Missing TUM sensor depth bytes")
            raw = stream.read()
            import hashlib
            observed = hashlib.sha256(raw).hexdigest()
            if observed != digest:
                raise DataError("Official sensor depth PNG differs from frozen association")
            with Image.open(BytesIO(raw)) as image:
                if image.size != (640, 480):
                    raise DataError("Expected 640x480 official measured depth")
                full = np.asarray(image)
            if full.dtype != np.uint16 or full.shape != (480, 640):
                raise DataError("Expected 16-bit TUM sensor depth, not RGB or predicted Z")
            # TUM raw depth PNG is 5000 integer units per metre. Nearest exact
            # preserves measured samples; never interpolate sensor holes.
            half = cv2.resize(full, (320, 240), interpolation=cv2.INTER_NEAREST_EXACT)
            z = half.astype(np.float32) / 5000.0
            z[(z < 0.15) | (z > MAX_DEPTH_M)] = 0.0
            depths[idx] = z
            hashes[idx] = observed
            if len(depths) == len(expected):
                break
    if set(depths) != set(source_indices):
        raise DataError("Official archive missing required measured depth frames")
    return depths, hashes


def make_candidate(frozen_input: Path, manifest: dict, output: Path,
                   anchors: list[int], sensor: dict[int, np.ndarray], level: int
                   ) -> tuple[Path, dict]:
    """Copy existing DA2 RGB+Z recording; replace *only* declared sensor frames."""
    if output.exists():
        raise DataError("Refusing to overwrite candidate")
    shutil.copytree(frozen_input, output)
    record = json.loads((output / "recording.json").read_text())
    if len(record["frames"]) != len(manifest["frames"]):
        raise DataError("Frozen input changed during copy")
    used = []
    for i in anchors:
        frame = record["frames"][i]
        source_idx = frame["source_anchor_index"]
        if source_idx not in sensor:
            raise DataError("Requested sensor frame was not loaded")
        old = np.load(output / frame["depth"], allow_pickle=False)
        if old.shape != (240, 320) or old.dtype != np.float32:
            raise DataError("Unexpected frozen RGB-predicted depth shape")
        np.save(output / frame["depth"], sensor[source_idx], allow_pickle=False)
        used.append({"candidate_index": i, "source_index": source_idx,
                     "measured_valid_pixels": int(np.count_nonzero(sensor[source_idx]))})
    record["name"] = f"TUM room {level}/30 sensor-Z frame anchors plus DA2 RGB inferred Z"
    record["notes"] = ("Known metric camera poses; measured sensor Z ONLY at declared "
                       f"{level} anchors; other depths DA2 Small RGB-predicted; no prior Mesh")
    (output / "recording.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return output / "recording.json", {"level": level, "anchors": used}


def relabel_fusion(mesh_path: Path, report_path: Path, level: int) -> None:
    """Correct generic Open3D exporter provenance without modifying geometry."""
    original = b"comment Open3D sparse TSDF; measured RGB-D; known poses; no inferred depth"
    replacement = (f"comment Open3D sparse TSDF; {level}/30 measured Z frames; "
                   "remaining Z DA2 RGB-predicted; known metric poses; no prior Mesh").encode()
    raw = mesh_path.read_bytes()
    if raw.count(original) != 1:
        raise DataError("Unexpected PLY source label")
    mesh_path.write_bytes(raw.replace(original, replacement, 1))
    receipt = json.loads(report_path.read_text())
    receipt["geometry_source"] = f"{level}/30 measured sensor Z frames + DA2 RGB-predicted Z"
    receipt["measured_sensor_frames"] = level
    receipt["rgb_predicted_depth_frames"] = 30 - level
    receipt["prior_mesh_used"] = False
    receipt["mesh_sha256"] = file_sha256(mesh_path)
    report_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Unmodified real #99 source artifact")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=None,
                        help="Optional local official TUM room tgz, hash verified")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    work = args.work.resolve()
    frozen, replay, manifest = checked_replay(source)
    if output.exists() or work.exists():
        raise DataError("Output/work must be fresh; never silently reuse real data")
    output.mkdir(parents=True)
    work.mkdir(parents=True)
    if len(manifest["frames"]) != 30:
        raise DataError("Frozen DA2 30-view cohort changed")
    assoc = json.loads((replay / "source/association.json").read_text())
    expanded = set(frozen["selection"]["expanded_indices"])
    holdout = set(frozen["selection"]["holdout_indices"])
    source_ids = [f["source_anchor_index"] for f in manifest["frames"]]
    if (len(set(source_ids)) != 30 or not set(source_ids).issubset(expanded)
            or set(source_ids) & holdout):
        raise DataError("Sensor anchors must be existing train frames, never holdouts")
    archive = work / "official-room.tgz"
    archive_receipt = {}
    get_archive(archive, archive_receipt, args.archive.resolve() if args.archive else None)
    if file_sha256(archive) != FROZEN_ARCHIVE_SHA256:
        raise DataError("Official room archive SHA differs from pinned RGB source")
    eight = anchor_indices(30, 8)
    sensor, hashes = verified_sensor_depths(
        archive, assoc, [source_ids[i] for i in eight])
    previous = source / "multiview-mesh/mesh.ply"
    mesh_glb(previous, output / "zero-sensor-rgb-only.glb")
    report = {
        "experiment": "TUM fr1_room measured sensor frame ladder 0/4/8/30",
        "source_run": SOURCE_RUN,
        "source_archive_sha256": FROZEN_ARCHIVE_SHA256,
        "source_ledger_sha256": file_sha256(replay / "ledger.json"),
        "source_inferred_manifest_sha256": file_sha256(replay / "inferred/recording.json"),
        "original_rgb": "640x480 recorded TUM; 320x240 frozen RGB candidate",
        "known_metric_poses": 30,
        "learned_depth": "Depth Anything V2 Small RGB relative depth, previously calibrated with known-pose RGB geometry",
        "metric_scale": "known TUM camera translations and frozen RGB calibration",
        "no_mesh_prior": True,
        "sensor_depth_training_only": True,
        "sensor_depth_holdout_used": False,
        "source_mesh_posthoc_only": True,
        "anchor_selection": "nested largest temporal gap, independent of sensor values",
        "source_mesh_sha256": file_sha256(previous),
        "official_archive": archive_receipt,
        "variants": {"0": {"measured_frames": 0, "predicted_frames": 30,
                            "mesh_sha256": file_sha256(previous)}},
        "issue_74": "FAIL / NOT ACCEPTED; visual human review still required"}
    for level in (4, 8):
        indices = anchor_indices(30, level)
        manifest_path, receipt = make_candidate(
            replay / "inferred", manifest, output / f"input-{level}",
            indices, sensor, level)
        destination = output / f"mesh-{level}"
        fusion = fuse_sparse(manifest_path, destination, voxel_m=.04,
                             truncation_m=.12, max_depth_m=4., pixel_stride=2)
        mesh = destination / "mesh.ply"
        relabel_fusion(mesh, destination / "report.json", level)
        v, f = read_echo_ply(mesh)
        mesh_glb(mesh, output / f"hybrid-{level}-sensor.glb")
        report["variants"][str(level)] = {
            **receipt, "predicted_frames": 30 - level,
            "mesh_sha256": file_sha256(mesh), "faces": len(f),
            "components": face_components(len(v), f),
            "fusion": fusion,
            "verified_sensor_png_sha256": {str(i): hashes[i] for i in
                                           [source_ids[j] for j in indices]}}
    # Only AFTER both sparse-anchor candidates exist, read remaining sensor
    # images for the fully measured 30-frame reference.
    remaining = [i for i in source_ids if i not in sensor]
    more, more_hashes = verified_sensor_depths(archive, assoc, remaining)
    sensor.update(more)
    hashes.update(more_hashes)
    manifest_path, receipt = make_candidate(
        replay / "inferred", manifest, output / "input-30",
        anchor_indices(30, 30), sensor, 30)
    fusion = fuse_sparse(manifest_path, output / "mesh-30", voxel_m=.04,
                         truncation_m=.12, max_depth_m=4., pixel_stride=2)
    full_mesh = output / "mesh-30/mesh.ply"
    relabel_fusion(full_mesh, output / "mesh-30/report.json", 30)
    v, f = read_echo_ply(full_mesh)
    mesh_glb(full_mesh, output / "measured-30-sensor.glb")
    report["variants"]["30"] = {
        **receipt, "predicted_frames": 0, "mesh_sha256": file_sha256(full_mesh),
        "faces": len(f), "components": face_components(len(v), f),
        "fusion": fusion, "verified_sensor_png_sha256": {str(i): hashes[i] for i in source_ids}}
    # Frozen three real heldout camera RGBs and two genuinely free views.
    for level in (4, 8, 30):
        candidate = output / f"mesh-{level}/mesh.ply"
        visual_dir = output if level == 8 else output / f"comparison-{level}"
        visual_dir.mkdir(parents=True, exist_ok=True)
        views = five_view_comparison(source, replay, previous, candidate,
                                     visual_dir, frozen)
        image_path = visual_dir / "source-old-new-3plus2.png"
        with Image.open(image_path) as im:
            board = im.copy()
        draw = ImageDraw.Draw(board)
        for row, item in enumerate(views):
            y = row * 480
            draw.rectangle((640, y, 1919, y + 28), fill=(12, 12, 12))
            draw.text((650, y + 7), f"0 measured / DA2 RGB Mesh {item['old_covered_pixels']} px",
                      fill="white")
            draw.text((1290, y + 7), f"{level} measured frames / Mesh {item['new_covered_pixels']} px",
                      fill="white")
        board.save(image_path)
        report["variants"][str(level)]["five_views"] = views
        report["variants"][str(level)]["visual_png_sha256"] = file_sha256(image_path)
    report["visual_conclusion"] = (
        "UNREVIEWED; 3+2 geometry must be inspected, not inferred from face counts")
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "real_mesh_generated_visual_review_required",
                      "sensor_frame_levels": list(report["variants"]),
                      "eight_view_comparison": report["variants"]["8"]["five_views"],
                      "issue_74": report["issue_74"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
