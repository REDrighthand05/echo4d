"""CPU multi-view plane sweep on real TUM fr1_room known-pose RGB, no sensor Z.

This is a direct replacement for the one-horizontal-pair SGBM depth producer.
It reuses the existing room importer, TSDF, 3+2 renderer, and frozen controls.
No new reconstruction planner, model weights, or synthetic geometry completion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

from echo4d.holdout import read_echo_ply
from echo4d.model import Camera, DataError, file_sha256, safe_path
from sparse_open3d_fusion import fuse_sparse
from tum_highres_tsdf import face_components, invoke
from tum_room_sparse_quality import ROOT, get_archive, extract_rgbd, select_training
from tum_room_rgb_stereo_ablation import (
    compare_five, preserve_replay_inputs, correct_inferred_mesh_provenance)

ANCHORS = 32
NEIGHBORS = 4
DEPTHS_M = np.geomspace(.45, 4., 56)


def choose_neighbors(poses: list[np.ndarray], anchor: int, limit: int = NEIGHBORS) -> list[int]:
    """Select real training cameras with overlapping orientation and lateral parallax."""
    a = poses[anchor]
    candidates = []
    for j, b in enumerate(poses):
        if j == anchor:
            continue
        t = b[:3, 3] - a[:3, 3]
        baseline = float(np.linalg.norm(t))
        lateral = float(np.linalg.norm(t - np.dot(t, a[:3, 2]) * a[:3, 2]))
        facing = float(np.dot(a[:3, 2], b[:3, 2]))
        if not (.055 <= baseline <= .45 and lateral >= .045 and
                facing >= math.cos(math.radians(19))):
            continue
        score = abs(baseline - .14) + .35 * (1 - facing) + .0002 * abs(j-anchor)
        candidates.append((score, j))
    candidates.sort()
    return [j for _, j in candidates[:limit]]


def sweep_depth(reference: np.ndarray, sources: list[np.ndarray],
                K: np.ndarray, reference_pose: np.ndarray,
                source_poses: list[np.ndarray], depths: np.ndarray = DEPTHS_M
                ) -> tuple[np.ndarray, dict]:
    """Warp >=2 independent RGB views over metric fronto-parallel planes.

    Homography maps reference pixel to source pixel:
      K (R_source^T R_ref + R_source^T(t_ref-t_source) e3^T / Z) K^-1.
    Missing/untextured/unsupported pixels stay zero, not hallucinated.
    """
    h, w = reference.shape[:2]
    if (reference.dtype != np.uint8 or reference.shape != (h, w, 3)
            or len(sources) != len(source_poses) or len(sources) < 2
            or any(s.shape != reference.shape or s.dtype != np.uint8 for s in sources)
            or K.shape != (3, 3) or not np.isfinite(K).all()
            or np.any(np.asarray(depths) <= 0)):
        raise DataError("Plane sweep requires calibrated RGB, >=2 sources and positive depths")
    ref = cv2.GaussianBlur(cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    texture = cv2.boxFilter(
        np.abs(cv2.Sobel(ref, cv2.CV_32F, 1, 0, ksize=3)) +
        np.abs(cv2.Sobel(ref, cv2.CV_32F, 0, 1, ksize=3)),
        -1, (7, 7), normalize=True)
    grays = [cv2.GaussianBlur(cv2.cvtColor(s, cv2.COLOR_RGB2GRAY), (3, 3), 0)
             for s in sources]
    inv_K = np.linalg.inv(K)
    transforms = []
    for pose in source_poses:
        R = pose[:3, :3].T @ reference_pose[:3, :3]
        t = pose[:3, :3].T @ (reference_pose[:3, 3] - pose[:3, 3])
        transforms.append((R, t))
    best = np.full((h, w), np.inf, dtype=np.float32)
    result = np.zeros((h, w), dtype=np.float32)
    source_mask = np.full((h, w), 255, dtype=np.uint8)
    for z in depths:
        costs = []
        for gray, (R, t) in zip(grays, transforms):
            M = R.copy()
            M[:, 2] += t / float(z)
            H = K @ M @ inv_K
            if abs(np.linalg.det(H)) < 1e-8:
                continue
            flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
            warped = cv2.warpPerspective(gray, H, (w, h), flags=flags)
            valid = cv2.warpPerspective(
                source_mask, H, (w, h),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
            difference = cv2.boxFilter(
                cv2.absdiff(ref, warped).astype(np.float32), -1, (5, 5),
                normalize=True)
            costs.append(np.where(valid, difference, np.inf))
        if len(costs) < 2:
            continue
        stack = np.stack(costs)
        # Two different cameras must agree; not a single best-pair score.
        score = .5 * (np.min(stack, axis=0) +
                       np.partition(stack, 1, axis=0)[1])
        better = score < best
        best[better] = score[better]
        result[better] = float(z)
    valid = (best < 25.) & (texture > 9.) & np.isfinite(best)
    result[~valid] = 0
    return result, {
        "valid_pixels": int(np.count_nonzero(valid)),
        "median_photometric_cost": float(np.median(best[valid])) if valid.any() else None,
        "depth_source": "known-pose RGB multi-view plane sweep; no sensor Z",
        "depth_planes_m": len(depths),
        "source_cameras": len(sources)}


def build_manifest(imported: Path, target: Path, report: dict) -> Path:
    raw = json.loads((imported / "recording.json").read_text())
    _, selected, receipt = select_training(raw)
    report["selection"] = receipt
    camera = Camera.from_dict(raw["intrinsics"]).scaled(2)
    if (camera.width, camera.height) != (320, 240):
        raise DataError("Expected original 640x480 RGB, inference at 320x240")
    frames = selected["frames"]
    poses = [np.asarray(f["cameraToWorld"], dtype=np.float64).reshape(4, 4, order="F")
             for f in frames]
    eligible = [(i, choose_neighbors(poses, i)) for i in range(len(frames))]
    eligible = [(i, js) for i, js in eligible if len(js) >= 2]
    report["eligible_anchors"] = len(eligible)
    if len(eligible) < 8:
        raise DataError("Too few multi-view overlapping real RGB cameras")
    samples = np.linspace(0, len(eligible)-1, min(ANCHORS, len(eligible)))
    chosen = [eligible[int(round(x))] for x in samples]
    target.mkdir(parents=True, exist_ok=False)
    (target / "color").mkdir()
    (target / "depth").mkdir()
    K = np.array([[camera.fx, 0, camera.cx],
                  [0, camera.fy, camera.cy], [0, 0, 1]], dtype=np.float64)
    cache = {}
    def rgb(i):
        if i not in cache:
            with Image.open(safe_path(imported, frames[i]["color"])) as im:
                if im.size != (640, 480):
                    raise DataError("Original room RGB size mismatch")
                cache[i] = np.asarray(
                    im.convert("RGB").resize((320, 240), Image.Resampling.BILINEAR))
        return cache[i]
    output, receipts = [], []
    for i, neighbors in chosen:
        z, detail = sweep_depth(rgb(i), [rgb(j) for j in neighbors],
                                K, poses[i], [poses[j] for j in neighbors])
        receipts.append({"anchor": i, "neighbors": neighbors,
                         "source_anchor_index": receipt["expanded_indices"][i],
                         "source_neighbor_indices": [receipt["expanded_indices"][j] for j in neighbors],
                         **detail})
        if detail["valid_pixels"] < 150:
            continue
        k = len(output)
        Image.fromarray(rgb(i)).save(target / "color" / f"{k:03d}.png")
        np.save(target / "depth" / f"{k:03d}.npy", z, allow_pickle=False)
        output.append({"t": float(frames[i]["t"]-frames[0]["t"]),
                       "cameraToWorld": frames[i]["cameraToWorld"],
                       "color": f"color/{k:03d}.png",
                       "depth": f"depth/{k:03d}.npy",
                       "source_anchor_index": receipt["expanded_indices"][i]})
    report["inference_receipts"] = receipts
    report["inferred_depth_frames"] = len(output)
    if len(output) < 8:
        raise DataError("Multi-view RGB generated fewer than 8 depth maps")
    manifest = {
        "version": "echo-dataset-1", "source": "recorded", "units": "meters",
        "name": "TUM room known-pose RGB multi-view inferred Z",
        "duration": max(.001, float(output[-1]["t"])),
        "depth_scale": 1, "intrinsics": camera.to_dict(), "frames": output,
        "notes": "Z from 2+ RGB reprojections and known metric camera poses; no sensor depth"}
    path = target / "recording.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return path


def run(args, report):
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    archive = work / "room.tgz"
    get_archive(archive, report, args.archive.resolve() if args.archive else None)
    report["source_archive_sha256"] = file_sha256(archive)
    report["extraction"] = extract_rgbd(archive, work / "source")
    imported = work / "room-import"
    invoke("import-source-associations", [
        sys.executable, "-m", "echo4d", "import-tum", str(work/"source"/ROOT),
        "--output", str(imported), "--intrinsics", "525", "525", "319.5", "239.5",
        "--stride", "8", "--limit", "180", "--downsample", "1", "--world-up", "z"],
        report, timeout=900)
    high = json.loads((args.baseline/"room-sparse-quality-report.json").read_text())
    high_mesh = args.baseline/"room-96-fine-wide"/"mesh.ply"
    if (high.get("archive_observed_sha256") != report["source_archive_sha256"]
            or file_sha256(high_mesh) != high["results"]["room-96-fine-wide"]["mesh_sha256"]):
        raise DataError("Measured high-info room control source/Mesh mismatch")
    manifest = build_manifest(imported, work/"inferred", report)
    if report["selection"]["expanded_indices"] != high["training_selection"]["expanded_indices"]:
        raise DataError("Training RGB identities differ from measured room control")
    report["replay_inputs"] = preserve_replay_inputs(
        imported, manifest.parent, args.baseline, args.report_dir/"replay-inputs",
        report["selection"])
    report["inferred_manifest_sha256"] = file_sha256(manifest)
    report["comparison_limits"] = {
        "highinfo": "96 measured-Z frames, 25mm voxels, stride 1",
        "previous_rgb_stereo": "41 inferred-Z frames, 40mm voxels, stride 2",
        "new_multiview": f"{report['inferred_depth_frames']} inferred-Z frames, 40mm voxels, stride 2",
        "not_single_variable_ablation": True}
    fuse_sparse(manifest, args.report_dir/"multiview-mesh",
                voxel_m=.04, truncation_m=.12, max_depth_m=4., pixel_stride=2)
    report["fusion"] = correct_inferred_mesh_provenance(args.report_dir/"multiview-mesh")
    # Correct inherited SGBM-only label to the actual multi-view algorithm.
    ply = args.report_dir/"multiview-mesh"/"mesh.ply"
    old = b"RGB-stereo inferred depth; known metric poses; no sensor Z"
    new = b"RGB multi-view plane-sweep inferred Z; known metric poses; no sensor Z"
    data = ply.read_bytes()
    if data.count(old) != 1:
        raise DataError("Expected exact inherited PLY source label")
    ply.write_bytes(data.replace(old, new, 1))
    report["fusion"]["mesh_sha256"] = file_sha256(ply)
    fusion_report = args.report_dir/"multiview-mesh"/"report.json"
    fusion_receipt = json.loads(fusion_report.read_text())
    fusion_receipt["geometry_source"] = "known-pose multi-view RGB plane sweep, no sensor Z"
    fusion_receipt["mesh_sha256"] = file_sha256(ply)
    fusion_report.write_text(json.dumps(fusion_receipt, indent=2)+"\n")
    v, f = read_echo_ply(ply)
    report["new_mesh"] = {"sha256": file_sha256(ply), "vertices": len(v),
                          "faces": len(f), "connectivity": face_components(len(v), f)}
    report["baseline_mesh"] = {"sha256": file_sha256(high_mesh), "role": "posthoc comparison"}
    report["visual_highinfo_vs_multiview"] = compare_five(
        imported, high_mesh, ply, high, args.report_dir/"visual-highinfo-vs-multiview")
    if args.stereo is not None:
        stereo_mesh = args.stereo/"rgb-stereo-mesh"/"mesh.ply"
        if not stereo_mesh.is_file():
            raise DataError("Previous real stereo control Mesh missing")
        report["stereo_mesh"] = {"sha256": file_sha256(stereo_mesh),
                                 "role": "posthoc same-room previous RGB-only algorithm"}
        report["visual_stereo_vs_multiview"] = compare_five(
            imported, stereo_mesh, ply, high, args.report_dir/"visual-stereo-vs-multiview")
    import trimesh
    from tum_graph_visual import read_vertex_rgb
    for name, mesh in (("measured-room", high_mesh), ("multiview-rgb", ply)):
        vv, ff = read_echo_ply(mesh)
        cc = read_vertex_rgb(mesh, len(vv))
        glb = args.report_dir/(name+".glb")
        trimesh.Trimesh(vertices=vv, faces=ff, vertex_colors=cc, process=False).export(glb)
        report.setdefault("glbs", {})[name] = file_sha256(glb)
    report["status"] = "real_room_multiview_mesh_generated_visual_FAIL_pending_review"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--stereo", type=Path)
    p.add_argument("--archive", type=Path)
    args = p.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "tum-fr1-room-known-pose-multiview-plane-sweep-v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "checkout_sha": os.environ.get("GITHUB_SHA"),
        "retained": ["real 640x480 RGB (320x240 matching)", "96 known metric training poses",
                     "camera intrinsics", "frozen holdout camera poses"],
        "removed_from_reconstruction": ["measured sensor depth pixels", "existing Mesh prior"],
        "source_archive_contains_sensor_depth": True,
        "importer_uses_sensor_depth_for_timestamp_association": True,
        "sensor_Z_used_for_inference_or_fusion": False,
        "metric_scale": "known TUM camera translations in metres",
        "issue_74": "FAIL / NOT ACCEPTED", "status": "started"}
    start = time.perf_counter()
    try:
        run(args, report)
        code = 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        code = 1
    finally:
        report["elapsed_seconds"] = round(time.perf_counter()-start, 2)
        (args.report_dir/"report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps({"status": report["status"], "error": report.get("error"),
                          "new_mesh": report.get("new_mesh")}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
