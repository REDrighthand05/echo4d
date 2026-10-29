"""Replay frozen TUM fr1_room DA2 RGB depth; synchronize metric scales and fuse.

Input is the real #99 artifact, NOT measured sensor depth or prior Mesh. The
prior Mesh and three original heldout RGBs are opened only after fusion for A/B.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import trimesh

from da2_global_scale import synchronize_scales
from echo4d.model import Camera, DataError, file_sha256
from echo4d.holdout import read_echo_ply
from sparse_open3d_fusion import fuse_sparse
from tum_graph_visual import read_vertex_rgb, rasterize_vertex_rgb
from tum_highres_tsdf import face_components

FROZEN_RUN = 40790769632
FROZEN_ARCHIVE_SHA256 = "5ace47a1d2e53696bc939a84999293a04a7226958e848a609666691fd3cc38da"


def checked_replay(previous):
    report = json.loads((previous/"report.json").read_text())
    replay = previous/"replay-inputs"
    ledger_path = replay/"ledger.json"
    if (report.get("source_archive_sha256") != FROZEN_ARCHIVE_SHA256
            or report.get("replay_inputs", {}).get("ledger_sha256") != file_sha256(ledger_path)
            or report.get("new_mesh", {}).get("sha256") != file_sha256(previous/"multiview-mesh/mesh.ply")):
        raise DataError("Frozen source, ledger or previous Mesh changed")
    ledger = json.loads(ledger_path.read_text())
    if ledger.get("sensor_depth_arrays_retained") is not False:
        raise DataError("Sensor depth must not enter RGB replay")
    for item in ledger["files"]:
        path = replay/item["path"]
        if path.stat().st_size != item["bytes"] or file_sha256(path) != item["sha256"]:
            raise DataError("Frozen replay file changed: " + item["path"])
    manifest = json.loads((replay/"inferred/recording.json").read_text())
    if (len(manifest["frames"]) != 30 or manifest["intrinsics"]["width"] != 320
            or manifest["intrinsics"]["height"] != 240):
        raise DataError("Unexpected DA2 replay frames or resolution")
    return report, replay, manifest


def build_candidate(replay, manifest, output):
    source = replay/"inferred"
    target = output/"candidate-input"
    shutil.copytree(source, target)
    frames = manifest["frames"]
    intr = manifest["intrinsics"]
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]],
                  [0, 0, 1]], dtype=np.float64)
    poses = [np.array(f["cameraToWorld"], dtype=np.float64).reshape(4, 4, order="F")
             for f in frames]
    depths = [np.load(source/f["depth"], allow_pickle=False) for f in frames]
    rgbs = [np.asarray(Image.open(source/f["color"]).convert("RGB")) for f in frames]
    revised, scales = synchronize_scales(depths, rgbs, poses, K)
    for f, z in zip(frames, revised):
        np.save(target/f["depth"], z, allow_pickle=False)
    manifest["name"] = "TUM room DA2 known-pose RGB global depth scale"
    manifest["notes"] = ("DA2 Small relative RGB depth, RGB ZNCC known-pose metric anchors, "
                         "RGB-only global per-camera scale; no sensor Z or prior Mesh")
    (target/"recording.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    return target/"recording.json", scales


def mesh_glb(ply, output):
    vertices, faces = read_echo_ply(ply)
    rgb = read_vertex_rgb(ply, len(vertices))
    trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=rgb,
                    process=False).export(output)
    return vertices, faces, rgb


def five_view_comparison(previous, replay, old_mesh, new_mesh, output, frozen):
    visual = frozen["visual_da2_vs_consensus"]
    if [x["filename"] for x in visual] != [
            "heldout-7.png", "heldout-92.png", "heldout-162.png",
            "free-3.png", "free-4.png"]:
        raise DataError("Frozen 3+2 cameras changed")
    intr = json.loads((replay/"inferred/recording.json").read_text())["intrinsics"]
    camera = Camera(intr["width"], intr["height"], intr["fx"], intr["fy"],
                    intr["cx"], intr["cy"])
    assoc = json.loads((replay/"source/association.json").read_text())
    old_v, old_f = read_echo_ply(old_mesh)
    new_v, new_f = read_echo_ply(new_mesh)
    old_rgb = read_vertex_rgb(old_mesh, len(old_v))
    new_rgb = read_vertex_rgb(new_mesh, len(new_v))
    board = Image.new("RGB", (1920, 5*480), (12, 12, 12))
    draw = ImageDraw.Draw(board)
    views = []
    for row, item in enumerate(visual):
        pose = np.asarray(item["camera_to_world"], dtype=float).reshape(4, 4, order="F")
        old_im, old_cov = rasterize_vertex_rgb(old_v, old_f, old_rgb, camera, pose)
        new_im, new_cov = rasterize_vertex_rgb(new_v, new_f, new_rgb, camera, pose)
        if row < 3:
            idx = int(item["filename"].split("-")[1].split(".")[0])
            entry = assoc["frames"][idx]
            path = replay/"source"/entry["rgb_path"]
            if file_sha256(path) != entry["rgb_sha256"]:
                raise DataError("Original 640x480 RGB changed")
            with Image.open(path) as im:
                source = im.convert("RGB").copy()
            if source.size != (640, 480):
                raise DataError("Original RGB not 640x480")
        else:
            source = Image.new("RGB", (640, 480), (12, 12, 12))
        board.paste(source, (0, row*480))
        board.paste(Image.fromarray(old_im).resize((640, 480), Image.Resampling.NEAREST),
                    (640, row*480))
        board.paste(Image.fromarray(new_im).resize((640, 480), Image.Resampling.NEAREST),
                    (1280, row*480))
        draw.text((10, row*480+8), item["filename"]+" | source 640x480", fill="white")
        draw.text((650, row*480+8), f"old DA2 Mesh {old_cov} px", fill="white")
        draw.text((1290, row*480+8), f"global-scale Mesh {new_cov} px", fill="white")
        views.append({"view": item["filename"], "old_covered_pixels": old_cov,
                      "new_covered_pixels": new_cov})
    board.save(output/"source-old-new-3plus2.png")
    return views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous, output = args.previous.resolve(), args.output.resolve()
    frozen, replay, manifest = checked_replay(previous)
    output.mkdir(parents=True, exist_ok=False)
    candidate, scales = build_candidate(replay, manifest, output)
    # Reuse existing Open3D TSDF, unchanged parameters, no second planner.
    fuse_sparse(candidate, output/"candidate-mesh", voxel_m=.04,
                truncation_m=.12, max_depth_m=4., pixel_stride=2)
    ply = output/"candidate-mesh/mesh.ply"
    original = b"comment Open3D sparse TSDF; measured RGB-D; known poses; no inferred depth"
    replacement = b"comment Open3D sparse TSDF; DA2 RGB global scale; known metric poses; no sensor Z"
    raw = ply.read_bytes()
    if raw.count(original) != 1:
        raise DataError("Unknown TSDF provenance header; refusing misleading Mesh")
    ply.write_bytes(raw.replace(original, replacement, 1))
    fusion_path = output/"candidate-mesh/report.json"
    fusion = json.loads(fusion_path.read_text())
    fusion["geometry_source"] = "DA2 inferred RGB Z, known-pose RGB globally synchronized; no sensor Z"
    fusion["mesh_sha256"] = file_sha256(ply)
    fusion_path.write_text(json.dumps(fusion, indent=2, allow_nan=False)+"\n")
    old_ply = previous/"multiview-mesh/mesh.ply"
    old_v, old_f, _ = mesh_glb(old_ply, output/"old-da2.glb")
    new_v, new_f, _ = mesh_glb(ply, output/"new-global-scale.glb")
    views = five_view_comparison(previous, replay, old_ply, ply, output, frozen)
    report = {
        "experiment": "TUM fr1_room frozen DA2 RGB global scale synchronization",
        "source_run": f"https://github.com/REDrighthand05/echo4d/actions/runs/{FROZEN_RUN}",
        "source_archive_sha256": FROZEN_ARCHIVE_SHA256,
        "source_ledger_sha256": file_sha256(replay/"ledger.json"),
        "source_inferred_manifest_sha256": file_sha256(replay/"inferred/recording.json"),
        "old_mesh_sha256": file_sha256(old_ply), "new_mesh_sha256": file_sha256(ply),
        "old_glb_sha256": file_sha256(output/"old-da2.glb"),
        "new_glb_sha256": file_sha256(output/"new-global-scale.glb"),
        "input_kept": ["640x480 source RGB (320x240 inference)", "30 DA2 inferred depth maps",
                       "96 known metric training poses for original RGB calibration", "intrinsics"],
        "input_removed": ["measured sensor depth pixels", "prior existing Mesh"],
        "metric_scale": "known-pose RGB anchor calibration plus global RGB/pose correction",
        "model_reused": "DA2 Small Apache-2.0 weights previously executed; not rerun",
        "old_faces": len(old_f), "new_faces": len(new_f),
        "old_components": face_components(len(old_v), old_f),
        "new_components": face_components(len(new_v), new_f),
        "scales": scales, "five_views": views, "fusion": fusion,
        "sensor_depth_arrays_used": False, "old_mesh_used_in_candidate": False,
        "independent_surface_truth": False,
        "issue_74": "FAIL_NOT_ACCEPTED; visual review required"}
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"old_faces": len(old_f), "new_faces": len(new_f),
                      "heldout_pair_rmse_before_m": scales.get("heldout_pair_rmse_before_m"),
                      "heldout_pair_rmse_after_m": scales.get("heldout_pair_rmse_after_m"),
                      "five_views": views, "issue_74": report["issue_74"]}))


if __name__ == "__main__":
    main()
