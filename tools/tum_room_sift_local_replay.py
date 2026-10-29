"""Replay frozen real TUM fr1_room RGB geometry with SIFT triangulated anchors.

Candidate consumes only DA2 RGB-predicted depths, source RGB and metric poses.
Measured sensor Z and previous Mesh are never opened before fusion.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from echo4d.model import DataError, file_sha256
from sparse_open3d_fusion import fuse_sparse
from tum_highres_tsdf import face_components
from tum_room_da2_global_scale_replay import (
    checked_replay, mesh_glb, five_view_comparison)
from known_pose_sift_depth import correct_sequence

CONTROL_RUN = 40797864880
SOURCE_RUN = 40790769632
CONTROL_MESH_SHA256 = "20b156dd18d63aa62190bcacc02a50c563b77ca144e739875892028197c0f2dd"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Unmodified #99 source artifact for post-fusion 3+2 only")
    parser.add_argument("--control", type=Path, required=True,
                        help="Unmodified #100 RGB global-scale artifact")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, control, output = (x.resolve() for x in
                               (args.source, args.control, args.output))
    # Check both immutable real-scene controls, before any candidate processing.
    frozen, replay, _ = checked_replay(source)
    control_report = json.loads((control/"report.json").read_text())
    old_ply = control/"candidate-mesh/mesh.ply"
    if (control_report.get("new_mesh_sha256") != CONTROL_MESH_SHA256 or
            file_sha256(old_ply) != CONTROL_MESH_SHA256 or
            control_report.get("source_ledger_sha256") != file_sha256(replay/"ledger.json") or
            control_report.get("sensor_depth_arrays_used") is True):
        raise DataError("Control Mesh or frozen real RGB source mismatch")
    origin = control/"candidate-input"
    manifest = json.loads((origin/"recording.json").read_text())
    if (len(manifest["frames"]) != 30 or
            manifest["intrinsics"]["width"] != 320 or
            manifest["intrinsics"]["height"] != 240 or
            any("sensor" in f["depth"].lower() for f in manifest["frames"])):
        raise DataError("Unexpected candidate RGB-only depth input")
    frames, intr = manifest["frames"], manifest["intrinsics"]
    K = np.array([[intr["fx"], 0, intr["cx"]],
                  [0, intr["fy"], intr["cy"]], [0, 0, 1]], dtype=np.float64)
    poses = [np.asarray(f["cameraToWorld"], np.float64).reshape(4, 4, order="F")
             for f in frames]
    depths = [np.load(origin/f["depth"], allow_pickle=False) for f in frames]
    rgbs = [np.asarray(Image.open(origin/f["color"]).convert("RGB"))
            for f in frames]
    if any(z.shape != (240, 320) or rgb.shape != (240, 320, 3)
           for z, rgb in zip(depths, rgbs)):
        raise DataError("Input RGB and inferred depth resolutions differ")
    # Critical: old Mesh and heldout sensor depth are not passed to this function.
    revised, geometry = correct_sequence(depths, rgbs, poses, K)
    output.mkdir(parents=True, exist_ok=False)
    candidate = output/"candidate-input"
    shutil.copytree(origin, candidate)
    for f, z in zip(frames, revised):
        np.save(candidate/f["depth"], z, allow_pickle=False)
    manifest["name"] = "TUM fr1_room known-pose RGB SIFT local geometry"
    manifest["notes"] = ("DA2 Small predicted Z, known metric poses, RGB-only SIFT "
                         "triangulation local warp; no measured sensor Z or prior Mesh")
    (candidate/"recording.json").write_text(json.dumps(manifest, indent=2)+"\n")
    # Existing Open3D TSDF implementation; do not fork the Mesh engine.
    fuse_sparse(candidate/"recording.json", output/"candidate-mesh",
                voxel_m=.04, truncation_m=.12, max_depth_m=4., pixel_stride=2)
    ply = output/"candidate-mesh/mesh.ply"
    original = b"comment Open3D sparse TSDF; measured RGB-D; known poses; no inferred depth"
    replacement = b"comment Open3D sparse TSDF; SIFT RGB local geometry; known poses; no sensor Z"
    raw = ply.read_bytes()
    if raw.count(original) != 1:
        raise DataError("Unrecognized TSDF provenance header")
    ply.write_bytes(raw.replace(original, replacement, 1))
    fusion_path = output/"candidate-mesh/report.json"
    fusion = json.loads(fusion_path.read_text())
    fusion["geometry_source"] = "DA2 RGB predicted Z + RGB SIFT triangulation, known poses"
    fusion["mesh_sha256"] = file_sha256(ply)
    fusion_path.write_text(json.dumps(fusion, indent=2)+"\n")
    old_v, old_f, _ = mesh_glb(old_ply, output/"old-global-scale.glb")
    new_v, new_f, _ = mesh_glb(ply, output/"new-sift-local.glb")
    views = five_view_comparison(source, replay, old_ply, ply, output, frozen)
    # Correct inherited visual labels; underlying 3+2 geometry is unchanged.
    board_path = output/"source-old-new-3plus2.png"
    with Image.open(board_path) as image:
        board = image.copy()
    draw = ImageDraw.Draw(board)
    for row, item in enumerate(views):
        y = row*480
        draw.rectangle((640, y, 1919, y+25), fill=(12, 12, 12))
        draw.text((650, y+8), "old global-scale Mesh %s px" %
                  item["old_covered_pixels"], fill="white")
        draw.text((1290, y+8), "new SIFT-local Mesh %s px" %
                  item["new_covered_pixels"], fill="white")
    board.save(board_path)
    report = {
        "experiment": "TUM fr1_room known-pose RGB SIFT triangulation local depth",
        "source_run": SOURCE_RUN, "control_run": CONTROL_RUN,
        "retained": ["real 640x480 RGB (320x240 candidate)", "30 DA2 inferred Z",
                     "known metric camera poses", "camera intrinsics"],
        "removed": ["measured sensor depth pixels", "prior existing Mesh"],
        "metric_scale": "known TUM poses + earlier RGB-only metric anchors",
        "source_ledger_sha256": file_sha256(replay/"ledger.json"),
        "old_mesh_sha256": file_sha256(old_ply),
        "new_mesh_sha256": file_sha256(ply),
        "old_glb_sha256": file_sha256(output/"old-global-scale.glb"),
        "new_glb_sha256": file_sha256(output/"new-sift-local.glb"),
        "five_view_png_sha256": file_sha256(board_path),
        "old_faces": len(old_f), "new_faces": len(new_f),
        "old_components": face_components(len(old_v), old_f),
        "new_components": face_components(len(new_v), new_f),
        "five_views": views, "triangulated_geometry": geometry,
        "sensor_depth_arrays_used": False,
        "prior_mesh_used_in_candidate": False,
        "independent_surface_truth": False,
        "issue_74": "FAIL_NOT_ACCEPTED_until_human_visual_review"}
    (output/"report.json").write_text(json.dumps(report, indent=2,
                                                allow_nan=False)+"\n")
    print(json.dumps({"old_faces": len(old_f), "new_faces": len(new_f),
                      "triangulated_pairs": geometry["triangulated_rgb_pairs"],
                      "frames_corrected": geometry["frames_corrected"],
                      "five_views": views, "issue_74": report["issue_74"]}))


if __name__ == "__main__":
    main()
