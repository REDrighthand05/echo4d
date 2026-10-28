"""Real TUM frozen 48-view RGB-MVS depth -> alternative TSDF edge extraction.

Reuse inferred Z and poses from Actions #40692184301, not measured depth.
Integrate the SAME volume as the original projective TSDF; change only the
mesh extraction support rule. Original RGB is recovered from pinned TUM.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from echo4d.fusion import FusionConfig, TSDFVolume
from echo4d.known_pose_edge_extraction import extract_observed_edge_mesh
from echo4d.model import Camera, Frame, file_sha256
from echo4d.output import mesh_summary, write_ply
from tum_public_roundtrip import ROOT, BOUNDS
from tum_rgb_stereo import HOLDOUT, rgb_pose_frames, load_stereo_rgb


def write_verified_control(baseline, original: Path, replayed: Path) -> dict:
    """Require byte-identical baseline reproduction, not merely equal face counts."""
    replayed.parent.mkdir(parents=True, exist_ok=True)
    write_ply(replayed, baseline)
    actual, expected = file_sha256(replayed), file_sha256(original)
    if actual != expected:
        raise ValueError("Replayed control Mesh SHA differs from frozen control")
    return {"replayed_sha256": actual, "frozen_sha256": expected, "byte_identical": True}


def replay(old: Path, source_root: Path, target: Path) -> dict:
    manifest = json.loads((old / "inputs/recording.json").read_text())
    entries = manifest["frames"]
    camera = Camera.from_dict(manifest["intrinsics"])
    if (len(entries) != 48 or camera.width != 320 or camera.height != 240
            or manifest["units"] != "meters" or manifest["depth_scale"] != 1):
        raise ValueError("Expected frozen metric 48-frame RGB-inferred recording")
    if not (old / "reconstruction/mesh.ply").is_file():
        raise FileNotFoundError("Original control Mesh missing")
    if target.exists():
        raise FileExistsError("Output must not already exist")
    frames = rgb_pose_frames(source_root / ROOT)
    cfg = FusionConfig(tuple(map(float, BOUNDS)), voxel=.05, truncation=.15,
                       max_depth=4., max_voxels=4_000_000, min_observations=1)
    volume = TSDFVolume(cfg)
    source_rgb = []
    for index, entry in enumerate(entries):
        absolute_t = frames[0]["t_s"] + float(entry["t"])
        matches = [f for f in frames if f["index"] not in HOLDOUT
                   and abs(f["t_s"] - absolute_t) < 1e-5]
        if len(matches) != 1:
            raise ValueError("Training frame/pose cannot be uniquely recovered")
        selected = matches[0]
        rgb = load_stereo_rgb(selected["rgb"], 320)
        pose = np.asarray(entry["cameraToWorld"], dtype=np.float64).reshape(
            (4, 4), order="F")
        depth = np.load(old / "inputs" / entry["depth"], allow_pickle=False)
        if depth.shape != (240, 320) or depth.dtype != np.float32:
            raise ValueError("Frozen RGB-inferred depth shape/dtype changed")
        volume.integrate(Frame(float(entry["t"]), pose, depth, rgb),
                         camera, index)
        source_rgb.append(selected["rgb_sha256"])
    # Independently reproduce the original extractor from the same volume.
    baseline = volume.extract_mesh()
    original_mesh = mesh_summary(baseline)
    if original_mesh["triangles"] != 8419:
        raise ValueError("Frozen baseline TSDF no longer matches 8419 triangles")
    control_receipt = write_verified_control(
        baseline, old / 'reconstruction/mesh.ply', target / 'control-replayed.ply')
    candidate, stats = extract_observed_edge_mesh(volume)
    reconstruction = target / "reconstruction"
    reconstruction.mkdir()
    mesh = reconstruction / "mesh.ply"
    write_ply(mesh, candidate)
    # This report is NOT a visual acceptance verdict.
    result = {
        "experiment": "TUM frozen inferred-Z observed-edge TSDF extraction",
        "checkout_sha": os.environ.get("GITHUB_SHA"),
        "source": "TUM fr1_xyz original 640x480 RGB (fusion RGB 320x240)",
        "original_rgb_sha256": source_rgb,
        "inferred_depth_source": "RGB plane sweep from run 40692184301",
        "metric_pose_source": "TUM measured metric camera poses",
        "retained": ["48 RGB training frames", "measured poses", "intrinsics",
                     "48 frozen RGB-inferred depth maps"],
        "removed": ["measured sensor depth pixels", "prior Mesh during fusion"],
        "sensor_depth_png_files_opened": 0,
        "prior_mesh_used_for_reconstruction": False,
        "control_mesh_sha256": file_sha256(old / "reconstruction/mesh.ply"),
        "control_faces_replayed": original_mesh["triangles"],
        "control_replay": control_receipt,
        "control_inferred_manifest_sha256": file_sha256(old / "inputs/recording.json"),
        "control_inferred_depth_sha256": {entry["depth"]: file_sha256(old / "inputs" / entry["depth"]) for entry in entries},
        "control_mesh_summary": original_mesh,
        "new_mesh_summary": mesh_summary(candidate),
        "mesh_sha256": file_sha256(mesh),
        "edge_extraction": stats,
        "fusion_config": {"bounds": list(map(float, BOUNDS)), "voxel": .05,
                          "truncation": .15, "max_depth": 4., "min_observations": 1},
        "visual_acceptance": "FAIL pending actual visual recognition",
        "limitations": ["Extra triangles are attached observed TSDF crossings, not new measurements",
                        "No unknown-pose RGB-only or complete scene recovery claim"],
    }
    (target / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return {"mesh_sha256": result["mesh_sha256"],
            "old_faces": original_mesh["triangles"],
            "new_faces": result["new_mesh_summary"]["triangles"],
            "edge_extraction": stats}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.old, args.source_root, args.target)))
