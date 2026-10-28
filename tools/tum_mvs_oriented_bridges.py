"""Real frozen 48-view TUM A/B: local inferred-depth plane orientation gate.

Reruns the PR91 short-bridge path on the same source and frozen RGB-derived Z.
No measured sensor depth is opened. Original TSDF faces are preserved.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import trimesh

from echo4d.fusion import Mesh
from echo4d.known_pose_edge_orientation import filter_oriented_bridge_faces
from echo4d.model import Camera, file_sha256
from echo4d.output import mesh_summary, write_ply
from tum_mvs_short_bridges import run as previous_run


def run(old: Path, source_root: Path, target: Path) -> dict:
    if target.exists():
        raise FileExistsError("Use a new output directory")
    prior = target / "prior"
    previous_run(old, source_root, prior)
    manifest = json.loads((old / "inputs/recording.json").read_text())
    if manifest["units"] != "meters" or len(manifest["frames"]) != 48:
        raise ValueError("Expected frozen 48-view metric known-pose RGB depth")
    camera = Camera.from_dict(manifest["intrinsics"])
    depths = [np.load(old / "inputs" / frame["depth"], allow_pickle=False)
              for frame in manifest["frames"]]
    poses = [np.asarray(frame["cameraToWorld"], dtype=np.float64).reshape(4, 4, order="F")
             for frame in manifest["frames"]]
    original = trimesh.load_mesh(prior / "prior/unfiltered/control-replayed.ply",
                                 force="mesh", process=False)
    candidate = trimesh.load_mesh(prior / "prior/unfiltered/reconstruction/mesh.ply",
                                  force="mesh", process=False)
    keep, support = filter_oriented_bridge_faces(
        original.vertices, original.faces, candidate.vertices, candidate.faces,
        depths, poses, camera)
    selected = candidate.faces[keep]
    used, remap = np.unique(selected, return_inverse=True)
    result = Mesh(
        candidate.vertices[used].astype(np.float32),
        remap.reshape(-1, 3).astype(np.int32),
        candidate.vertex_normals[used].astype(np.float32),
        candidate.visual.vertex_colors[used, :3].astype(np.uint8),
        np.ones(len(used), dtype=np.uint32),
        np.zeros(len(used), dtype=np.int32),
        np.zeros(len(used), dtype=np.int32),
    )
    result.validate()
    reconstruction = target / "reconstruction"
    reconstruction.mkdir(parents=True)
    write_ply(reconstruction / "mesh.ply", result)
    comparison = target / "prior-comparison"
    (comparison / "reconstruction").mkdir(parents=True)
    old_mesh = prior / "reconstruction/mesh.ply"
    shutil.copyfile(old_mesh, comparison / "reconstruction/mesh.ply")
    # Prior runner freezes five camera poses in its comparison receipt.
    shutil.copyfile(prior / "prior-comparison/report.json", comparison / "report.json")
    report = {
        "checkout_sha": os.environ.get("GITHUB_SHA"),
        "source": "TUM fr1_xyz 640x480 original RGB, 320x240 inferred-Z; CC BY 4.0",
        "retained": ["48 real RGB frames", "48 frozen RGB-inferred Z maps",
                     "known metric camera poses", "calibrated intrinsics"],
        "removed": ["measured sensor depth pixels", "external prior Mesh"],
        "prior_short_bridge_mesh_sha256": file_sha256(old_mesh),
        "new_mesh_sha256": file_sha256(reconstruction / "mesh.ply"),
        "summary": mesh_summary(result),
        "support": support,
        "sensor_depth_files_opened": 0,
        "baseline_mesh_used_for_same_volume_face_labels_only": True,
        "visual_acceptance": "FAIL pending 3 original RGB plus 2 free views",
        "historical_fail": "PR91 short bridges: 60.26% of new-only sensor-valid pixels >30cm error",
    }
    (target / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return {"mesh_sha256": report["new_mesh_sha256"],
            "faces": len(result.faces), "support": support}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.old, args.source_root, args.target)))
