"""Run real TUM room DA2 Small with a post-inference RGB cross-view consensus.

This is a small adapter to the existing #98 real-room pipeline. The same
measured-depth baseline is *only* a posthoc control. No new planner/frontend.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import da2_depth_consensus as consensus
import tum_room_knownpose_multiview as base
import tum_room_depthanything_small as da2
from echo4d.model import file_sha256


def build_manifest_with_consensus(imported: Path, target: Path, report: dict) -> Path:
    manifest = da2.build_manifest_da2(imported, target, report)
    data = json.loads(manifest.read_text())
    frames = data["frames"]
    K_data = data["intrinsics"]
    K = np.array([[K_data["fx"], 0, K_data["cx"]],
                  [0, K_data["fy"], K_data["cy"]], [0, 0, 1]], dtype=np.float64)
    depths = [np.load(target / f["depth"], allow_pickle=False) for f in frames]
    rgbs = [np.asarray(Image.open(target / f["color"]).convert("RGB")) for f in frames]
    poses = [np.asarray(f["cameraToWorld"], dtype=np.float64).reshape(4, 4, order="F")
             for f in frames]
    revised, receipts = consensus.reconcile_depths(depths, rgbs, poses, K)
    for f, z in zip(frames, revised):
        np.save(target / f["depth"], z, allow_pickle=False)
    data["name"] = "TUM room DA2 Small known-pose RGB cross-view consensus"
    data["notes"] = ("DA2 Small relative Z, known-pose RGB metric anchors, "
                     "RGB-only cross-view predicted-Z consensus; no sensor Z or prior Mesh")
    manifest.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    report["consensus"] = {
        "method": "independent predicted RGB depth reprojection, visibility and color agreement",
        "source": "DA2 relative depth plus known-pose RGB metric anchors, no measured depth",
        "frames": len(frames),
        "filled_pixels": sum(r["filled_rgb_supported_pixels"] for r in receipts),
        "blended_pixels": sum(r["blended_consistent_pixels"] for r in receipts),
        "receipts": receipts}
    return manifest


def run_with_consensus(args, report):
    da2.run_with_model(args, report)
    report["algorithm_variant"] = "DA2 Small + known-pose RGB metric anchors + cross-view consensus"
    report["status"] = "real_room_DA2_crossview_mesh_generated_visual_FAIL_pending_review"
    report["issue_74"] = "FAIL / NOT ACCEPTED; inspect 3+2 Mesh"


def main():
    base.sweep_depth = da2.learned_sweep
    base.build_manifest = build_manifest_with_consensus
    base.run = run_with_consensus
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
