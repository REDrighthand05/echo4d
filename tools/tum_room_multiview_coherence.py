"""Real-room RGB-only inferred-Z *spatial coherence* adapter.

Child of the existing known-pose multi-view experiment (#94). No sensor Z,
no learned weights, no Mesh prior. Alters inferred depth BEFORE existing TSDF,
rather than post-filtering triangles. No filling of unobserved pixels.
"""
from __future__ import annotations

import os
from pathlib import Path
import warnings

import numpy as np

import tum_room_knownpose_multiview as base
from echo4d.model import DataError, file_sha256


def stabilize_depth(rgb: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, dict]:
    """Use RGB-local consensus to correct photometric winner-take-all depth spikes.

    A source pixel must already have RGB multi-view inferred depth. Neighbor
    pixels are spatial corroboration, NOT additional independent camera votes.
    Color-gating avoids averaging across obvious object boundaries. Low-texture
    or unsupported regions are deliberately left unchanged, not invented.
    """
    if (rgb.ndim != 3 or rgb.shape[:2] != depth.shape or rgb.shape[2] != 3
            or rgb.dtype != np.uint8 or depth.dtype != np.float32
            or not np.isfinite(depth).all() or (depth < 0).any()):
        raise DataError("Expected finite RGB-aligned nonnegative float32 inferred Z")
    h, w = depth.shape
    padded_z = np.pad(depth, 2, mode="constant", constant_values=0)
    padded_rgb = np.pad(rgb.astype(np.int16), ((2, 2), (2, 2), (0, 0)),
                        mode="edge")
    center = rgb.astype(np.int16)
    neighbors = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dx == 0 and dy == 0:
                continue
            zz = padded_z[dy + 2:dy + 2 + h, dx + 2:dx + 2 + w]
            cc = padded_rgb[dy + 2:dy + 2 + h, dx + 2:dx + 2 + w]
            color_distance = np.mean(np.abs(cc - center), axis=2)
            good = (zz > 0) & (color_distance <= 24.)
            neighbors.append(np.where(good, zz, np.nan))
    stack = np.stack(neighbors, axis=0)
    support = np.isfinite(stack).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(stack, axis=0)
        mad = np.nanmedian(np.abs(stack - median[None, :, :]), axis=0)
    coherent = ((depth > 0) & (support >= 6) & np.isfinite(median)
                & np.isfinite(mad) & (mad <= .035 + .035 * median))
    discrepancy = np.abs(depth - median)
    replace = coherent & (discrepancy > .12 + .06 * median)
    smooth = coherent & ~replace & (discrepancy < .045 + .025 * median)
    output = depth.copy()
    output[replace] = median[replace]
    output[smooth] = .75 * depth[smooth] + .25 * median[smooth]
    assert np.array_equal(output == 0, depth == 0), "Unknown depth must stay unknown"
    return output, {
        "original_valid_pixels": int(np.count_nonzero(depth)),
        "corrected_spikes": int(np.count_nonzero(replace)),
        "locally_smoothed_pixels": int(np.count_nonzero(smooth)),
        "unknown_pixels_filled": 0,
        "method": "RGB-gated 5x5 inferred-depth consensus, no sensor Z",
    }


_original_sweep = base.sweep_depth


def coherent_sweep(reference, sources, K, reference_pose, source_poses,
                   depths=base.DEPTHS_M):
    inferred, receipt = _original_sweep(reference, sources, K, reference_pose,
                                        source_poses, depths=depths)
    stabilized, changes = stabilize_depth(reference, inferred)
    return stabilized, {**receipt, "local_depth_coherence": changes}


_original_run = base.run


def run_with_old_comparison(args, report):
    report["algorithm_variant"] = (
        "known-pose RGB multiview plane sweep + RGB-gated inferred-depth coherence")
    _original_run(args, report)
    previous_dir = os.environ.get("ECHO_PREVIOUS_MULTIVIEW")
    if previous_dir:
        previous = Path(previous_dir)
        prior_report = base.json.loads((previous / "report.json").read_text())
        prior_mesh = previous / "multiview-mesh" / "mesh.ply"
        if (file_sha256(prior_mesh) != prior_report["new_mesh"]["sha256"]
                or prior_report["source_archive_sha256"] != report["source_archive_sha256"]
                or prior_report["selection"]["expanded_indices"]
                   != report["selection"]["expanded_indices"]):
            raise DataError("Old multiview control does not match real room/split/SHA")
        high = base.json.loads(
            (args.baseline / "room-sparse-quality-report.json").read_text())
        new_mesh = args.report_dir / "multiview-mesh" / "mesh.ply"
        report["visual_old_vs_coherent"] = base.compare_multiview_five(
            args.work / "room-import", prior_mesh, new_mesh, high,
            args.report_dir / "visual-old-vs-coherent",
            "previous RGB multiview inferred-Z / no prior Mesh")
        import trimesh
        from echo4d.holdout import read_echo_ply
        from tum_graph_visual import read_vertex_rgb
        vv, ff = read_echo_ply(prior_mesh)
        cc = read_vertex_rgb(prior_mesh, len(vv))
        old_glb = args.report_dir / "old-multiview-rgb.glb"
        trimesh.Trimesh(vertices=vv, faces=ff, vertex_colors=cc,
                        process=False).export(old_glb)
        report["old_multiview_control"] = {
            "mesh_sha256": file_sha256(prior_mesh),
            "glb_sha256": file_sha256(old_glb),
            "role": "posthoc visual A/B only; never candidate input"}
    report["issue_74"] = "FAIL / NOT ACCEPTED"
    report["status"] = "real_room_coherence_mesh_generated_visual_FAIL_pending_review"


def main():
    base.sweep_depth = coherent_sweep
    base.run = run_with_old_comparison
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
