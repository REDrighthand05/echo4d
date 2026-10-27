"""M2 bridge: execute ONE typed known-pose metric RGB-D -> CPU TSDF graph.

This is a real provider invocation, not a claim of independent geometry quality.
No automatic RGB-only depth, pose estimation, or unverified type conversion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from .cli import reconstruct_command
from .composable import ArtifactType, ModuleSpec, compose
from .model import DataError, file_sha256, load_recording


def _checkout_sha() -> str | None:
    """Report the actual checkout if git metadata is available; never guess."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, check=True, timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    sha = result.stdout.strip().lower()
    return sha if len(sha) == 40 and all(c in "0123456789abcdef" for c in sha) else None


def _provider_revision() -> str:
    """Hash the actual provider implementation files, not a mutable branch name."""
    from . import cli, composable, evaluation, fusion, model, output
    digest = hashlib.sha256()
    for path in (Path(__file__).resolve(), *(Path(module.__file__).resolve() for module in
                   (cli, composable, evaluation, fusion, model, output))):
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def run_known_pose_tsdf(args: argparse.Namespace) -> dict:
    if len(args.expected_manifest_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in args.expected_manifest_sha256
    ):
        raise DataError("expected manifest SHA256 must be 64 lowercase hex characters")
    if args.source not in ("recorded", "synthetic"):
        raise DataError("source must be recorded or synthetic")
    if args.pose_evidence not in ("measured", "estimated", "generated") or args.depth_evidence not in (
        "measured", "estimated", "generated"
    ):
        raise DataError("pose/depth evidence class must be explicit")
    if args.source == "synthetic" and (args.pose_evidence, args.depth_evidence) != ("generated", "generated"):
        raise DataError("synthetic recording cannot claim measured or estimated sensor evidence")
    if args.source == "recorded" and (args.pose_evidence == "generated" or args.depth_evidence == "generated"):
        raise DataError("recorded input with generated geometry needs a separate provider")
    if args.output.resolve() == args.recording.resolve().parent or args.output.resolve().is_relative_to(args.recording.resolve().parent):
        raise DataError("output must be outside the source recording directory")
    recording = load_recording(args.recording)
    if recording.source != args.source:
        raise DataError("recording source contradicts declared source")
    if recording.manifest_sha256 != args.expected_manifest_sha256:
        raise DataError("recording manifest SHA256 differs from frozen input")

    def port(kind: str, frame: str, evidence: str, *, timed: bool, pose: bool = False) -> ArtifactType:
        return ArtifactType(
            kind, frame, "metric", "frame_pts" if timed else "none", evidence, "train",
            args.scene_id, args.coordinate_id, args.time_id if timed else "none",
            "m", "s" if timed else "none", "right",
            "camera_to_world" if pose else "none",
        )

    poses = port("poses", "world", args.pose_evidence, timed=True, pose=True)
    depth = port("depth_z", "camera", args.depth_evidence, timed=True)
    mesh = port("mesh", "world", "estimated", timed=False)
    provider = ModuleSpec(
        "echo_cpu_tsdf", (poses, depth), (mesh,), 1, True, _provider_revision()
    )
    planned = compose((poses, depth), (provider,), mesh, max_steps=1, max_states=4, max_routes=1)
    if len(planned.routes) != 1 or planned.routes[0].modules != (provider.module_id,):
        raise DataError("no compatible installed CPU TSDF route")
    result = reconstruct_command(argparse.Namespace(
        recording=args.recording, output=args.output, bounds=args.bounds,
        voxel=args.voxel, truncation=None, max_depth=args.max_depth,
        max_voxels=args.max_voxels, min_observations=args.min_observations,
        stride=1, limit=None, reference=None, threshold=.05,
        no_scene=False, save_volume=False, quiet=True,
    ))
    if file_sha256(args.recording) != args.expected_manifest_sha256:
        raise DataError("recording manifest changed during provider execution")
    mesh_path = args.output / "mesh.ply"
    if not mesh_path.is_file() or file_sha256(mesh_path) != result["mesh_sha256"]:
        raise DataError("provider Mesh SHA256 verification failed")
    return {
        "status": "completed_unreviewed", "route_id": planned.routes[0].path_id,
        "modules": [{"id": provider.module_id, "revision_sha256": provider.revision}],
        "input_manifest_sha256": recording.manifest_sha256,
        "input_source_claim": args.source,
        "pose_evidence_claim": args.pose_evidence, "depth_evidence_claim": args.depth_evidence,
        "scene_id": args.scene_id, "coordinate_id": args.coordinate_id, "time_id": args.time_id,
        "checkout_sha": _checkout_sha(), "python_version": sys.version.split()[0],
        "output_mesh_sha256": result["mesh_sha256"],
        "provider_report_sha256": file_sha256(args.output / "report.json"),
        "frames": result["frames"], "vertices": result["vertices"], "triangles": result["triangles"],
        "provider_elapsed_seconds": result["elapsed_seconds"],
        "evidence_level": "E2_synthetic" if args.source == "synthetic" else "E3_candidate_unverified_origin",
        "independent_surface_truth": False, "human_visual_review": False,
        "limitations": ["Source/evidence labels are caller claims, not independently authenticated",
                        "No train/holdout protocol is proven by this runner",
                        "Same-sensor agreement is not independent surface accuracy",
                        "#74 3+2 visual acceptance is not performed"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path, help="new JSON file outside output/source")
    parser.add_argument("--source", required=True, choices=("recorded", "synthetic"))
    parser.add_argument("--pose-evidence", required=True, choices=("measured", "estimated", "generated"))
    parser.add_argument("--depth-evidence", required=True, choices=("measured", "estimated", "generated"))
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--coordinate-id", required=True)
    parser.add_argument("--time-id", required=True)
    parser.add_argument("--bounds", required=True, nargs=6, type=float)
    parser.add_argument("--voxel", type=float, default=.04)
    parser.add_argument("--max-depth", type=float, default=8.)
    parser.add_argument("--max-voxels", type=int, default=4_000_000)
    parser.add_argument("--min-observations", type=int, default=1)
    args = parser.parse_args(argv)
    if args.receipt.resolve().is_relative_to(args.output.resolve()) or args.receipt.resolve().is_relative_to(args.recording.resolve().parent):
        parser.error("receipt must be outside output and source directories")
    if args.receipt.exists() or not args.receipt.parent.is_dir():
        parser.error("receipt must be a new file in an existing directory")
    started = time.perf_counter()
    receipt = {"schema": "echo-composable-execution-1", "started_utc": datetime.now(timezone.utc).isoformat(),
               "command": "python -m echo4d.composable_runtime", "status": "running"}
    try:
        receipt.update(run_known_pose_tsdf(args))
        code = 0
    except (DataError, ValueError, OSError, MemoryError) as exc:
        receipt.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                        "checkout_sha": _checkout_sha(), "independent_surface_truth": False,
                        "human_visual_review": False})
        code = 2
    receipt["elapsed_seconds"] = time.perf_counter() - started
    with args.receipt.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "receipt": str(args.receipt),
                      "mesh_sha256": receipt.get("output_mesh_sha256")}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
