"""Second real algorithm on the *same* pinned TUM 48/12 split as TSDF.

The first TSDF run imports and pins the source. This script does NOT download,
reimport, change the split, or treat sensor holdout as independent surface GT.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from echo4d import projective_meshing
from echo4d.composable import ArtifactType, ModuleSpec, compose
from echo4d.graph_runner import (
    InputFile, archive_pinned_files, execute_graph, projective_recording_adapter,
)
from echo4d.model import DataError, file_sha256, load_recording
from echo4d.output import write_json
from tum_composable_graph import HOLDOUT, PINNED_ARCHIVE_SHA256, TOTAL
from tum_graph_holdout import score_frozen_holdout


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DataError(f"Expected object receipt: {path}")
    return value


def check_baseline(work: Path, baseline: Path) -> tuple[dict, dict, dict]:
    """Fail closed before constructing an alternative on mismatched data."""
    full = work / "imported" / "recording.json"
    train = work / "imported" / "recording-train.json"
    original = load_recording(full)
    training = load_recording(train)
    if (len(original) != TOTAL or len(training) != TOTAL - len(HOLDOUT)
            or original.source != "recorded" or training.source != "recorded"
            or training.entries != [f for i, f in enumerate(original.entries) if i not in HOLDOUT]):
        raise DataError("Frozen TUM 60/48/12 source or split mismatch")
    report = read_json(baseline / "report.json")
    graph = read_json(baseline / "graph-execution.json")
    holdout = read_json(baseline / "heldout-view-consistency.json")
    mesh = baseline / "mesh.ply"
    mesh_sha = file_sha256(mesh)
    if (report.get("status") != "executed_unscored"
            or graph.get("status") != "executed_unscored"
            or graph.get("quality") != "not_evaluated"
            or graph.get("modules") != ["known_pose_sensor_tsdf"]
            or report.get("mesh_sha256") != mesh_sha
            or graph.get("outputs", {}).get(graph.get("target"), {}).get("sha256") != mesh_sha
            or report.get("source", {}).get("archive_sha256") != PINNED_ARCHIVE_SHA256
            or report.get("input", {}).get("train_manifest_sha256") != file_sha256(train)
            or report.get("input", {}).get("original_manifest_sha256") != file_sha256(full)
            or report.get("input", {}).get("holdout_indices") != list(HOLDOUT)
            or graph.get("checkout_sha") != report.get("checkout_sha")):
        raise DataError("Baseline graph/source/Mesh not pinned to fixed TUM protocol")
    pins = graph.get("input_sha256")
    if not isinstance(pins, dict) or list(pins.values()) != [file_sha256(train)]:
        raise DataError("Baseline graph manifest pin missing")
    if (holdout.get("status") != "diagnostic_only"
            or holdout.get("source_train_manifest_sha256") != file_sha256(train)
            or holdout.get("source_full_manifest_sha256") != file_sha256(full)
            or holdout.get("mesh_sha256") != mesh_sha
            or holdout.get("diagnostic", {}).get("heldout_indices") != list(HOLDOUT)
            or holdout.get("diagnostic", {}).get("heldout_frames") != 12):
        raise DataError("Baseline 12-frame holdout diagnostic is not comparable")
    return report, graph, holdout


def run(work: Path, baseline: Path, alternative: Path, *, pixel_stride: int = 4) -> dict:
    report, tsdf_graph, tsdf_holdout = check_baseline(work, baseline)
    checkout = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
        text=True, timeout=5).strip()
    if checkout != report["checkout_sha"] or checkout != tsdf_graph["checkout_sha"]:
        raise DataError("Both algorithms must execute at the identical checkout SHA")
    manifest = work / "imported" / "recording-train.json"
    full = work / "imported" / "recording.json"
    common = dict(scene_id="tum_fr1_xyz_fixed60", coordinate_id="tum_z_up",
                  partition="train", length_unit="m", handedness="right")
    inp = ArtifactType("echo_recording", "world", "metric", "frame_pts",
                       "measured", time_id="tum_sensor_time", time_unit="s",
                       pose_direction="none", **common)
    out = ArtifactType("mesh", "world", "metric", "none", "estimated",
                       time_id="none", time_unit="none", pose_direction="none", **common)
    provider = Path(projective_meshing.__file__).resolve()
    module = ModuleSpec("observed_projective_triangles", (inp,), (out,), 1, True,
                        file_sha256(provider))
    routes = compose((inp,), (module,), out)
    if len(routes.routes) != 1:
        raise DataError("Alternative typed route is missing or ambiguous")
    adapter = projective_recording_adapter(module.module_id, provider, inp, out,
                                           pixel_stride=pixel_stride)
    alternative.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    graph = execute_graph(routes.routes[0], (module,), {module.module_id: adapter},
                          {inp: InputFile(manifest, file_sha256(manifest))},
                          out, work / "graph-projective", checkout_sha=checkout)
    elapsed = round(time.perf_counter() - started, 3)
    mesh = work / "graph-projective" / graph["outputs"][out.label()]["path"]
    shutil.copyfile(mesh, alternative / "mesh.ply")
    shutil.copyfile(work / "graph-projective" / "execution.json",
                    alternative / "graph-execution.json")
    mesh_sha = file_sha256(alternative / "mesh.ply")
    if mesh_sha != graph["outputs"][out.label()]["sha256"]:
        raise DataError("Alternative Mesh changed after graph execution")
    alt_report = {
        "experiment": "tum-fr1-xyz-fixed60-typed-graph-projective-v1",
        "status": "executed_unscored", "algorithm": "per-frame-projective-triangles-v1",
        "fusion": False, "geometry_quality": "not_evaluated",
        "source": report["source"], "input": report["input"],
        "checkout_sha": checkout, "mesh_sha256": mesh_sha,
        "graph_seconds": elapsed, "graph": graph,
        "independent_surface_truth": "not_available",
        "human_visual_review": "not_performed",
        "desk_v2_frozen": "FAIL 16.7779cm > 15cm; unchanged",
        "issue_74": "FAIL / NOT ACCEPTED",
    }
    provider_report = mesh.parent / "provider-report.json"
    alt_report["archived_metadata"] = archive_pinned_files({
        "full-recording.json": InputFile(full, report["input"]["original_manifest_sha256"]),
        "train-recording.json": InputFile(manifest, report["input"]["train_manifest_sha256"]),
        "association.json": InputFile(work / "imported" / "association.json", report["input"]["association_sha256"]),
        "provider-report.json": InputFile(provider_report, file_sha256(provider_report)),
    }, alternative / "metadata")
    write_json(alternative / "report.json", alt_report)
    diagnostic = score_frozen_holdout(full, manifest, alternative / "mesh.ply",
                                      alternative / "graph-execution.json",
                                      alternative / "report.json")
    write_json(alternative / "heldout-view-consistency.json", diagnostic)
    # This is a diagnostic comparison only, not selection on a heldout set.
    a = tsdf_holdout["diagnostic"]
    b = diagnostic["diagnostic"]
    summary = {
        "schema": "echo-tum-two-geometry-providers-1",
        "status": "diagnostic_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkout_sha": checkout,
        "same_frozen_training_and_holdout": True,
        "train_manifest_sha256": file_sha256(manifest),
        "full_manifest_sha256": file_sha256(full),
        "holdout_indices": list(HOLDOUT),
        "tsdf": {
            "algorithm": "projective-tsdf-v1",
            "mesh_sha256": tsdf_holdout["mesh_sha256"],
            "graph_receipt_sha256": file_sha256(baseline / "graph-execution.json"),
            "holdout_receipt_sha256": file_sha256(baseline / "heldout-view-consistency.json"),
            "coverage": a["observed_depth_coverage"], "rmse_m": a["rmse_m"],
            "mae_m": a["mae_m"],
        },
        "projective": {
            "algorithm": "per-frame-projective-triangles-v1",
            "mesh_sha256": mesh_sha,
            "graph_receipt_sha256": file_sha256(alternative / "graph-execution.json"),
            "holdout_receipt_sha256": file_sha256(alternative / "heldout-view-consistency.json"),
            "coverage": b["observed_depth_coverage"], "rmse_m": b["rmse_m"],
            "mae_m": b["mae_m"],
        },
        "quality_acceptance": "not_evaluated",
        "independent_surface_truth": False,
        "human_visual_review": False,
        "no_winner_selected": True,
        "limitations": [
            "Same RGB-D sensor heldout views, NOT independent surface truth",
            "Known ground-truth TUM poses, NOT RGB-only or estimated SLAM",
            "Using this holdout to select a model requires a new frozen scene for generalization",
            "Projective unmerged patches may be fragmented and have more triangles",
            "#74 visual acceptance FAIL; desk V2 16.7779cm/15cm FAIL unchanged",
        ],
    }
    write_json(alternative / "ab-diagnostic.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--alternative", type=Path, required=True)
    parser.add_argument("--pixel-stride", type=int, default=4)
    args = parser.parse_args()
    try:
        summary = run(args.work.resolve(), args.baseline.resolve(),
                      args.alternative.resolve(), pixel_stride=args.pixel_stride)
    except Exception as error:
        args.alternative.mkdir(parents=True, exist_ok=True)
        write_json(args.alternative / "ab-failure.json", {
            "schema": "echo-tum-two-geometry-providers-failure-1",
            "status": "blocked" if isinstance(error, (DataError, OSError, ValueError)) else "failed",
            "error_type": type(error).__name__, "error": str(error),
            "quality_acceptance": "not_evaluated",
            "independent_surface_truth": False,
            "human_visual_review": False,
        })
        print(f"TUM_PROJECTIVE_AB_FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": summary["status"],
                      "tsdf": summary["tsdf"]["mesh_sha256"],
                      "projective": summary["projective"]["mesh_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
