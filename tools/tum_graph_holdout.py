"""Frozen 48/12 real TUM graph holdout diagnostic, not independent surface truth.

Only run after a completed Mesh/graph receipt exists. Never tune reconstruction
using these same-sensor holdouts or upgrade #74 visual acceptance.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

from echo4d.holdout import evaluate_frames, read_echo_ply
from echo4d.model import DataError, file_sha256, load_recording, safe_path
from tum_composable_graph import HOLDOUT, TOTAL
from tum_public_roundtrip import PINNED_ARCHIVE_SHA256


def _read_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise DataError("Expected JSON receipt object")
    return data


def score_frozen_holdout(full_manifest, train_manifest, mesh_path, execution_path, report_path):
    """Evaluate all fixed 12 heldouts after verifying the actual graph and split."""
    full_manifest, train_manifest, mesh_path = (Path(p).resolve() for p in
                                                 (full_manifest, train_manifest, mesh_path))
    if full_manifest.parent != train_manifest.parent:
        raise DataError("Full and train manifests must share one source root")
    full, train = load_recording(full_manifest), load_recording(train_manifest)
    if (len(full) != TOTAL or len(train) != TOTAL - len(HOLDOUT)
            or full.source != "recorded" or train.source != "recorded"
            or full.inline or train.inline or full.camera != train.camera
            or full.depth_scale != train.depth_scale):
        raise DataError("Not the frozen 60/48 recorded TUM manifest pair")
    if len(HOLDOUT) != 12 or len(set(HOLDOUT)) != 12:
        raise DataError("Invalid frozen holdout indices")
    expected = [frame for i, frame in enumerate(full.entries) if i not in HOLDOUT]
    if train.entries != expected:
        raise DataError("Train manifest differs from the exact 48-frame complement")
    train_depth = [safe_path(full.root, f["depth"]) for f in train.entries]
    holdout_depth = [safe_path(full.root, full.entries[i]["depth"]) for i in HOLDOUT]
    if (len(set(train_depth)) != 48 or len(set(holdout_depth)) != 12
            or set(train_depth) & set(holdout_depth)):
        raise DataError("Train/holdout depth source paths overlap or repeat")

    execution, report = _read_json(execution_path), _read_json(report_path)
    if (execution.get("status") != "executed_unscored"
            or execution.get("quality") != "not_evaluated"
            or report.get("status") != "executed_unscored"
            or report.get("source", {}).get("archive_sha256") != PINNED_ARCHIVE_SHA256):
        raise DataError("Cannot score an unverified or incomplete real TUM graph")
    inp = report.get("input", {})
    if (inp.get("original_manifest_sha256") != file_sha256(full_manifest)
            or inp.get("train_manifest_sha256") != file_sha256(train_manifest)
            or inp.get("holdout_indices") != list(HOLDOUT)
            or inp.get("train_frames") != 48 or inp.get("holdout_frames") != 12):
        raise DataError("Frozen manifest or split differs from graph run")
    pins = execution.get("input_sha256")
    if (not isinstance(pins, dict) or len(pins) != 1
            or file_sha256(train_manifest) not in pins.values()):
        raise DataError("Graph receipt does not pin training manifest")
    target, outputs = execution.get("target"), execution.get("outputs")
    mesh_sha = file_sha256(mesh_path)
    if (not isinstance(target, str) or not isinstance(outputs, dict)
            or not isinstance(outputs.get(target), dict)
            or outputs[target].get("sha256") != mesh_sha
            or report.get("mesh_sha256") != mesh_sha):
        raise DataError("Mesh does not match both pinned execution receipts")

    # Pin heldout files before accessing any heldout depth pixel.
    source_hashes = {}
    for index in HOLDOUT:
        for key in ("depth", "color", "mask"):
            if key in full.entries[index]:
                path = safe_path(full.root, full.entries[index][key])
                source_hashes[f"{index}:{key}"] = (path, file_sha256(path))
    started = time.perf_counter()
    vertices, faces = read_echo_ply(mesh_path)
    diagnostic = evaluate_frames(vertices, faces, full, list(HOLDOUT), max_depth=4.0)
    if (diagnostic.get("heldout_frames") != 12
            or diagnostic.get("heldout_indices") != list(HOLDOUT)
            or [r.get("index") for r in diagnostic.get("per_frame", [])] != list(HOLDOUT)):
        raise DataError("Diagnostic omitted or reordered a frozen holdout frame")
    for path, expected_sha in source_hashes.values():
        if file_sha256(path) != expected_sha:
            raise DataError("Heldout source changed during evaluation")
    if file_sha256(mesh_path) != mesh_sha:
        raise DataError("Mesh changed during evaluation")
    return {
        "schema": "echo-graph-sensor-holdout-diagnostic-1",
        "status": "diagnostic_only", "quality_acceptance": "not_evaluated",
        "independent_surface_truth": False, "human_visual_review": False,
        "source_scope": "same_TUM_RGBD_sensor_withheld_views",
        "training_frames": 48, "holdout_frames": 12,
        "source_archive_sha256": PINNED_ARCHIVE_SHA256,
        "source_full_manifest_sha256": file_sha256(full_manifest),
        "source_train_manifest_sha256": file_sha256(train_manifest),
        "graph_receipt_sha256": file_sha256(execution_path),
        "graph_report_sha256": file_sha256(report_path),
        "mesh_sha256": mesh_sha,
        "holdout_source_sha256": {k: sha for k, (_, sha) in source_hashes.items()},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "diagnostic": diagnostic,
        "limitations": [
            "Same RGB-D sensor depth, not independent scanned surface truth",
            "Known TUM poses, not independent SLAM",
            "Selecting a winner with these holdouts needs a new frozen scene",
            "Overlap RMSE ignores missing surfaces; coverage is separate",
            "No #74 original RGB 3+2 renderings or independent human acceptance",
            "Historical Desk V2 16.7779cm / 15cm FAIL unchanged",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("full-manifest", "train-manifest", "mesh", "execution", "report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("Output must be a new file in an existing directory")
    try:
        result = score_frozen_holdout(args.full_manifest, args.train_manifest,
                                      args.mesh, args.execution, args.report)
        code = 0
    except Exception as error:
        result = {
            "schema": "echo-graph-sensor-holdout-diagnostic-1",
            "status": "blocked" if isinstance(error, (DataError, OSError)) else "failed",
            "quality_acceptance": "not_evaluated", "independent_surface_truth": False,
            "human_visual_review": False, "error_type": type(error).__name__,
            "error": str(error),
        }
        print(f"FROZEN_HOLDOUT_{result['status'].upper()}: {error}", file=sys.stderr)
        code = 1
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "receipt": str(args.output)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
