"""Issue A: frozen #25/#26 versus PR #34 on one B-V1 mesh and pose/depth cohort.

This is a controlled evaluator regression test, NOT independent scanned-surface GT.
#25/#26 scoring entrypoints are loaded verbatim from pinned Git commits; the
mesh rasterizer, model and depth reader are shared and must be audited separately.
Do not use AB-V2-final data here: #26's frozen split contract is B-V1.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

from echo4d.metric_holdout import evaluate_sensor_metric_holdouts
from echo4d.model import DataError, file_sha256, safe_path
from tum_public_roundtrip import ROOT
from tum_sensor_metric_mesh import DOWNSAMPLE, split_names

PR25_SHA = "072186fe7989efb300759b9e5442e2ca56d208f7"
PR26_SHA = "454eaaa8d2d99d19adbeb40210930b360d6d219a"
KEYS = ("coverage", "mae_m", "rmse_m", "p95_abs_m")
COUNT_KEYS = ("sensor_valid_pixels", "overlap_pixels")
TOLERANCE = 1e-9


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_frozen(path: Path, qualified_name: str):
    if not path.is_file():
        raise DataError(f"Frozen evaluator missing: {path}")
    spec = importlib.util.spec_from_file_location(qualified_name, path)
    if spec is None or spec.loader is None:
        raise DataError(f"Cannot import frozen evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


def require_same_number(label: str, expected: float, actual: float, *, integer: bool = False) -> float:
    if isinstance(expected, bool) or isinstance(actual, bool):
        raise DataError(f"{label}: boolean is not a numeric result")
    if integer:
        if int(expected) != expected or int(actual) != actual or int(expected) != int(actual):
            raise DataError(f"{label}: count mismatch {expected} vs {actual}")
        return 0.0
    a, b = float(expected), float(actual)
    if not math.isfinite(a) or not math.isfinite(b):
        raise DataError(f"{label}: nonfinite comparison")
    delta = b - a
    if not math.isclose(a, b, rel_tol=0, abs_tol=TOLERANCE):
        raise DataError(f"{label}: delta={delta:.12g} exceeds {TOLERANCE}")
    return delta


def compare_scores(current: dict, legacy25: dict, legacy26: dict, split: dict) -> dict:
    """Fail closed on lost frames, source mismatch and unequal mesh scores."""
    expected = split["holdout"]
    if len(expected) != 12 or len(set(expected)) != 12:
        raise DataError("Expected exactly 12 unique B-V1 holdout frames")
    streams = [current["per_frame"], legacy25["per_frame"], legacy26["per_frame"]]
    for label, frames in zip(("PR34", "PR25", "PR26"), streams):
        if [f["name"] for f in frames] != expected:
            raise DataError(f"{label}: holdout frame identity/order changed")
    if current["mesh_sha256"] != legacy25["mesh_sha256"]:
        raise DataError("#25 and #34 evaluated different mesh bytes")
    if current["gt_camera_poses_loaded"] is not False or legacy25["gt_camera_poses_loaded"] is not False:
        raise DataError("GT poses may not enter frozen B evaluation")
    if any(f["status"] != "measured" for f in streams[2]):
        raise DataError("#26 has a zero-overlap frame")
    rows = []
    max_delta = 0.0
    for f34, f25, f26 in zip(*streams):
        s34, s25 = f34["score"], f25["score"]
        if (f34["depth_filename"] != f25["depth_filename"] or
                f34["depth_sha256"] != f25["depth_sha256"] or
                f34["depth_sha256"] != f26["sensor_depth_sha256"]):
            raise DataError(f"{f34['name']}: actual depth source differs")
        row = {
            "name": f34["name"],
            "depth_filename": f34["depth_filename"],
            "depth_sha256": f34["depth_sha256"],
            "valid_pixels": s34["sensor_valid_pixels"],
            "overlap_pixels": s34["overlap_pixels"],
            "coverage": s34["observed_depth_coverage"],
            "mae_m": s34["mae_m"],
            "rmse_m": s34["rmse_m"],
            "p95_abs_m": s34["p95_abs_m"],
            "delta_PR25": {}, "delta_PR26": {},
        }
        for label, other in (("PR25", s25), ("PR26", f26)):
            for key in COUNT_KEYS:
                require_same_number(f"{f34['name']}:{label}:{key}",
                                    s34[key], other[key], integer=True)
            for key in KEYS:
                target_key = "observed_depth_coverage" if key == "coverage" else key
                delta = require_same_number(f"{f34['name']}:{label}:{key}",
                                            s34[target_key], other[target_key])
                row[f"delta_{label}"][key] = delta
                max_delta = max(max_delta, abs(delta))
        rows.append(row)
    overall = {"PR34": {}, "PR25": {}, "PR26": {}}
    for label, data in (("PR34", current), ("PR25", legacy25), ("PR26", legacy26)):
        valid = data["valid_measured_pixels"] if label != "PR26" else data["sensor_valid_pixels"]
        require_same_number(label + ":valid_count", current["valid_measured_pixels"], valid, integer=True)
        require_same_number(label + ":overlap_count", current["overlap_pixels"],
                            data["overlap_pixels"], integer=True)
        for key in KEYS:
            raw_key = ("observed_depth_coverage" if key == "coverage" and label == "PR26"
                       else key)
            require_same_number(label + ":" + key, current[key], data[raw_key])
            overall[label][key] = data[raw_key]
    # Independent regrouping catches per-frame weighting mistakes. Percentiles
    # cannot be reconstructed from summary percentiles; full residuals are used
    # by each evaluator for aggregate P95.
    valid_sum = sum(r["valid_pixels"] for r in rows)
    covered_sum = sum(r["overlap_pixels"] for r in rows)
    require_same_number("per-frame valid sum", current["valid_measured_pixels"], valid_sum, integer=True)
    require_same_number("per-frame overlap sum", current["overlap_pixels"], covered_sum, integer=True)
    require_same_number("weighted coverage", current["coverage"], covered_sum / valid_sum)
    weighted_mae = sum(r["overlap_pixels"] * r["mae_m"] for r in rows) / covered_sum
    weighted_rmse = math.sqrt(sum(r["overlap_pixels"] * r["rmse_m"] ** 2
                                  for r in rows) / covered_sum)
    require_same_number("weighted MAE", current["mae_m"], weighted_mae)
    require_same_number("weighted RMSE", current["rmse_m"], weighted_rmse)
    return {"per_frame": rows, "overall": overall, "max_abs_numeric_delta": max_delta,
            "weighted_mae_m": weighted_mae, "weighted_rmse_m": weighted_rmse}


def run(work: Path, results: Path, path25: Path, path26: Path) -> dict:
    split = read_json(results / "b-48train-12holdout.json")
    metric = read_json(results / "metric" / "metric-trajectory-from-sensor.json")
    metric_report = read_json(results / "metric" / "report.json")
    source = metric_report["test_video"]["source_rgb_manifest"]
    if split.get("protocol", {}).get("ab_version") != "legacy-B-V1":
        raise DataError("#26 frozen protocol supports B-V1 only; AB-V2 comparisons are invalid")
    train, held = split_names(source)
    if [x["name"] for x in split["train"]] != train or split["holdout"] != held:
        raise DataError("Actual split does not equal original Issue #15 B-V1")
    if (split.get("depth_numeric_units_per_meter") != 5000 or
            metric.get("units") != "meters" or
            metric.get("scale_is_from_external_gt_poses") is not False or
            metric.get("metric_scale_estimated_from_measured_depth") is not True):
        raise DataError("Camera-Z depth / sensor anchored metric scale provenance mismatch")
    if len(metric.get("anchor_diagnostics", [])) != 3:
        raise DataError("Scale must have three anchored views")
    anchors = [a["name"] for a in metric["anchor_diagnostics"]]
    if len(set(anchors)) != 3 or set(anchors) & set(held):
        raise DataError("Scale anchors overlap heldout data")
    dataset = work / "unpacked" / ROOT
    train_paths = [safe_path(dataset, x["depth"]).resolve() for x in split["train"]]
    if len(set(train_paths)) != 48:
        raise DataError("Training data path alias or reuse")
    sfm = work / "rgb-only-colmap"
    sfm_report = read_json(sfm / "report.json")
    model_path = sfm / sfm_report["result"]["sfm_model_dir"]
    kwargs = {
        "mesh_path": results / "mesh" / "mesh.ply",
        "training_recording": work / "b-sensor-recording" / "recording.json",
        "metric": metric, "split": split, "source": source,
        "dataset_root": dataset, "model_path": model_path,
        "downsample": DOWNSAMPLE, "max_gap_s": .03, "max_depth_m": 4.,
    }
    original25 = load_frozen(path25, "echo4d._issue_a_pr25")
    original26 = load_frozen(path26, "_issue_a_pr26")
    current = evaluate_sensor_metric_holdouts(**kwargs)
    old25 = original25.evaluate_sensor_metric_holdouts(**kwargs)
    old26 = original26.evaluate_locked_B_run(work, results, {})
    comparison = compare_scores(current, old25, old26, split)
    return {
        "status": "passed_same_mesh_B_V1_cross_score",
        "scope": "Shared rasterizer and sensor reader; frozen evaluator entrypoints, not an independent geometry oracle",
        "protocol": "legacy-B-V1 (Issue #15); NOT the AB-V2-final A/B protocol",
        "source_PR34": "d030f615aae94b70218badb28c29300da7894285",
        "source_PR25": PR25_SHA, "source_PR26": PR26_SHA,
        "source_code_sha256": {
            "PR25": hashlib.sha256(path25.read_bytes()).hexdigest(),
            "PR26": hashlib.sha256(path26.read_bytes()).hexdigest(),
        },
        "mesh_sha256": file_sha256(kwargs["mesh_path"]),
        "train_frames": 48, "holdout_frames": 12,
        "depth_units_per_meter": 5000, "camera_poses": "estimated metric SfM, no GT",
        **comparison,
        "limits": [
            "Old #26 was hard-bound to B-V1; AB-V2-final not compared here.",
            "All three evaluators share rasterize_mesh_depth, read_measured_depth and source imagery.",
            "Same-sensor consistency does not prove independent physical surface accuracy.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--legacy25", type=Path, required=True)
    parser.add_argument("--legacy26", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        report = run(args.work.resolve(), args.report_dir.resolve(),
                     args.legacy25.resolve(), args.legacy26.resolve())
        rc = 0
    except Exception as exc:
        report = {"status": "failed", "error_type": type(exc).__name__,
                  "error": str(exc), "source_PR25": PR25_SHA, "source_PR26": PR26_SHA}
        rc = 1
    (args.output / "cross_score.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    md = ["# PR #34 / Issue A same-Mesh cross-score", "",
          f"- Status: {report['status']}",
          f"- Protocol: {report.get('protocol', 'not completed')}",
          f"- Mesh SHA256: {report.get('mesh_sha256', 'not completed')}",
          f"- Max absolute numeric delta: {report.get('max_abs_numeric_delta', 'not completed')}",
          f"- Failure: {report.get('error', 'none')}", "",
          "See cross_score.json for 12 per-frame counts, coverage, MAE, RMSE, P95 and deltas.",
          "This is same-sensor view-consistency, not independently scanned 3D surface truth."]
    (args.output / "SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
