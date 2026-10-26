"""Real TUM sensor PNG -> indexed raw depth -> strict COLMAP core roundtrip.

One indexed importer only: echo4d import-colmap. A TUM *groundtruth-derived*
COLMAP TEXT model is used solely to validate formatting and coordinate/unit
conversions, NOT to claim RGB SfM, independent monocular scale or surface GT.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from echo4d.model import DataError, file_sha256, load_recording, safe_path
from tum_public_roundtrip import (
    URL, ROOT, PINNED_ARCHIVE_SHA256, PINNED_ARCHIVE_BYTES,
    download, extract, known_pose_colmap,
)

RAW_DEPTH_UNITS_PER_METRE = 5000
FRAME_LIMIT = 12
STRIDE = 12
POSE_TOLERANCE = 1e-9
DEPTH_TOLERANCE_M = 1e-6


def invoke(name: str, args: list[str], receipt: dict, *, expected_rc: int = 0) -> dict:
    entry = {"name": name, "argv": args, "status": "running"}
    receipt.setdefault("commands", []).append(entry)
    proc = subprocess.run(args, capture_output=True, text=True, timeout=900)
    entry.update({"returncode": proc.returncode, "stdout_tail": proc.stdout[-1500:],
                  "stderr_tail": proc.stderr[-3000:]})
    entry["status"] = "passed" if proc.returncode == expected_rc else "failed"
    if proc.returncode != expected_rc:
        raise DataError(f"{name}: expected exit={expected_rc}; got={proc.returncode}: {proc.stderr[-700:]}")
    if expected_rc and proc.stdout.strip():
        raise DataError(f"{name}: failed importer unexpectedly emitted success JSON")
    return entry


def assess_real_roundtrip(baseline: Path, indexed: Path, source: Path,
                          associations: list[dict], provenance: dict) -> dict:
    a = load_recording(baseline / "recording.json")
    b = load_recording(indexed / "recording.json")
    if len(a) != FRAME_LIMIT or len(b) != FRAME_LIMIT or a.camera != b.camera:
        raise DataError("Real TUM frame count or camera intrinsics changed")
    indexed_metadata = provenance.get("indexed_depth")
    if not isinstance(indexed_metadata, dict):
        raise DataError("No published indexed depth evidence")
    if indexed_metadata.get("raw_units_per_metre") != RAW_DEPTH_UNITS_PER_METRE:
        raise DataError("Raw sensor depth scale metadata changed")
    if indexed_metadata.get("raw_unit_label_verified_independently") is not False:
        raise DataError("Human-supplied unit label must not be marked independently proven")
    items = {e["image"]: e for e in indexed_metadata["frames"]}
    if len(items) != FRAME_LIMIT or len(associations) != FRAME_LIMIT:
        raise DataError("Missing raw-source per-frame provenance")
    max_pose = max_timestamp = max_depth = 0.
    depth_pixels = nonzero_pixels = 0
    per_frame = []
    for i, ((_, aa, ea), (_, bb, eb), assoc) in enumerate(
        zip(a.iter_frames(), b.iter_frames(), associations, strict=True)
    ):
        name = PurePosixPath(ea["color"]).name
        item = items.get(name)
        if item is None or item["raw_depth"] != assoc["depth_path"]:
            raise DataError("Indexed raw depth does not match TUM ground source association")
        original = safe_path(source, assoc["depth_path"])
        with Image.open(original) as image:
            raw = np.asarray(image)
        if raw.shape != aa.depth.shape or raw.dtype.kind not in ("u", "i"):
            raise DataError("Real source is not same-resolution integer TUM camera-Z PNG")
        if file_sha256(original) != assoc["depth_sha256"] or (
            file_sha256(original) != item["raw_depth_sha256"]
        ):
            raise DataError("Real original source PNG checksum differs")
        if not np.array_equal(np.rint(bb.depth.astype(np.float64) *
                                     RAW_DEPTH_UNITS_PER_METRE).astype(np.int64),
                              raw.astype(np.int64)):
            raise DataError("Indexed metric depth failed to roundtrip to the real integer source")
        if not np.array_equal(aa.color, bb.color):
            raise DataError("Real registered RGB pixels altered")
        p = float(np.max(np.abs(aa.camera_to_world - bb.camera_to_world)))
        t = abs(aa.t - bb.t)
        d = float(np.max(np.abs(aa.depth - bb.depth)))
        max_pose, max_timestamp, max_depth = max(max_pose, p), max(max_timestamp, t), max(max_depth, d)
        if p > POSE_TOLERANCE or t > POSE_TOLERANCE or d > DEPTH_TOLERANCE_M:
            raise DataError(f"Real indexed values differed beyond FP/pose tolerances at {name}")
        if not np.allclose(aa.camera_to_world[:3, :3], bb.camera_to_world[:3, :3],
                           rtol=0, atol=POSE_TOLERANCE):
            raise DataError("COLMAP camera rotation changed during metric scale conversion")
        depth_file = indexed / "depth" / str(PurePosixPath(name).with_suffix(".npy"))
        if file_sha256(depth_file) != item["metric_depth_sha256"]:
            raise DataError("Published metre-valued NPY checksum differs")
        pixels = int(raw.size)
        valid = int(np.count_nonzero(raw))
        depth_pixels += pixels
        nonzero_pixels += valid
        per_frame.append({
            "name": name, "source_png_path": assoc["depth_path"],
            "source_png_sha256": item["raw_depth_sha256"],
            "indexed_metric_npy_sha256": item["metric_depth_sha256"],
            "pixels": pixels, "nonzero_sensor_pixels": valid,
            "max_camera_matrix_delta": p,
            "timestamp_delta_s": t,
            "max_metric_depth_delta_m": d,
            "raw_source_integer_roundtrip_exact": True,
        })
    if depth_pixels == 0 or nonzero_pixels == 0:
        raise DataError("No real sensor depth pixels evaluated")
    return {
        "frames": len(per_frame),
        "real_source_pixels": depth_pixels,
        "real_nonzero_sensor_pixels": nonzero_pixels,
        "max_pose_element_delta": max_pose,
        "max_timestamp_delta_s": max_timestamp,
        "max_metric_depth_delta_m": max_depth,
        "integer_source_roundtrip_exact": True,
        "registered_rgb_exact": True,
        "per_frame": per_frame,
    }


def experiment(args: argparse.Namespace, report: dict) -> None:
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    archive = work / "source.tgz"
    if args.archive is not None:
        candidate = args.archive.resolve()
        if not candidate.is_file():
            raise DataError("Offline TUM archive is missing")
        shutil.copyfile(candidate, archive)
        if (file_sha256(archive) != PINNED_ARCHIVE_SHA256 or
                archive.stat().st_size != PINNED_ARCHIVE_BYTES):
            raise DataError("Offline archive hash/size differs from pinned public TUM source")
        report["download"] = {"source": "offline supplied; URL not re-verified",
                              "sha256": file_sha256(archive), "bytes": archive.stat().st_size}
    else:
        download(archive, report)
    report["archive_verified"] = True
    report["extract"] = extract(archive, work / "dataset")
    root = work / "dataset" / ROOT
    baseline = work / "baseline-from-TUM"
    invoke("baseline_TUM_known_poses", [
        sys.executable, "-m", "echo4d", "import-tum", str(root),
        "--output", str(baseline),
        "--intrinsics", "525", "525", "319.5", "239.5",
        "--stride", str(STRIDE), "--limit", str(FRAME_LIMIT),
        "--downsample", "1", "--world-up", "z",
    ], report)
    model = work / "gt-as-COLMAP-TEXT"
    report["model_source"] = known_pose_colmap(baseline, model)
    assoc_file = baseline / "association.json"
    original_assoc = json.loads(assoc_file.read_text(encoding="utf-8"))
    associations = original_assoc["frames"]
    baserec = load_recording(baseline / "recording.json")
    if len(baserec) != FRAME_LIMIT or len(associations) != FRAME_LIMIT:
        raise DataError("Pinned TUM import could not supply twelve matched real frames")
    index = work / "raw-depth-index.txt"
    with index.open("w", encoding="utf-8") as file:
        for entry, assoc in zip(baserec.entries, associations, strict=True):
            image_name = PurePosixPath(entry["color"]).name
            depth_name = assoc["depth_path"]
            if any(ch.isspace() for ch in (image_name + depth_name)):
                raise DataError("Unsafe real source filename for canonical whitespace index")
            file.write(f"{image_name} {depth_name}\n")
    report["index_sha256"] = file_sha256(index)
    report["source_association_sha256"] = file_sha256(assoc_file)
    imported = work / "indexed-colmap-output"
    command = [
        sys.executable, "-m", "echo4d", "import-colmap",
        "--model", str(model), "--images", str(baseline / "color"),
        "--depth", str(root), "--depth-index", str(index),
        "--depth-scale", str(RAW_DEPTH_UNITS_PER_METRE),
        "--depth-unit", "TUM registered PNG raw 5000 units per metre",
        "--timestamps", str(model / "timestamps.csv"),
        "--meters-per-colmap-unit", "1",
        "--scale-source", "TUM known camera trajectory supplied in metres; conversion-only GT fixture",
        "--output", str(imported)
    ]
    invoke("single_core_indexed_real_TUM_PNG", command, report)
    provenance = json.loads((imported / "import-provenance.json").read_text(encoding="utf-8"))
    report["real_source_roundtrip"] = assess_real_roundtrip(
        baseline, imported, root, associations, provenance)
    report["published_recording_sha256"] = file_sha256(imported / "recording.json")
    report["published_provenance_sha256"] = file_sha256(imported / "import-provenance.json")

    # The first source frame really is a 16-bit sensor integer PNG; unit=1
    # should be rejected because raw counts would become implausible metres.
    bad = list(command)
    bad[bad.index("--depth-scale") + 1] = "1"
    bad[bad.index("--output") + 1] = str(work / "negative-wrong-units")
    invoke("real_TUM_wrong_units_rollback", bad, report, expected_rc=2)
    if (work / "negative-wrong-units").exists():
        raise DataError("Invalid raw PNG units published partial recording")

    # Another real-data negative path: exact index coverage is compulsory.
    invalid = work / "missing-one-depth-index.txt"
    invalid.write_text("\n".join(index.read_text(encoding="utf-8").splitlines()[:-1]) + "\n",
                       encoding="utf-8")
    bad = list(command)
    bad[bad.index("--depth-index") + 1] = str(invalid)
    bad[bad.index("--output") + 1] = str(work / "negative-missing-frame")
    invoke("real_TUM_missing_index_rollback", bad, report, expected_rc=2)
    if (work / "negative-missing-frame").exists():
        raise DataError("Partial index published a partial recording")
    report["rollback_negative_tests"] = ["wrong_raw_units", "missing_index_frame"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="Offline re-run, must match pinned official SHA")
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "issue47-A-single-indexed-COLMAP-real-TUM-PNG-v1",
        "start_utc": datetime.now(timezone.utc).isoformat(),
        "actual_checkout_sha": os.environ.get("ECHO_TESTED_CHECKOUT_SHA"),
        "source_pr39_base": "8bb15b646ce6fadfb2bd54b85655901e26dd99f9",
        "github_run": (f"https://github.com/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/"
                       f"{os.environ.get('GITHUB_RUN_ID')}" if os.environ.get("GITHUB_RUN_ID") else None),
        "source_url": URL,
        "pinned_source_archive_sha256": PINNED_ARCHIVE_SHA256,
        "source_bytes": PINNED_ARCHIVE_BYTES,
        "parameters": {"frames": FRAME_LIMIT, "stride": STRIDE, "downsample": 1,
                       "depth_raw_units_per_metres": RAW_DEPTH_UNITS_PER_METRE,
                       "pose_tolerance": POSE_TOLERANCE,
                       "depth_tolerance_m": DEPTH_TOLERANCE_M},
        "scope": "actual public raw sensor PNG/index against TUM known-pose metric importer; no COLMAP SfM",
        "independent_scene_surface_truth": False,
        "automated_independent_scale_proof": False,
        "commands": [],
        "status": "running",
    }
    start = time.perf_counter()
    rc = 0
    try:
        experiment(args, report)
        report["status"] = "passed_real_source_indexed_depth_contract_not_surface_accuracy"
    except Exception as exc:
        rc = 1
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        print(f"INDEXED_TUM_FAILURE: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        (args.report_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8")
        result = report.get("real_source_roundtrip", {})
        lines = [
            "# Issue #47 A — real indexed TUM raw-PNG source roundtrip", "",
            f"- Status: {report['status']}",
            f"- Run: {report['github_run']}",
            f"- Actual checkout: {report['actual_checkout_sha']}",
            f"- Archive pinned SHA256: {PINNED_ARCHIVE_SHA256}",
            f"- Real frames: {result.get('frames', 'not completed')}",
            f"- Real depth pixels: {result.get('real_source_pixels', 'not completed')}",
            f"- Max metric depth difference (m): {result.get('max_metric_depth_delta_m', 'not completed')}",
            f"- Integer PNG reconstruction: {result.get('integer_source_roundtrip_exact', 'not completed')}",
            f"- Negative rollbacks: {report.get('rollback_negative_tests', 'not completed')}",
            f"- Failure: {report.get('error', {}).get('message', 'none')}", "",
            "TUM groundtruth-derived COLMAP TEXT proves format/unit transformation only.",
            "It is NOT video pose estimation, independent monocular scale or scanned-surface truth."
        ]
        (args.report_dir / "SUMMARY.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "report": str(args.report_dir / "report.json")}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
