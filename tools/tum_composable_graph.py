"""Pinned public TUM fr1_xyz -> known-pose RGB-D -> typed M2 graph -> real Mesh.

A distinct execution pathway, NOT a second reconstruction algorithm, RGB-only
reconstruction, independent surface truth, or a repair of frozen desk V2.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from echo4d.composable import ArtifactType, ModuleSpec, compose
from echo4d.graph_runner import InputFile, archive_pinned_files, execute_graph, tsdf_recording_adapter
from echo4d.model import DataError, file_sha256, load_recording
from tum_public_roundtrip import (BOUNDS, PINNED_ARCHIVE_BYTES,
    PINNED_ARCHIVE_SHA256, ROOT, URL, VOXEL, download, extract)

TOTAL = 60
HOLDOUT = (4, 5, 9, 10, 14, 19, 24, 25, 29, 34, 35, 55)
IMPORT_FLAGS = ("--intrinsics", "525", "525", "319.5", "239.5",
                "--stride", "12", "--limit", "60", "--downsample", "4",
                "--world-up", "z")


def split_training(data: dict) -> dict:
    """Frozen 48/12 split; no held-out depth pixels consulted."""
    if not isinstance(data, dict) or data.get("version") != "echo-dataset-1" or data.get("source") != "recorded":
        raise DataError("Requires recorded ECHO dataset; synthetic source rejected")
    frames = data.get("frames")
    if not isinstance(frames, list) or len(frames) != TOTAL:
        raise DataError("Expected exactly 60 imported TUM frames")
    if len(set(HOLDOUT)) != 12 or max(HOLDOUT) >= TOTAL:
        raise DataError("Invalid predeclared holdout indices")
    result = dict(data)
    result["frames"] = [frame for i, frame in enumerate(frames) if i not in HOLDOUT]
    if len(result["frames"]) != 48:
        raise DataError("Train/holdout overlap")
    result["notes"] = "TUM xyz fixed 60: known GT pose + sensor depth; 48 train, 12 untouched holdout; not desk V2"
    return result


def run(args: argparse.Namespace, report: dict) -> None:
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    checkout = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
        text=True, timeout=5).strip()
    if os.environ.get("ECHO_HEAD_SHA") and checkout != os.environ["ECHO_HEAD_SHA"]:
        raise DataError("Checkout differs from pinned Actions head")
    report["checkout_sha"] = checkout
    report["source"] = {
        "url": URL, "archive_sha256": PINNED_ARCHIVE_SHA256,
        "archive_bytes": PINNED_ARCHIVE_BYTES,
        "pose": "TUM groundtruth.txt interpolated, not COLMAP SfM",
        "depth": "TUM measured Z, 1/5000 units per metre",
        "rgb_source_resolution": [640, 480],
        "algorithm_resolution": [160, 120]}
    archive = work / "source.tgz"
    if args.archive is None:
        download(archive, report)
    else:
        supplied = args.archive.resolve()
        if not supplied.is_file() or supplied.stat().st_size != PINNED_ARCHIVE_BYTES:
            raise DataError("Offline archive missing or wrong byte count")
        shutil.copyfile(supplied, archive)
        report["download"] = {"method": "offline, origin unverified",
            "bytes": archive.stat().st_size, "sha256": file_sha256(archive)}
    if (report["download"]["sha256"] != PINNED_ARCHIVE_SHA256 or
            report["download"]["bytes"] != PINNED_ARCHIVE_BYTES):
        raise DataError("TUM archive differs from pinned source SHA and size")
    report["extraction"] = extract(archive, work / "dataset")
    imported = work / "imported"
    command = [sys.executable, "-m", "echo4d", "import-tum",
               str(work / "dataset" / ROOT), "--output", str(imported), *IMPORT_FLAGS]
    report["import_command"] = command
    start = time.perf_counter()
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=600)
        report["import_exit_code"] = process.returncode
        report["import_stderr_tail"] = process.stderr[-2000:]
        if process.returncode:
            raise RuntimeError("Real TUM import failed; inspect stderr")
    finally:
        report["import_seconds"] = round(time.perf_counter() - start, 3)
    original = imported / "recording.json"
    training = split_training(json.loads(original.read_text(encoding="utf8")))
    manifest = imported / "recording-train.json"
    manifest.write_text(json.dumps(training, sort_keys=True, ensure_ascii=False,
                                  allow_nan=False) + "\n", encoding="utf8")
    recording = load_recording(manifest)
    if len(recording) != 48 or recording.source != "recorded":
        raise DataError("Invalid real training manifest")
    report["input"] = {
        "original_manifest_sha256": file_sha256(original),
        "train_manifest_sha256": file_sha256(manifest),
        "association_sha256": file_sha256(imported / "association.json"),
        "train_indices": [i for i in range(TOTAL) if i not in HOLDOUT],
        "holdout_indices": list(HOLDOUT), "train_frames": 48, "holdout_frames": 12,
        "camera": recording.camera.to_dict()}

    common = dict(scene_id="tum_fr1_xyz_fixed60", coordinate_id="tum_z_up",
                  partition="train", length_unit="m", handedness="right")
    inp = ArtifactType("echo_recording", "world", "metric", "frame_pts",
                       "measured", time_id="tum_sensor_time", time_unit="s",
                       pose_direction="none", **common)
    out = ArtifactType("mesh", "world", "metric", "none",
                       "estimated", time_id="none", time_unit="none",
                       pose_direction="none", **common)
    from echo4d import graph_runner
    provider_file = Path(graph_runner.__file__).resolve()
    module = ModuleSpec("known_pose_sensor_tsdf", (inp,), (out,), 1, True,
                        file_sha256(provider_file))
    routes = compose((inp,), (module,), out)
    if len(routes.routes) != 1:
        raise DataError("Expected exactly one executable typed route")
    adapter = tsdf_recording_adapter(module.module_id, provider_file, inp, out,
        bounds=tuple(float(v) for v in BOUNDS), voxel=float(VOXEL), max_voxels=4_000_000)
    graph_dir = work / "graph"
    start = time.perf_counter()
    execution = execute_graph(routes.routes[0], (module,), {module.module_id: adapter},
        {inp: InputFile(manifest, file_sha256(manifest))}, out,
        graph_dir, checkout_sha=checkout)
    report["graph_seconds"] = round(time.perf_counter() - start, 3)
    report["graph"] = execution
    output = execution["outputs"][out.label()]
    mesh = graph_dir / output["path"]
    if file_sha256(mesh) != output["sha256"]:
        raise DataError("Mesh output differs from graph receipt")
    shutil.copyfile(mesh, args.report_dir / "mesh.ply")
    shutil.copyfile(graph_dir / "execution.json", args.report_dir / "graph-execution.json")
    if file_sha256(args.report_dir / "mesh.ply") != output["sha256"]:
        raise DataError("Copied Mesh SHA mismatch")
    provider_report = mesh.parent / "report.json"
    report["archived_metadata"] = archive_pinned_files({
        "full-recording.json": InputFile(original, report["input"]["original_manifest_sha256"]),
        "train-recording.json": InputFile(manifest, report["input"]["train_manifest_sha256"]),
        "association.json": InputFile(imported / "association.json", report["input"]["association_sha256"]),
        "provider-report.json": InputFile(provider_report, file_sha256(provider_report)),
    }, args.report_dir / "metadata")
    report["mesh_sha256"] = output["sha256"]
    report["mesh_bytes"] = output["bytes"]
    report["status"] = "executed_unscored"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="Offline pinned archive, optional")
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "experiment": "tum-fr1-xyz-fixed60-typed-graph-tsdf-v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "geometry_quality": "not_evaluated",
        "independent_surface_truth": "not_available",
        "human_visual_review": "not_performed",
        "desk_v2_frozen": "FAIL 16.7779cm > 15cm; unchanged",
        "issue_74": "FAIL / NOT ACCEPTED",
        "not_claimed": ["RGB-only", "COLMAP SfM", "second algorithm",
                        "independent surface accuracy", "3+2 view acceptance"]}
    started = time.perf_counter()
    try:
        run(args, report)
        code = 0
    except Exception as error:
        code = 1
        report["status"] = "blocked" if isinstance(error, DataError) else "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        print(f"TUM_GRAPH_{report['status'].upper()}: {error}", file=sys.stderr)
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        (args.report_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf8")
        lines = [
            "# Real TUM via typed M2 graph", "",
            f"- Execution: **{report['status']}**",
            f"- Checkout: {report.get('checkout_sha', 'not_checked')}",
            f"- Archive SHA: {report.get('download', {}).get('sha256', 'not_downloaded')}",
            "- Split: 48 training / 12 reserved, not desk V2",
            f"- Mesh SHA: {report.get('mesh_sha256', 'not_produced')}",
            f"- Graph seconds: {report.get('graph_seconds', 'not_run')}",
            "- Independent geometry quality: **NOT EVALUATED**",
            "- #74 visual acceptance: **FAIL**",
            f"- Error: {report.get('error', {}).get('message', 'none')}",
            "", "One known-pose sensor-depth TSDF algorithm, not two strategies."]
        (args.report_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
