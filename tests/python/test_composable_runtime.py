"""M2 provider bridge: real CPU invocation on analytic data; no real-scene claim."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from echo4d.composable_runtime import main
from echo4d.demo import create_demo
from echo4d.model import file_sha256


def command(recording: Path, output: Path, receipt: Path, *, sha: str, source: str = "synthetic",
            pose: str = "generated", depth: str = "generated") -> list[str]:
    return ["--recording", str(recording), "--expected-manifest-sha256", sha,
            "--output", str(output), "--receipt", str(receipt),
            "--source", source, "--pose-evidence", pose, "--depth-evidence", depth,
            "--scene-id", "fixture", "--coordinate-id", "fixture_world", "--time-id", "fixture_pts",
            "--bounds", "-1.8", "-.2", "-1.9", "1.8", "2.2", "1.8", "--voxel", ".06"]


def test_analytic_recording_runs_actual_tsdf_provider_and_hashes(tmp_path):
    data, out, receipt = tmp_path / "data", tmp_path / "mesh", tmp_path / "receipt.json"
    create_demo(data, frames=8)
    manifest = data / "recording.json"
    assert main(command(manifest, out, receipt, sha=file_sha256(manifest))) == 0
    result = json.loads(receipt.read_text(encoding="utf-8"))
    assert result["status"] == "completed_unreviewed"
    assert result["modules"][0]["id"] == "echo_cpu_tsdf"
    assert result["frames"] == 8 and result["triangles"] > 0
    assert result["output_mesh_sha256"] == file_sha256(out / "mesh.ply")
    assert result["provider_report_sha256"] == file_sha256(out / "report.json")
    assert result["evidence_level"] == "E2_synthetic"
    assert result["independent_surface_truth"] is False
    assert result["human_visual_review"] is False
    assert len(result["route_id"]) == 64


def test_wrong_frozen_manifest_fails_before_geometry_and_keeps_failure_receipt(tmp_path):
    data, out, receipt = tmp_path / "data", tmp_path / "mesh", tmp_path / "failure.json"
    create_demo(data, frames=2)
    assert main(command(data / "recording.json", out, receipt, sha="0" * 64)) == 2
    result = json.loads(receipt.read_text(encoding="utf-8"))
    assert result["status"] == "failed" and "SHA256" in result["error"]
    assert not out.exists()


@pytest.mark.parametrize("source,pose,depth", [
    ("recorded", "measured", "measured"),
    ("synthetic", "measured", "generated"),
    ("synthetic", "generated", "measured"),
])
def test_source_and_evidence_mismatch_rejected(tmp_path, source, pose, depth):
    data, out, receipt = tmp_path / "data", tmp_path / "mesh", tmp_path / "failure.json"
    create_demo(data, frames=2)
    manifest = data / "recording.json"
    assert main(command(manifest, out, receipt, sha=file_sha256(manifest),
                        source=source, pose=pose, depth=depth)) == 2
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "failed"
    assert not out.exists()


def test_existing_receipt_cannot_be_overwritten(tmp_path):
    data, out, receipt = tmp_path / "data", tmp_path / "mesh", tmp_path / "failure.json"
    create_demo(data, frames=2)
    receipt.write_text("original", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(command(data / "recording.json", out, receipt, sha=file_sha256(data / "recording.json")))
    assert receipt.read_text(encoding="utf-8") == "original"
    assert not out.exists()
