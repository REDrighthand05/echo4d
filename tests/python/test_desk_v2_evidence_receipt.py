"""Fail-closed regression for the immutable real desk V2 *failed* experiment receipt.

This checks internal arithmetic, frozen roles, and provenance pins only.
It cannot re-hash expired Actions artifacts or prove independent surface truth.
Run: python -m pytest -q tests/python/test_desk_v2_evidence_receipt.py
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re

import pytest


RECEIPT = (
    Path(__file__).resolve().parents[2]
    / "docs/evidence/issue47-desk-v2-real-heldout-failure-20261027.json"
)
HELDOUT = (4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59)
FIELDS = (
    "holdout_index", "original_depth_png_sha256", "valid_pixels",
    "overlap_pixels", "coverage_rounded_6dp", "rmse_m_rounded_6dp",
    "mae_m_rounded_6dp", "p95_abs_m_rounded_6dp",
)
PINNED = {
    "candidate_head_sha": "04501849ba700dcf3c32fe9fd7b7ea57c2edba8c",
    "actual_tested_merge_checkout_sha": "00f51a9e81a62ed8f807f6042ab5174257d2bc1d",
    "official_archive_sha256": "e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c",
    "source_plan_sha256": "368546ed7f3137078992289b0d4f23ce2ac03bf471e87ddedbe325bbdf2e95f2",
    "frozen_protocol_sha256": "5649a7372eb70bc881b46a6867833e89ef3b0d8ec1e306a7af02cc2458151736",
    "heldout_report_sha256": "33a2206079430bf46288e8dce5a9ee4b183bf2336782c5732c1a932b90a7474b",
    "mesh_sha256": "320da61fcd421c0e55aa6fc014750aeaa3e6e2d98b607e9feaae268f83d2a939",
    "artifact_zip_sha256": "d6bffd3eb08acccbc7e28b74f2a087c48972408683c2ef8eec324a290ea074a4",
}
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
HEX_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")


def verify_receipt(data: dict) -> None:
    """Reject silent edits, missing holdouts, misleading PASS, or bad aggregation."""
    assert data["schema"] == "echo4d-desk-v2-real-failed-gate-evidence-1"
    provenance = data["provenance"]
    for key, expected in PINNED.items():
        assert provenance[key] == expected
        validator = HEX_GIT_SHA if key in ("candidate_head_sha", "actual_tested_merge_checkout_sha") else HEX_SHA256
        assert validator.fullmatch(provenance[key])
    assert provenance["base_prs"] == [58, 59, 60]

    contract = data["fixed_contract"]
    assert contract["version"] == "desk-metric-mesh-v2-source-paired"
    assert (
        contract["selected_rgb"], contract["geometry_train"],
        contract["geometry_holdout"], contract["scale_anchors"],
        contract["scale_diagnostics"],
    ) == (75, 48, 12, 3, 7)
    assert contract["first_selected_original_rgb_index"] == 17
    assert contract["source_rows_rejected_before_selection"] == 17
    assert contract["max_pairing_gap_s"] == 0.03
    assert contract["voxel_m"] == 0.07
    assert contract["min_global_coverage"] == 0.8
    assert contract["max_global_rmse_m"] == 0.15

    observed = data["observed"]
    assert observed["registered_rgb"] == 75
    assert observed["source_pairs_within_30ms"] == observed["source_pairs_required"] == 68
    assert observed["mesh_vertices"] == 4543
    assert observed["mesh_triangles"] == 7578
    assert math.isfinite(observed["measured_scale_m_per_sfm_unit"])
    assert observed["measured_scale_m_per_sfm_unit"] > 0

    assert tuple(data["per_view_fields"]) == FIELDS
    rows = data["per_view"]
    assert len(rows) == len(HELDOUT)
    assert tuple(row[0] for row in rows) == HELDOUT
    assert len({row[1] for row in rows}) == len(HELDOUT)
    for row in rows:
        assert len(row) == len(FIELDS)
        _, depth_sha, valid, overlap, coverage, rmse, mae, p95 = row
        assert HEX_SHA256.fullmatch(depth_sha)
        assert type(valid) is int and type(overlap) is int
        assert 0 < overlap <= valid
        assert math.isclose(coverage, overlap / valid, abs_tol=1e-6)
        assert all(math.isfinite(v) and v >= 0 for v in (rmse, mae, p95))
        assert mae <= rmse + 1e-6

    valid_total = sum(row[2] for row in rows)
    overlap_total = sum(row[3] for row in rows)
    assert (valid_total, overlap_total) == (169467, 158851)
    checks = data["independent_checks"]
    assert checks["per_view_valid_pixels_sum"] == valid_total
    assert checks["per_view_overlap_pixels_sum"] == overlap_total
    assert observed["holdout_valid_pixels"] == valid_total
    assert observed["holdout_overlap_pixels"] == overlap_total
    coverage = overlap_total / valid_total
    assert math.isclose(observed["coverage"], coverage, abs_tol=1e-12)

    # Individual scores are rounded to 6dp; aggregate comparison needs tolerance.
    squared = [(row[0], row[3] * row[5] ** 2) for row in rows]
    total_squared = sum(score for _, score in squared)
    rmse = math.sqrt(total_squared / overlap_total)
    mae = sum(row[3] * row[6] for row in rows) / overlap_total
    assert math.isclose(observed["overlap_rmse_m"], rmse, abs_tol=2e-6)
    assert math.isclose(observed["overlap_mae_m"], mae, abs_tol=2e-6)
    assert math.isclose(checks["weighted_rmse_from_per_view_approx_m"], rmse, abs_tol=2e-6)

    ranked = sorted(squared, key=lambda item: item[1], reverse=True)
    assert [index for index, _ in ranked[:3]] == checks["top_three_squared_error_frames"]
    assert [index for index, _ in ranked[:3]] == [4, 59, 54]
    fraction = sum(score for _, score in ranked[:3]) / total_squared
    assert math.isclose(
        checks["top_three_fraction_of_squared_error_approx"], fraction,
        abs_tol=2e-6,
    )

    coverage_pass = coverage >= contract["min_global_coverage"]
    rmse_pass = rmse <= contract["max_global_rmse_m"]
    assert observed["coverage_gate_pass"] is coverage_pass is True
    assert observed["rmse_gate_pass"] is rmse_pass is False
    assert observed["final_result"] == "FAIL_FROZEN_15CM_RMSE_GATE"
    assert data["limits"] and any("not an independent" in line for line in data["limits"])
    # Reported external-hash comparisons are historical claims, not reverified here.
    assert all(
        checks[key] is True for key in (
            "zip_sha_matches_actions_log", "mesh_sha_matches_raw_ply",
            "source_plan_sha_matches_protocol_and_report",
            "protocol_sha_matches_report", "holdout_sha_matches_report",
        )
    )


def test_real_desk_v2_failure_receipt_is_internally_consistent():
    verify_receipt(json.loads(RECEIPT.read_text(encoding="utf-8")))


@pytest.mark.parametrize("mutation", (
    "missing_holdout", "duplicated_holdout", "faked_pass", "altered_rmse",
    "altered_overlap", "wrong_mesh_hash", "altered_top_three",
    "changed_frozen_threshold",
))
def test_desk_v2_receipt_tampering_fails_closed(mutation):
    data = deepcopy(json.loads(RECEIPT.read_text(encoding="utf-8")))
    if mutation == "missing_holdout":
        data["per_view"].pop()
    elif mutation == "duplicated_holdout":
        data["per_view"][1][1] = data["per_view"][0][1]
    elif mutation == "faked_pass":
        data["observed"]["rmse_gate_pass"] = True
        data["observed"]["final_result"] = "PASS"
    elif mutation == "altered_rmse":
        data["observed"]["overlap_rmse_m"] = 0.10
    elif mutation == "altered_overlap":
        data["per_view"][0][3] -= 1
    elif mutation == "wrong_mesh_hash":
        data["provenance"]["mesh_sha256"] = "0" * 64
    elif mutation == "altered_top_three":
        data["independent_checks"]["top_three_squared_error_frames"] = [9, 14, 34]
    elif mutation == "changed_frozen_threshold":
        data["fixed_contract"]["max_global_rmse_m"] = 0.20
    with pytest.raises(AssertionError):
        verify_receipt(data)
