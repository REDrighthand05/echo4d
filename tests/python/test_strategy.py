"""Route decisions are user-controlled; E0 combinations must not become E3."""
from __future__ import annotations

import json

import pytest

from echo4d.cli import main
from echo4d.model import DataError
from echo4d.strategy import (
    INPUT_CATALOG, PROFILE_VERSION, ROUTES, render_strategy_markdown,
    strategy_catalog, strategy_plan, validate_profile,
)


def profile(inputs, primary=None, backup=None):
    result = {"version": PROFILE_VERSION, "goal": "investigate_limits",
              "inputs": {key: {"status": "present"} for key in inputs}}
    if primary:
        result["choice"] = {"primary": primary}
        if backup:
            result["choice"]["backup"] = backup
    return result


def route(result, name):
    return next(row for row in result["routes"] if row["id"] == name)


def test_catalog_unique_route_ids_and_registered_input_fields():
    keys = {row["id"] for row in INPUT_CATALOG}
    ids = [row["id"] for row in ROUTES]
    assert len(ids) == len(set(ids))
    assert len(keys) == len(INPUT_CATALOG)
    for item in ROUTES:
        assert item["implementation"] in ("integrated_experiment", "partial",
                                           "not_integrated")
        assert all(group and set(group) <= keys for group in item["requires"])
        assert set(item["optional"]) <= keys
        assert item["checks"] and item["unknown"]
    assert json.loads(json.dumps(strategy_catalog()))["version"]


def test_rich_video_imu_mesh_depth_is_not_silently_reduced_to_rgb_only():
    evidence = profile(["rgb_video", "imu", "existing_mesh", "camera_calibration",
                        "sensor_depth", "time_sync", "mesh_alignment"])
    result = strategy_plan(evidence)
    assert route(result, "rich_video_imu_depth_mesh")["input_match"] == "candidate"
    assert route(result, "video_imu_mesh_no_depth")["input_match"] == "candidate"
    assert result["no_auto_selection"]
    assert result["user_choices"] == []
    assert result["assessment"]["scene_reconstructed"] == "not_evaluated"
    assert result["assessment"]["input_evidence_audited_by_tool"] is False
    assert route(result, "rich_video_imu_depth_mesh")["implementation"] == "not_integrated"
    assert route(result, "known_pose_measured_depth_tsdf")["input_match"] == "missing_inputs"


def test_imu_and_mesh_require_registration_not_fictional_coverage():
    p = profile(["rgb_video", "imu", "existing_mesh", "camera_calibration", "time_sync"])
    result = strategy_plan(p)
    row = route(result, "video_imu_mesh_no_depth")
    assert row["input_match"] == "candidate"
    assert row["implementation"] == "not_integrated"
    assert "Mesh" in row["checks_not_performed"][0]
    assert row["still_unknown"]
    assert result["assessment"]["independent_geometry_gate"] == "not_evaluated"


def test_video_only_and_single_photo_do_not_invent_metric_geometry():
    only_video = strategy_plan(profile(["rgb_video", "camera_calibration"]))
    sfm = route(only_video, "rgb_video_sfm_only")
    assert sfm["input_match"] == "candidate"
    assert "绝对尺度" in sfm["still_unknown"]
    assert route(only_video, "video_sensor_scaled_tsdf")["input_match"] == "missing_inputs"
    only_photo = strategy_plan(profile(["rgb_images"]))
    assert route(only_photo, "multi_view_images")["input_match"] == "candidate"
    assert route(only_photo, "rich_video_imu_depth_mesh")["input_match"] == "missing_inputs"


def test_user_can_select_blocked_routes_but_report_keeps_missing_inputs():
    p = profile(["rgb_video", "camera_calibration"],
                primary="rich_video_imu_depth_mesh", backup="rgb_video_sfm_only")
    result = strategy_plan(p)
    assert not result["no_auto_selection"]
    assert result["user_choices"][0]["input_match"] == "missing_inputs"
    assert result["user_choices"][0]["missing_required_groups"]
    assert result["user_choices"][1]["input_match"] == "candidate"
    assert all(row["requires_user_approval_to_run"] for row in result["user_choices"])


def test_ablations_are_only_input_logic_not_fake_quality_predictions():
    p = profile(["rgb_video", "imu", "existing_mesh", "camera_calibration",
                 "time_sync", "sensor_depth"])
    result = strategy_plan(p)
    removed = {row["removed_input"]: row for row in result["input_ablation"]}
    assert "rich_video_imu_depth_mesh" in removed["imu"]["routes_which_lose_required_input"]
    assert "video_imu_mesh_no_depth" in removed["existing_mesh"]["routes_which_lose_required_input"]
    assert "rgb_video_sfm_only" not in removed["imu"]["routes_which_lose_required_input"]
    assert "不代表真实运行" in removed["imu"]["note"]


def test_checked_is_only_a_claim_and_requires_evidence_pointer():
    data = profile(["rgb_video"])
    data["inputs"]["rgb_video"] = {"status": "checked"}
    with pytest.raises(DataError, match="evidence"):
        validate_profile(data)
    data["inputs"]["rgb_video"]["evidence"] = "official public RGB data SHA256:example"
    result = strategy_plan(data)
    assert not result["assessment"]["input_evidence_audited_by_tool"]
    assert result["profile"]["inputs"]["rgb_video"]["status"] == "checked"


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(version="unknown"),
    lambda p: p.update(goal="do-everything"),
    lambda p: p.update(inputs={"secret_gps_device_id": {"status": "present"}}),
    lambda p: p.update(inputs={"rgb_video": {"status": "yes"}}),
    lambda p: p.update(choice={"primary": "unknown-route"}),
    lambda p: p.update(choice={"primary": "rgb_video_sfm_only",
                               "backup": "rgb_video_sfm_only"}),
    lambda p: p.update(inputs={"rgb_video": {"status": "present",
                                             "unexpected": True}}),
])
def test_invalid_profile_fails_closed(mutation):
    p = profile(["rgb_video"])
    mutation(p)
    with pytest.raises(DataError):
        strategy_plan(p)


def test_catalog_and_plan_have_human_readable_table():
    plain = render_strategy_markdown(strategy_catalog())
    assert "输入信息与策略候选矩阵" in plain
    assert "IMU" in plain
    assert "已有Mesh" in plain
    plan = render_strategy_markdown(strategy_plan(profile(["rgb_video"])))
    assert "用户选择" in plan and "not_evaluated" in plan


def test_unhashable_enums_never_escape_as_unstructured_errors():
    p = profile(["rgb_video"])
    p["goal"] = ["invalid-list"]
    with pytest.raises(DataError, match="goal"):
        strategy_plan(p)
    p = profile(["rgb_video"])
    p["inputs"]["rgb_video"]["status"] = ["invalid-list"]
    with pytest.raises(DataError, match="Input"):
        strategy_plan(p)
    p = profile(["rgb_video"])
    p["choice"] = {"primary": ["invalid-list"]}
    with pytest.raises(DataError, match="route"):
        strategy_plan(p)


def test_duplicate_json_key_is_rejected_not_silently_overwritten(tmp_path, capsys):
    path = tmp_path / "duplicate.json"
    path.write_text('{"version":"echo-strategy-profile-1",'
                    '"inputs":{"rgb_video":{"status":"present"},'
                    '"rgb_video":{"status":"absent"}}}',encoding="utf-8")
    assert main(["strategy", "--profile", str(path), "--format", "json"]) == 2
    output = json.loads(capsys.readouterr().err)
    assert output["type"] == "DataError"
    assert "Duplicate" in output["error"]


def test_cost_license_and_review_snapshot_not_falsely_populated():
    catalog = strategy_catalog()
    assert catalog["catalog_review"]["update_required_if_repo_changed"] is True
    assert all(x["cost_on_this_input"] == "not_measured" for x in catalog["routes"])
    result = strategy_plan(profile(["rgb_video", "camera_calibration"]))
    assert all(x["external_dependency_license_review"] ==
               "not_performed_for_chosen_versions" for x in result["routes"])
    assert result["catalog_review"]["main_sha"]


def test_cli_catalog_and_profile_usable_without_running_reconstruction(tmp_path, capsys):
    assert main(["strategy", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["version"] == "echo-strategy-matrix-1"
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile(["rgb_video", "camera_calibration"])),
                    encoding="utf-8")
    assert main(["strategy", "--profile", str(path), "--format", "json"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["assessment"]["scene_reconstructed"] == "not_evaluated"
    assert route(plan, "rgb_video_sfm_only")["input_match"] == "candidate"
    assert main(["strategy", "--profile", str(tmp_path / "missing.json")]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["type"] == "DataError"


@pytest.mark.parametrize("field", ["goal", "status", "primary", "backup"])
def test_array_in_enum_field_returns_structured_cli_error(tmp_path, capsys, field):
    data = profile(["rgb_video"])
    if field == "goal":
        data["goal"] = []
    elif field == "status":
        data["inputs"]["rgb_video"]["status"] = []
    else:
        data["choice"] = {field: []}
    with pytest.raises(DataError):
        validate_profile(data)
    path = tmp_path / "invalid-profile.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert main(["strategy", "--profile", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["type"] == "DataError"
