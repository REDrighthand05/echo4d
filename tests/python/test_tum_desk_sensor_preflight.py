"""Index-only Freiburg1 desk pairing: frozen source roles, no image decoding."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from echo4d.model import DataError
from tum_desk_sensor_preflight import audit_indexes


def make_indexes(root: Path, *, first_gap=False, duplicate=False):
    rgb = root / "rgb.txt"
    depth = root / "depth.txt"
    rgb_rows = ["# original RGB times"] + [
        f"{i * .1:.6f} rgb/{i:06d}.png" for i in range(600)]
    depth_rows = ["# original measured sensor Z times"]
    for i in range(600):
        time = i * .1 + (.041 if first_gap and i == 0 else .005)
        path = "depth/000016.png" if duplicate and i == 8 else f"depth/{i:06d}.png"
        depth_rows.append(f"{time:.6f} {path}")
    rgb.write_text("\n".join(rgb_rows) + "\n")
    depth.write_text("\n".join(depth_rows) + "\n")
    return rgb, depth


def test_exact_roles_and_original_times(tmp_path):
    rgb, depth = make_indexes(tmp_path)
    result = audit_indexes(rgb, depth)
    assert result["status"] == "PASS_FROZEN_V1_INDEX_PAIRING"
    assert result["required_role_count"] == 68
    assert result["required_geometry_train_count"] == 48
    assert result["required_geometry_holdout_count"] == 12
    assert result["required_anchor_count"] == 3
    assert result["required_diagnostic_count"] == 7
    assert result["accepted_role_count"] == 68
    assert result["uses_depth_image_pixels"] is False
    assert result["rows"][0]["absolute_delta_ms"] == pytest.approx(5.0)
    assert "geometry_holdout" in result["rows"][4]["roles"]
    assert set(result["rows"][15]["roles"]) == {"geometry_train", "scale_anchor"}


def test_first_source_frame_has_unmet_gap_not_replaced(tmp_path):
    rgb, depth = make_indexes(tmp_path, first_gap=True)
    result = audit_indexes(rgb, depth)
    assert result["status"] == "BLOCKED_FROZEN_V1_INDEX_PAIRING"
    assert result["rows"][0]["frame"] == "frame_00000.png"
    assert result["rows"][0]["accepted_for_v1"] is False
    assert "NO_DEPTH_WITHIN_FROZEN_30MS" in result["rows"][0]["reasons"]
    assert result["rows"][0]["absolute_delta_ms"] == pytest.approx(41.0)
    assert result["rows"][1]["frame"] == "frame_00001.png"


def test_duplicate_depth_source_even_if_two_times(tmp_path):
    rgb, depth = make_indexes(tmp_path, duplicate=True)
    result = audit_indexes(rgb, depth)
    assert result["status"] == "BLOCKED_FROZEN_V1_INDEX_PAIRING"
    assert any("DUPLICATE_SELECTED_DEPTH_SOURCE" in x["reasons"]
               for x in result["rows"])


def test_missing_real_depth_member_does_not_hide_missingness(tmp_path):
    rgb, depth = make_indexes(tmp_path)
    members = {f"rgb/{i:06d}.png" for i in range(600)}
    members |= {f"depth/{i:06d}.png" for i in range(600)}
    members.remove("depth/000000.png")
    result = audit_indexes(rgb, depth, members)
    assert result["rows"][0]["reasons"] == ["DEPTH_INDEX_PATH_NOT_IN_ARCHIVE"]


def test_missing_rgb_member_rejected(tmp_path):
    rgb, depth = make_indexes(tmp_path)
    with pytest.raises(DataError, match="Indexed RGB image absent"):
        audit_indexes(rgb, depth, members=set())


def test_unsafe_depth_index_path_rejected(tmp_path):
    rgb, depth = make_indexes(tmp_path)
    content = depth.read_text().replace("depth/000000.png", "../evil.png")
    depth.write_text(content)
    with pytest.raises(DataError, match="Unsafe original depth"):
        audit_indexes(rgb, depth)


def test_missing_75th_original_rgb_rejected(tmp_path):
    rgb, depth = make_indexes(tmp_path)
    rgb.write_text("\n".join(rgb.read_text().splitlines()[:250]) + "\n")
    with pytest.raises(DataError, match="Not enough original RGB"):
        audit_indexes(rgb, depth)
