"""Known-answer M1 graph contracts, never a claim of real Mesh quality."""
import pytest
from echo4d.composable import ArtifactType, ModuleSpec, compose


def a(kind, frame="none", scale="none", clock="none", evidence="measured", partition="train"):
    return ArtifactType(kind, frame, scale, clock, evidence, partition)


VIDEO = a("rgb_video", clock="frame_pts")
FRAMES = a("rgb_frames", frame="camera", clock="frame_pts")
POSE_REL = a("poses", frame="world", scale="relative", clock="frame_pts", evidence="estimated")
POSE_MET = a("poses", frame="world", scale="metric", clock="frame_pts", evidence="estimated")
ANCHOR = a("metric_anchor", scale="metric")
DEPTH_MET = a("depth_z", frame="camera", scale="metric", clock="frame_pts")
DEPTH_REL = a("depth_z", frame="camera", scale="relative", clock="frame_pts", evidence="estimated")
MESH_MET = a("mesh", frame="world", scale="metric", evidence="estimated")
HOLDOUT = a("depth_z", frame="camera", scale="metric", clock="frame_pts", partition="holdout")


def mod(name, ins, outs, cost=1, installed=True):
    return ModuleSpec(name, tuple(ins), tuple(outs), cost, installed)


def catalog():
    return (
        mod("decode", [VIDEO], [FRAMES]),
        mod("sfm_a", [FRAMES], [POSE_REL], 2),
        mod("sfm_b", [FRAMES], [POSE_REL], 4),
        mod("scale_from_sensor", [POSE_REL, ANCHOR], [POSE_MET]),
        mod("tsdf", [POSE_MET, DEPTH_MET], [MESH_MET], 3),
    )


def test_two_distinct_swap_in_algorithms_reach_same_metric_mesh():
    result = compose((VIDEO, ANCHOR, DEPTH_MET), catalog(), MESH_MET)
    assert [r.modules for r in result.routes] == [
        ("decode", "sfm_a", "scale_from_sensor", "tsdf"),
        ("decode", "sfm_b", "scale_from_sensor", "tsdf"),
    ]
    assert result.not_executed and all(r.status == "not_executed" for r in result.routes)
    assert all(len(r.path_id) == 64 for r in result.routes)


def test_reproducible_plans_independent_of_catalog_order():
    first = compose((VIDEO, ANCHOR, DEPTH_MET), catalog(), MESH_MET)
    second = compose((DEPTH_MET, VIDEO, ANCHOR), tuple(reversed(catalog())), MESH_MET)
    assert [r.path_id for r in first.routes] == [r.path_id for r in second.routes]


def test_relative_depth_cannot_impersonate_measured_metric_z():
    assert not compose((VIDEO, ANCHOR, DEPTH_REL), catalog(), MESH_MET).routes


def test_unsynchronized_sensor_clocks_cannot_implicitly_match():
    different_clock = a("depth_z", frame="camera", scale="metric", clock="sensor_synced")
    assert not compose((VIDEO, ANCHOR, different_clock), catalog(), MESH_MET).routes


def test_leftover_world_or_local_pose_cannot_implicitly_convert():
    local_pose = a("poses", frame="local", scale="metric", clock="frame_pts", evidence="estimated")
    assert not compose((local_pose, DEPTH_MET), (mod("tsdf", [POSE_MET, DEPTH_MET], [MESH_MET]),), MESH_MET).routes


def test_not_installed_is_not_executable_even_with_matching_ports():
    assert not compose((POSE_MET, DEPTH_MET), (mod("future", [POSE_MET, DEPTH_MET], [MESH_MET], installed=False),), MESH_MET).routes


def test_holdout_never_enters_building_graph():
    with pytest.raises(ValueError, match="heldout"):
        mod("leak", [HOLDOUT], [MESH_MET])
    assert not compose((POSE_MET, HOLDOUT), (mod("tsdf", [POSE_MET, DEPTH_MET], [MESH_MET]),), MESH_MET).routes
    with pytest.raises(ValueError, match="holdout"):
        compose((HOLDOUT,), (), HOLDOUT)


def test_resources_and_depth_cap_bound_search():
    assert not compose((VIDEO, ANCHOR, DEPTH_MET), catalog(), MESH_MET, max_steps=3).routes
    assert len(compose((VIDEO, ANCHOR, DEPTH_MET), catalog(), MESH_MET, budget_units=7).routes) == 1
    assert compose((VIDEO, ANCHOR, DEPTH_MET), catalog(), MESH_MET, max_states=2).truncated


def test_frozen_modules_cannot_cycle_or_invent_new_artifacts():
    x = a("x")
    y = a("y")
    fake = a("unreachable")
    cyclic = (mod("x_to_y", [x], [y]), mod("y_to_x", [y], [x]))
    assert not compose((x,), cyclic, fake, max_steps=200, max_states=200).routes
    assert compose((x,), cyclic, fake).visited_states == 2


def test_duplicate_ids_invalid_even_when_installed_false():
    with pytest.raises(ValueError, match="duplicate module"):
        compose((VIDEO,), (mod("same", [VIDEO], [FRAMES]), mod("same", [VIDEO], [FRAMES], installed=False)), MESH_MET)


@pytest.mark.parametrize("bad", [None, [], 0, True, -1])
def test_limits_reject_non_positive_or_wrong_types(bad):
    with pytest.raises(ValueError):
        compose((VIDEO,), catalog(), MESH_MET, max_routes=bad)


@pytest.mark.parametrize("kwargs", [
    {"frame": "camera", "scale": "none", "clock": "none", "evidence": "unknown", "partition": "train"},
    {"frame": "camera", "scale": "metric", "clock": [], "evidence": "measured", "partition": "train"},
    {"frame": "camera", "scale": "metric", "clock": "none", "evidence": "measured", "partition": "invalid"},
])
def test_untrusted_port_type_metadata_rejected(kwargs):
    with pytest.raises(ValueError):
        ArtifactType("depth_z", **kwargs)


def test_original_asset_and_generated_mesh_cannot_become_measured_mesh():
    raw = a("mesh", frame="world", scale="metric", evidence="generated")
    assert not compose((raw,), (), MESH_MET).routes


def test_explicit_adapter_can_bridge_different_clock_but_cannot_fake_execution():
    synced = a("depth_z", frame="camera", scale="metric", clock="sensor_synced")
    align = mod("timestamp_alignment", [synced], [DEPTH_MET])
    out = compose((POSE_MET, synced), (align, mod("tsdf", [POSE_MET, DEPTH_MET], [MESH_MET])), MESH_MET)
    assert [r.modules for r in out.routes] == [("timestamp_alignment", "tsdf")]
    assert out.routes[0].status == "not_executed"
