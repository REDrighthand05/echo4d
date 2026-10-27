"""Known-answer M1 graph contracts, never a claim of real Mesh quality."""
import pytest
from echo4d.composable import ArtifactType, ModuleSpec, compose


def a(kind, frame="none", scale="none", clock="none", evidence="measured", partition="train",
      *, scene_id="fixture", coordinate_id=None, time_id=None,
      length_unit=None, time_unit=None, handedness=None, pose_direction=None):
    return ArtifactType(
        kind, frame, scale, clock, evidence, partition, scene_id,
        ("fixture_space" if frame != "none" else "none") if coordinate_id is None else coordinate_id,
        ("fixture_clock" if clock != "none" else "none") if time_id is None else time_id,
        ("m" if scale == "metric" else "none") if length_unit is None else length_unit,
        ("s" if clock != "none" else "none") if time_unit is None else time_unit,
        ("right" if frame != "none" else "none") if handedness is None else handedness,
        ("camera_to_world" if kind == "poses" else "none") if pose_direction is None else pose_direction,
    )


VIDEO = a("rgb_video", clock="frame_pts")
FRAMES = a("rgb_frames", frame="camera", clock="frame_pts")
POSE_REL = a("poses", frame="world", scale="relative", clock="frame_pts", evidence="estimated")
POSE_MET = a("poses", frame="world", scale="metric", clock="frame_pts", evidence="estimated")
ANCHOR = a("metric_anchor", scale="metric")
DEPTH_MET = a("depth_z", frame="camera", scale="metric", clock="frame_pts")
DEPTH_REL = a("depth_z", frame="camera", scale="relative", clock="frame_pts", evidence="estimated")
MESH_MET = a("mesh", frame="world", scale="metric", evidence="estimated")
HOLDOUT = a("depth_z", frame="camera", scale="metric", clock="frame_pts", partition="holdout")


def mod(name, ins, outs, cost=1, installed=True, revision="a" * 40):
    return ModuleSpec(name, tuple(ins), tuple(outs), cost, installed, revision)


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
        ArtifactType("depth_z", scene_id="fixture", coordinate_id="fixture_space",
                     time_id="none", length_unit="none", time_unit="none",
                     handedness="right", pose_direction="none", **kwargs)


def test_original_asset_and_generated_mesh_cannot_become_measured_mesh():
    raw = a("mesh", frame="world", scale="metric", evidence="generated")
    assert not compose((raw,), (), MESH_MET).routes


def test_explicit_adapter_can_bridge_different_clock_but_cannot_fake_execution():
    synced = a("depth_z", frame="camera", scale="metric", clock="sensor_synced")
    align = mod("timestamp_alignment", [synced], [DEPTH_MET])
    out = compose((POSE_MET, synced), (align, mod("tsdf", [POSE_MET, DEPTH_MET], [MESH_MET])), MESH_MET)
    assert [r.modules for r in out.routes] == [("timestamp_alignment", "tsdf")]
    assert out.routes[0].status == "not_executed"


def test_same_kind_from_another_scene_cannot_enter_fusion():
    other = a("depth_z", frame="camera", scale="metric", clock="frame_pts", scene_id="other_scene")
    assert not compose((POSE_MET, other), (mod("tsdf", [POSE_MET, DEPTH_MET], [MESH_MET]),), MESH_MET).routes


def test_same_clock_type_different_domain_requires_adapter():
    other = a("depth_z", frame="camera", scale="metric", clock="frame_pts", time_id="other_clock")
    assert not compose((POSE_MET, other), (mod("tsdf", [POSE_MET, DEPTH_MET], [MESH_MET]),), MESH_MET).routes


def test_same_world_label_different_coordinate_domain_requires_adapter():
    other = a("poses", frame="world", scale="metric", clock="frame_pts",
              evidence="estimated", coordinate_id="other_world")
    assert not compose((other, DEPTH_MET), (mod("tsdf", [POSE_MET, DEPTH_MET], [MESH_MET]),), MESH_MET).routes


def test_module_revision_changes_route_fingerprint_without_changing_ports():
    before = compose((VIDEO, ANCHOR, DEPTH_MET), catalog(), MESH_MET)
    changed = tuple(mod(m.module_id, m.inputs, m.outputs, m.estimated_units, m.installed,
                        revision="b" * 40 if m.module_id == "sfm_a" else m.revision)
                    for m in catalog())
    after = compose((VIDEO, ANCHOR, DEPTH_MET), changed, MESH_MET)
    assert [r.modules for r in before.routes] == [r.modules for r in after.routes]
    assert before.routes[0].path_id != after.routes[0].path_id
    assert before.routes[1].path_id == after.routes[1].path_id


def test_unassigned_data_must_be_split_before_build():
    raw = a("rgb_video", clock="frame_pts", partition="unassigned")
    with pytest.raises(ValueError, match="unassigned"):
        mod("unsafe_decode", [raw], [FRAMES])
    with pytest.raises(ValueError, match="unassigned"):
        mod("unsafe_output", [VIDEO], [a("mesh", partition="unassigned")])


@pytest.mark.parametrize("bad", ["main", "", "g" * 40, "0" * 39])
def test_unpinned_revision_rejected(bad):
    with pytest.raises(ValueError, match="revision"):
        mod("unversioned", [VIDEO], [FRAMES], revision=bad)


@pytest.mark.parametrize("context", [
    {"scene_id": "none"},
    {"coordinate_id": "none"},
    {"time_id": "none"},
])
def test_missing_or_unknown_domain_identity_is_rejected(context):
    with pytest.raises(ValueError):
        a("depth_z", frame="camera", scale="metric", clock="frame_pts", **context)


def test_centimetre_depth_cannot_silently_feed_metre_fusion():
    centimetres = a("depth_z", frame="camera", scale="metric", clock="frame_pts",
                    length_unit="cm")
    tsdf = mod("tsdf", [POSE_MET, DEPTH_MET], [MESH_MET])
    assert not compose((POSE_MET, centimetres), (tsdf,), MESH_MET).routes
    convert = mod("centimetres_to_metres", [centimetres], [DEPTH_MET])
    assert [r.modules for r in compose((POSE_MET, centimetres), (convert, tsdf), MESH_MET).routes] == [
        ("centimetres_to_metres", "tsdf")
    ]


def test_millisecond_pts_cannot_silently_feed_second_based_fusion():
    milliseconds = a("depth_z", frame="camera", scale="metric", clock="frame_pts",
                     time_unit="ms")
    tsdf = mod("tsdf", [POSE_MET, DEPTH_MET], [MESH_MET])
    assert not compose((POSE_MET, milliseconds), (tsdf,), MESH_MET).routes
    align = mod("pts_ms_to_seconds", [milliseconds], [DEPTH_MET])
    assert [r.modules for r in compose((POSE_MET, milliseconds), (align, tsdf), MESH_MET).routes] == [
        ("pts_ms_to_seconds", "tsdf")
    ]


def test_left_handed_depth_cannot_silently_feed_right_handed_fusion():
    left = a("depth_z", frame="camera", scale="metric", clock="frame_pts",
             handedness="left")
    assert not compose((POSE_MET, left),
                       (mod("tsdf", [POSE_MET, DEPTH_MET], [MESH_MET]),), MESH_MET).routes


def test_world_to_camera_poses_require_explicit_inversion():
    inverse = a("poses", frame="world", scale="metric", clock="frame_pts",
                evidence="estimated", pose_direction="world_to_camera")
    tsdf = mod("tsdf", [POSE_MET, DEPTH_MET], [MESH_MET])
    assert not compose((inverse, DEPTH_MET), (tsdf,), MESH_MET).routes
    invert = mod("invert_pose", [inverse], [POSE_MET])
    assert [r.modules for r in compose((inverse, DEPTH_MET), (invert, tsdf), MESH_MET).routes] == [
        ("invert_pose", "tsdf")
    ]


@pytest.mark.parametrize("field,value", [
    ("length_unit", "none"), ("length_unit", "feet"),
    ("time_unit", "none"), ("time_unit", "minutes"),
    ("handedness", "none"), ("handedness", "up"),
    ("pose_direction", "none"), ("pose_direction", "world"),
])
def test_spatial_pose_metadata_must_be_explicit_and_supported(field, value):
    kwargs = {field: value}
    with pytest.raises(ValueError):
        a("poses", frame="world", scale="metric", clock="frame_pts",
          evidence="estimated", **kwargs)


def test_nonmetric_and_nonspatial_ports_cannot_claim_units_or_orientation():
    with pytest.raises(ValueError, match="length unit"):
        a("rgb_video", clock="frame_pts", length_unit="m")
    with pytest.raises(ValueError, match="handedness"):
        a("rgb_video", clock="frame_pts", handedness="right")
    with pytest.raises(ValueError, match="transform direction"):
        a("rgb_video", clock="frame_pts", pose_direction="camera_to_world")


def test_route_id_changes_when_port_convention_changes():
    left = a("depth_z", frame="camera", scale="metric", clock="frame_pts",
             handedness="left")
    right_route = compose((POSE_MET, DEPTH_MET),
                          (mod("tsdf", [POSE_MET, DEPTH_MET], [MESH_MET]),), MESH_MET)
    left_route = compose((POSE_MET, left),
                         (mod("convert", [left], [DEPTH_MET]),
                          mod("tsdf", [POSE_MET, DEPTH_MET], [MESH_MET])), MESH_MET)
    assert len(right_route.routes) == len(left_route.routes) == 1
    assert right_route.routes[0].path_id != left_route.routes[0].path_id


def test_independent_dag_topological_orders_count_once_not_twice():
    seed, left, right = a("seed"), a("left"), a("right")
    modules = (
        mod("left_stage", [seed], [left]),
        mod("right_stage", [seed], [right]),
        mod("join", [left, right], [MESH_MET]),
    )
    result = compose((seed,), modules, MESH_MET)
    assert len(result.routes) == 1
    assert result.routes[0].modules == ("left_stage", "right_stage", "join")
    assert result.routes[0].estimated_units == 3
    assert not result.truncated


def test_real_alternate_provider_not_discarded_by_dag_dedup():
    seed, left, right = a("seed"), a("left"), a("right")
    modules = (
        mod("left_a", [seed], [left]),
        mod("left_b", [seed], [left]),
        mod("right_stage", [seed], [right]),
        mod("join", [left, right], [MESH_MET]),
    )
    result = compose((seed,), modules, MESH_MET, max_routes=3)
    assert len(result.routes) == 2
    assert {r.modules for r in result.routes} == {
        ("left_a", "right_stage", "join"),
        ("left_b", "right_stage", "join"),
    }
    assert len({r.path_id for r in result.routes}) == 2


def test_unrelated_side_branch_is_not_a_distinct_mesh_route():
    seed, unused = a("seed"), a("unused")
    modules = (
        mod("direct", [seed], [MESH_MET]),
        mod("sidecar", [seed], [unused]),
    )
    result = compose((seed,), modules, MESH_MET)
    assert len(result.routes) == 1
    assert result.routes[0].modules == ("direct",)
    assert result.routes[0].estimated_units == 1


def test_canonical_graph_identity_is_independent_of_catalog_order():
    seed, left, right = a("seed"), a("left"), a("right")
    modules = (
        mod("left_stage", [seed], [left]),
        mod("right_stage", [seed], [right]),
        mod("join", [left, right], [MESH_MET]),
    )
    forward = compose((seed,), modules, MESH_MET)
    backward = compose((seed,), tuple(reversed(modules)), MESH_MET)
    assert forward.routes == backward.routes


def test_max_states_bounds_unique_enqueued_states_and_marks_truncation():
    seed, left, right = a("seed"), a("left"), a("right")
    modules = (
        mod("left_stage", [seed], [left]),
        mod("right_stage", [seed], [right]),
        mod("join", [left, right], [MESH_MET]),
    )
    limited = compose((seed,), modules, MESH_MET, max_states=2)
    assert limited.visited_states <= 2
    assert limited.truncated
