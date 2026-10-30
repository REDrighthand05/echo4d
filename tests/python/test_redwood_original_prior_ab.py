import numpy as np
import pytest

from redwood_original_prior_ab import transfer_colors_chunked


class Mesh:
    def __init__(self, points, normals, colors=None):
        self.vertices = np.asarray(points, dtype=float)
        self.vertex_normals = np.asarray(normals, dtype=float)
        self.vertex_colors = np.asarray(colors if colors is not None else [], dtype=float)


def test_original_prior_colors_are_chunked_and_only_measured_when_supported():
    measured = Mesh([[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                    [[0, 0, 1]] * 3,
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    prior = Mesh([[0.01, 0, 0], [1.02, 0, 0], [10, 0, 0]],
                 [[0, 0, 1]] * 3)
    rgb, supported = transfer_colors_chunked(prior, measured, chunk=1)
    assert supported.tolist() == [True, True, False]
    assert np.allclose(rgb[0], [1, 0, 0])
    assert np.allclose(rgb[1], [0, 1, 0])
    assert np.allclose(rgb[2], [.53, .56, .59])


def test_wrong_facing_surface_does_not_get_measured_color():
    measured = Mesh([[0, 0, 0]], [[0, 0, 1]], [[1, 0, 0]])
    prior = Mesh([[0, 0, 0]], [[0, 0, -1]])
    rgb, supported = transfer_colors_chunked(prior, measured)
    assert not supported[0]
    assert np.allclose(rgb[0], [.53, .56, .59])


def test_missing_normals_are_rejected():
    measured = Mesh([[0, 0, 0]], [[0, 0, 1]], [[1, 0, 0]])
    prior = Mesh([[0, 0, 0]], [])
    with pytest.raises(ValueError):
        transfer_colors_chunked(prior, measured)


def test_five_camera_specs_are_fixed_and_original_camera_poses_preserved():
    from redwood_original_prior_ab import fixed_view_specs

    class Data:
        color_paths = [f"frame_{i:05d}.png" for i in range(600)]

    poses = [np.eye(4) for _ in Data.color_paths]
    specs = fixed_view_specs(Data(), poses)
    assert len(specs) == 5
    assert [s["kind"] for s in specs] == [
        "recorded", "recorded", "recorded", "free", "free"]
    assert len({s["file"] for s in specs}) == 5
    for spec in specs[:3]:
        assert np.array_equal(spec["pose"], poses[spec["frame"]])
    for spec in specs:
        assert spec["pose"].shape == (4, 4)
        assert np.isfinite(spec["pose"]).all()
