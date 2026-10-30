"""Synthetic component sanity checks; visual acceptance requires real Desk views."""
import numpy as np
import pytest
import trimesh

from tum_desk_component_prune import remove_small_islands


def test_small_island_removal_preserves_measured_rgb():
    large = trimesh.creation.icosphere(subdivisions=2, radius=.4)
    small = trimesh.creation.icosphere(subdivisions=0, radius=.1)
    small.apply_translation([2., 0., 0.])
    large.visual.vertex_colors = [20, 80, 180, 255]
    small.visual.vertex_colors = [250, 20, 20, 255]
    mixed = trimesh.util.concatenate((large, small))
    cleaned, stats = remove_small_islands(mixed, min_faces=100)
    assert len(cleaned.faces) == len(large.faces)
    assert stats["removed_triangles"] == len(small.faces)
    assert stats["input_components"] == 2
    assert stats["kept_components"] == 1
    assert np.all(cleaned.visual.vertex_colors[:, :3] == [20, 80, 180])
    np.testing.assert_allclose(cleaned.bounds, large.bounds)


def test_refuses_empty_output_and_invalid_threshold():
    small = trimesh.creation.icosphere(subdivisions=0)
    small.visual.vertex_colors = [1, 2, 3, 255]
    with pytest.raises(ValueError, match="No component"):
        remove_small_islands(small, min_faces=100)
    with pytest.raises(ValueError, match="min_faces"):
        remove_small_islands(small, min_faces=0)
