"""Geometry-only source choice: synthetic tests, not TUM quality evidence."""
import numpy as np
import pytest
from echo4d.model import Camera, DataError
from echo4d.known_pose_mvs_sources import select_plane_sweep_views


def scene():
    camera = Camera(320, 240, 262.5, 262.5, 159.75, 119.75)
    xs = [0., -.07, .07, -.11, .11, -.16, .16, -.22, .22]
    frames = []
    for x in xs:
        pose = np.eye(4)
        pose[0, 3] = x
        frames.append({'pose': pose})
    return camera, frames


def test_bilateral_sources_and_train_only():
    camera, frames = scene()
    choices, stats = select_plane_sweep_views(frames, camera, list(range(9)),
                                               max_anchors=9, max_sources=4)
    anchor_zero = dict(choices)[0]
    assert len(anchor_zero) == 4
    assert any(frames[i]['pose'][0, 3] < 0 for i in anchor_zero)
    assert any(frames[i]['pose'][0, 3] > 0 for i in anchor_zero)
    assert stats['selected_training_anchors'] >= 1
    assert stats['train_only'] is True
    choices, _ = select_plane_sweep_views(frames, camera, [0, 1, 2, 3, 4],
                                          max_anchors=5)
    assert all(i in {0, 1, 2, 3, 4} and set(js) <= {0, 1, 2, 3, 4}
               for i, js in choices)


def test_degenerate_pose_yields_no_false_sources():
    camera, frames = scene()
    for f in frames:
        f['pose'] = np.eye(4)
    choices, stats = select_plane_sweep_views(frames, camera, list(range(9)))
    assert choices == []
    assert stats['eligible_training_anchors'] == 0


def test_bad_budget_and_duplicate_training_rejected():
    camera, frames = scene()
    for indices, kwargs in [([0, 0, 1], {}), ([0, 10], {}),
                            ([0, 1, 2], {'max_sources': 1}),
                            ([0, 1, 2], {'max_anchors': 0})]:
        with pytest.raises(DataError):
            select_plane_sweep_views(frames, camera, indices, **kwargs)
