"""Synthetic geometry oracle and failure counterexamples, NOT real TUM evidence."""
import numpy as np
import pytest
pytest.importorskip('cv2')
from echo4d.model import Camera, DataError
from echo4d.known_pose_plane_sweep import infer_plane_sweep


def sample():
    c = Camera(160, 120, 160., 160., 79.5, 59.5)
    ref = np.random.default_rng(42).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    sources = []
    for tx in (-.1, .1):
        pose = np.eye(4)
        pose[0, 3] = tx
        shift = round(c.fx * tx / 2.)
        src = np.zeros_like(ref)
        if shift > 0:
            src[:, :-shift] = ref[:, shift:]
        else:
            src[:, -shift:] = ref[:, :shift]
        sources.append((src, pose))
    return c, ref, sources


def test_two_source_metric_plane_oracle():
    c, ref, sources = sample()
    depth, stats = infer_plane_sweep(ref, sources, c, np.eye(4))
    assert stats['measured_depth_used'] is False
    assert stats['source_view_count'] == 2
    assert stats['valid_depth_pixels'] > 8000
    assert abs(float(np.median(depth[depth > 0])) - 2.) < .08


def test_textureless_scene_remains_unknown():
    c, ref, sources = sample()
    ref[:] = 127
    sources = [(np.full_like(ref, 127), pose) for _, pose in sources]
    depth, _ = infer_plane_sweep(ref, sources, c, np.eye(4), planes=16)
    assert not np.any(depth)


def test_invalid_sources_and_unsafe_ranges_rejected():
    c, ref, sources = sample()
    with pytest.raises(DataError):
        infer_plane_sweep(ref, sources[:1], c, np.eye(4))
    with pytest.raises(DataError):
        infer_plane_sweep(ref, sources, c, np.eye(4), near=0.1)
    with pytest.raises(DataError):
        infer_plane_sweep(ref, sources, c, np.eye(4), planes=8)
    with pytest.raises(DataError):
        infer_plane_sweep(ref, [(src[:80], pose) for src, pose in sources], c, np.eye(4))
    with pytest.raises(DataError):
        infer_plane_sweep(ref, [(src, np.eye(4)) for src, _ in sources], c, np.eye(4))

def _four_source_plane(xs, depths, *, random_last=False):
    camera = Camera(160, 120, 160., 160., 79.5, 59.5)
    ref = np.random.default_rng(42).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    sources = []
    for i, (tx, z) in enumerate(zip(xs, depths)):
        pose = np.eye(4)
        pose[0, 3] = tx
        shift = round(camera.fx * tx / z)
        src = np.zeros_like(ref)
        if shift > 0:
            src[:, :-shift] = ref[:, shift:]
        else:
            src[:, -shift:] = ref[:, :shift]
        if random_last and i == len(xs) - 1:
            src = np.random.default_rng(123).integers(0, 256, ref.shape, dtype=np.uint8)
        sources.append((src, pose))
    return camera, ref, sources


def test_diverse_pairs_preserve_metric_plane_oracle():
    c, ref, sources = _four_source_plane((-.15, -.1, .1, .15), (2., 2., 2., 2.))
    depth, stats = infer_plane_sweep(ref, sources, c, np.eye(4), pairing='diverse')
    assert stats['source_pairing'] == 'diverse'
    assert stats['used_diverse_pairs'] is True
    assert stats['diverse_pair_count'] == 4
    assert stats['valid_depth_pixels'] > 9000
    assert abs(float(np.median(depth[depth > 0])) - 2.) < .08


def test_same_side_sources_fall_back_to_prior_algorithm():
    c, ref, sources = _four_source_plane((-.1, -.15), (2., 2.))
    old, _ = infer_plane_sweep(ref, sources, c, np.eye(4))
    new, stats = infer_plane_sweep(ref, sources, c, np.eye(4), pairing='diverse')
    assert stats['used_diverse_pairs'] is False
    assert stats['diverse_pair_count'] == 0
    np.testing.assert_array_equal(new, old)


def test_two_correlated_same_side_false_matches_are_suppressed():
    # Two occluded same-side cameras support a convincing WRONG 1m plane.
    # Only one opposite-side camera observes the true 2m plane; the other
    # has unrelated RGB. The old top-two vote invents a coherent wrong depth.
    c, ref, sources = _four_source_plane(
        (-.15, -.1, .1, .15), (1., 1., 2., 2.), random_last=True)
    old, _ = infer_plane_sweep(ref, sources, c, np.eye(4), pairing='top2')
    new, stats = infer_plane_sweep(ref, sources, c, np.eye(4), pairing='diverse')
    center = np.s_[25:95, 35:125]
    old_false = int(np.count_nonzero((old[center] > .8) & (old[center] < 1.2)))
    new_false = int(np.count_nonzero((new[center] > .8) & (new[center] < 1.2)))
    assert old_false > 4000
    assert new_false < old_false / 2
    assert stats['measured_depth_used'] is False


def test_unknown_pairing_rejected():
    c, ref, sources = _four_source_plane((-.1, .1), (2., 2.))
    with pytest.raises(DataError, match='pairing'):
        infer_plane_sweep(ref, sources, c, np.eye(4), pairing='anything')
