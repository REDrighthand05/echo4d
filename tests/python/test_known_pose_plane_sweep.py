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


def test_source_view_occlusion_uses_inferred_z_and_preserves_unknown():
    c, ref, sources = sample()
    old, _ = infer_plane_sweep(ref, sources, c, np.eye(4))
    shape = (c.height, c.width)
    unknown = [np.zeros(shape, dtype=np.float32) for _ in sources]
    same, stats = infer_plane_sweep(ref, sources, c, np.eye(4),
                                    source_depths=unknown)
    np.testing.assert_array_equal(same, old)
    assert stats['source_visibility'] is True
    assert stats['occluded_photo_samples'] == 0

    # First source has a stable nearer surface; the hypothesized 2m plane
    # cannot receive its vote. The other source alone cannot form two-view Z.
    foreground = [np.full(shape, 1., dtype=np.float32), unknown[1]]
    filtered, stats = infer_plane_sweep(ref, sources, c, np.eye(4),
                                        source_depths=foreground)
    assert stats['occluded_photo_samples'] > 0
    assert np.count_nonzero(filtered > 1.6) < np.count_nonzero(old > 1.6) / 4

    consistent = [np.full(shape, 2., dtype=np.float32) for _ in sources]
    retained, _ = infer_plane_sweep(ref, sources, c, np.eye(4),
                                    source_depths=consistent)
    assert np.count_nonzero(retained) >= np.count_nonzero(old) * .75


def test_source_view_depth_shape_and_sign_rejected():
    c, ref, sources = sample()
    shape = (c.height, c.width)
    valid = np.zeros(shape, dtype=np.float32)
    with pytest.raises(DataError, match='Source visibility'):
        infer_plane_sweep(ref, sources, c, np.eye(4), source_depths=[valid])
    with pytest.raises(DataError, match='Source visibility'):
        infer_plane_sweep(ref, sources, c, np.eye(4),
                          source_depths=[valid, -np.ones(shape, dtype=np.float32)])


def test_soft_source_visibility_preserves_unknown_and_limits_penalty():
    c, ref, sources = sample()
    shape = (c.height, c.width)
    unknown = [np.zeros(shape, dtype=np.float32) for _ in sources]
    old, _ = infer_plane_sweep(ref, sources, c, np.eye(4))
    neutral, stats = infer_plane_sweep(
        ref, sources, c, np.eye(4), source_depths=unknown,
        visibility_mode='soft')
    np.testing.assert_array_equal(neutral, old)
    assert stats['penalized_photo_samples'] == 0
    assert stats['visibility_mode'] == 'soft'

    foreground = [np.full(shape, 1., dtype=np.float32), unknown[1]]
    hard, _ = infer_plane_sweep(
        ref, sources, c, np.eye(4), source_depths=foreground)
    soft, stats = infer_plane_sweep(
        ref, sources, c, np.eye(4), source_depths=foreground,
        visibility_mode='soft')
    assert stats['penalized_photo_samples'] > 0
    assert stats['penalized_photo_samples'] == stats['occluded_photo_samples']
    # A soft vote cannot be equivalent to discarding all foreground votes.
    assert np.count_nonzero(soft > 1.6) > np.count_nonzero(hard > 1.6)

    with pytest.raises(DataError, match='visibility mode'):
        infer_plane_sweep(ref, sources, c, np.eye(4),
                          source_depths=unknown, visibility_mode='unsafe')
