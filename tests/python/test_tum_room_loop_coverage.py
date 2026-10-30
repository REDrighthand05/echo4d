"""Synthetic guards; actual room fusion is run by GitHub Actions."""
import tarfile

import pytest

from echo4d.model import DataError
from tum_room_loop_coverage import choose_views, safe_member, sparse_reference_indices, STRIDE


def test_fixed_room_viewpoints_are_diverse_and_train_only():
    train = tuple(i for i in range(180) if i % 7)
    recorded, free = choose_views(180, train)
    assert len(recorded) == 3 and len(free) == 2
    assert len(set(recorded + free)) == 5
    assert all(i in train for i in recorded + free)
    assert recorded[0] < 30 < recorded[1] < recorded[2]


def test_dense_source_and_exact_sparse_reference_indices():
    assert STRIDE == 4
    dense_train = tuple(i for i in range(340) if i % 7)
    sparse = sparse_reference_indices(dense_train)
    assert sparse == tuple(2 * i for i in range(170) if i % 7)
    assert len(sparse) == 145
    recorded, free = choose_views(340, sparse)
    assert all(i % 2 == 0 for i in recorded + free)


def test_refuse_insufficient_coverage():
    with pytest.raises(DataError, match="at least 110"):
        choose_views(60, tuple(range(50)))


@pytest.mark.parametrize("name", [
    "/etc/passwd", "rgbd_dataset_freiburg1_room/../escape",
    "rgbd_dataset_freiburg1_desk/depth/a.png",
])
def test_reject_untrusted_archive_paths(name):
    entry = tarfile.TarInfo(name)
    with pytest.raises(DataError, match="Unsafe"):
        safe_member(entry)


def test_reject_symlink_archive_members():
    entry = tarfile.TarInfo("rgbd_dataset_freiburg1_room/depth/evil.png")
    entry.type = tarfile.SYMTYPE
    with pytest.raises(DataError, match="Unsafe"):
        safe_member(entry)

def _translated(x):
    import numpy as np
    p = np.eye(4)
    p[0, 3] = x
    return p


def test_two_measured_farther_views_remove_unsupported_foreground():
    import numpy as np
    from echo4d.model import Camera
    from tum_room_loop_coverage import depth_conflict_mask, depth_local_min
    cam = Camera(21, 21, 12., 12., 10., 10.)
    source = np.ones((21, 21), np.float32)
    source[10, 10] = .8
    far = depth_local_min(np.full_like(source, 1.6))
    mask = depth_conflict_mask(source, _translated(0),
                               [(_translated(.1), far), (_translated(-.1), far)], cam)
    assert mask[10, 10]


def test_agreement_occlusion_unknown_and_single_conflict_are_preserved():
    import numpy as np
    from echo4d.model import Camera
    from tum_room_loop_coverage import depth_conflict_mask, depth_local_min
    cam = Camera(21, 21, 12., 12., 10., 10.)
    source = np.ones((21, 21), np.float32)
    agree = depth_local_min(source)
    far = depth_local_min(np.full_like(source, 1.6))
    closer = depth_local_min(np.full_like(source, .6))
    missing = depth_local_min(np.zeros_like(source))
    for other in (agree, closer, missing):
        assert not depth_conflict_mask(source, _translated(0),
            [(_translated(.1), other), (_translated(-.1), far)], cam).any()
    assert not depth_conflict_mask(source, _translated(0),
        [(_translated(.1), closer), (_translated(-.1), closer)], cam).any()


def test_crossview_neighbors_require_actual_baseline():
    from tum_room_loop_coverage import depth_neighbors
    found = depth_neighbors(0, [_translated(x) for x in (0., .01, .09, .14, .25, .30, .4)])
    assert 1 not in found and 2 in found and len(found) <= 6
