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
