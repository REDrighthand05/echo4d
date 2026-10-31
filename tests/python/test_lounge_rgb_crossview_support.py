"""Independent RGB-depth agreement: geometry regression without withheld depths."""
import numpy as np
import pytest
from types import SimpleNamespace
from lounge_rgb_crossview_support import corroborate_rgb_depths

CAM = SimpleNamespace(width=20, height=16, fx=30., fy=30., cx=9.5, cy=7.5)


def pose(x=0.):
    p = np.eye(4)
    p[0, 3] = x
    return p


def test_two_separated_rgb_views_confirm_real_plane():
    # Frontoparallel world plane at z=2m, both predicted depths agree.
    a = np.full((16,20), 2., np.float32)
    masks, receipts = corroborate_rgb_depths({0:a, 1:a}, [pose(0), pose(.3)], CAM)
    assert masks[0][:, 8:14].any()
    assert masks[1][:, 5:11].any()
    assert receipts[0]["crossview_confirmed_pixels"] > 20


def test_false_sheet_rejected_when_other_view_disagrees():
    a = np.full((16,20), 2., np.float32)
    b = np.full((16,20), 1., np.float32)
    masks, _ = corroborate_rgb_depths({0:a, 1:b}, [pose(0), pose(.3)], CAM)
    assert not masks[0].any()
    assert not masks[1].any()


def test_same_center_cannot_fake_independent_view():
    a = np.full((16,20), 2., np.float32)
    masks, receipts = corroborate_rgb_depths({0:a, 1:a}, [pose(0), pose(0)], CAM)
    assert not masks[0].any()
    assert receipts[0]["eligible_independent_cameras"] == 0


def test_missing_depth_abstains_and_three_view_threshold():
    a = np.full((16,20), 2., np.float32)
    b = a.copy()
    b[:, :] = 0
    masks, _ = corroborate_rgb_depths(
        {0:a, 1:b, 2:a}, [pose(0),pose(.3),pose(-.3)], CAM, min_other_views=2)
    assert not masks[0].any()


def test_invalid_inputs_fail_fast():
    with pytest.raises(ValueError, match="dimensions"):
        corroborate_rgb_depths({0:np.ones((2,3))}, [pose(0)], CAM)
    with pytest.raises(ValueError, match="independent"):
        corroborate_rgb_depths({0:np.ones((16,20))}, [pose(0)], CAM,
                               min_other_views=0)
