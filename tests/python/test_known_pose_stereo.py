"""Synthetic stereo oracle and counterexamples; not real-scene geometry evidence."""
import numpy as np
import pytest
cv2 = pytest.importorskip('cv2')
from echo4d.model import Camera, DataError
from echo4d.known_pose_stereo import infer_stereo, rectify_geometry


def setup():
    c=Camera(320,240,240.,240.,159.5,119.5)
    a=np.eye(4);b=np.eye(4);b[0,3]=.1
    return c,a,b


def test_plane_known_metric_depth_and_pose():
    camera,a,b=setup()
    rng=np.random.default_rng(31)
    left=rng.integers(0,256,(240,320,3),dtype=np.uint8)
    right=np.zeros_like(left);right[:,:-12]=left[:,12:]
    rgb,depth,new_camera,pose,stats=infer_stereo(left,right,camera,a,b)
    assert stats['measured_depth_used'] is False
    assert stats['valid_depth_pixels']>20000
    assert abs(np.median(depth[depth>0])-2.0)<.07
    assert rgb.shape==left.shape and new_camera.width==320
    assert np.allclose(pose[:3,3],a[:3,3])


def test_textureless_rgb_does_not_invent_dense_depth():
    camera,a,b=setup()
    img=np.full((240,320,3),128,np.uint8)
    _,depth,_,_,stats=infer_stereo(img,img,camera,a,b)
    assert stats['valid_depth_pixels']<100
    assert np.count_nonzero(depth)<100


def test_reverse_baseline_rejected_not_negative_depth():
    camera,a,b=setup()
    with pytest.raises(DataError):
        rectify_geometry(camera,b,a)


def test_wrong_resolution_and_invalid_disparity_rejected():
    camera,a,b=setup()
    rgb=np.zeros((240,320,3),np.uint8)
    with pytest.raises(DataError):
        infer_stereo(rgb[:200],rgb,camera,a,b)
    with pytest.raises(DataError):
        infer_stereo(rgb,rgb,camera,a,b,num_disparities=63)


def test_nearly_axial_pair_rejected_before_selection():
    camera=Camera(320,240,262.5,262.5,159.75,119.75)
    # Deterministic synthetic rigid pose: baseline and angle pass the original
    # gates, but OpenCV 4.12 alpha=0 returns fx=-8.3552855 for this pair.
    right=np.array([
        [.9807764929526009,.1643240072202502,-.1052382607357041,-.009400549307721306],
        [-.17928489248086166,.9717427437529457,-.153534905775928,-.007716478028937172],
        [.07703504526981696,.16945105669359628,.9825232522366588,.0716123446644652],
        [0,0,0,1]])
    with pytest.raises(DataError,match='Focal lengths'):
        rectify_geometry(camera,np.eye(4),right)


def test_nonfinite_rectification_rejected(monkeypatch):
    camera,a,b=setup()
    original=cv2.stereoRectify
    def broken(*args,**kwargs):
        result=list(original(*args,**kwargs))
        result[4]=result[4].copy();result[4][0,0]=np.nan
        return tuple(result)
    monkeypatch.setattr(cv2,'stereoRectify',broken)
    with pytest.raises(DataError,match='nonfinite'):
        rectify_geometry(camera,a,b)


def test_train_only_pair_backups_are_bounded_and_deterministic():
    # Train-only synthetic motion. Each selected anchor has alternative partners;
    # no withheld frame may enter either side of a stereo pair.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
    from tum_rgb_stereo import select_pairs, HOLDOUT, TOTAL
    camera, _, _ = setup()
    frames = []
    for i in range(TOTAL):
        pose = np.eye(4)
        pose[0, 3] = i * .01
        frames.append({'pose': pose})
    pairs = select_pairs(frames, camera, max_pairs=12, backups_per_left=3)
    assert pairs == select_pairs(frames, camera, max_pairs=12, backups_per_left=3)
    assert len(pairs) <= 36
    assert len({left for left, _ in pairs}) == 12
    assert all(left not in HOLDOUT and right not in HOLDOUT for left, right in pairs)
    assert all(sum(a == left for a, _ in pairs) <= 3 for left, _ in pairs)
    assert any(sum(a == left for a, _ in pairs) > 1 for left, _ in pairs)
    with pytest.raises(DataError, match='budget'):
        select_pairs(frames, camera, max_pairs=0)
    with pytest.raises(DataError, match='budget'):
        select_pairs(frames, camera, backups_per_left=5)


def test_native_640_rgb_preserves_original_pixels_and_intrinsics(tmp_path):
    from PIL import Image
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
    from tum_rgb_stereo import stereo_resolution, load_stereo_rgb
    yy, xx = np.indices((480, 640))
    pixels = np.stack((xx % 256, yy % 256, (xx + 2 * yy) % 256), axis=-1).astype(np.uint8)
    path = tmp_path / 'original.png'
    Image.fromarray(pixels).save(path)
    full, stride, disparities = stereo_resolution(640)
    assert (full.width, full.height, full.fx, full.cx, full.cy) == (640, 480, 525., 319.5, 239.5)
    assert stride == 1 and disparities == 128
    assert np.array_equal(load_stereo_rgb(path, 640), pixels)
    small, stride, disparities = stereo_resolution(320)
    assert (small.width, small.height, small.fx, small.cx, small.cy) == (320, 240, 262.5, 159.75, 119.75)
    assert stride == 2 and disparities == 64
    assert np.array_equal(load_stereo_rgb(path, 320), pixels[::2, ::2])
    for width in (0, 321, 800, True):
        with pytest.raises(DataError, match='width'):
            stereo_resolution(width)


def test_native_rgb_rejects_wrong_original_resolution(tmp_path):
    from PIL import Image
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
    from tum_rgb_stereo import load_stereo_rgb
    bad = tmp_path / 'wrong.png'
    Image.fromarray(np.zeros((240, 320, 3), np.uint8)).save(bad)
    with pytest.raises(DataError, match='640x480'):
        load_stereo_rgb(bad, 640)


def test_native_depth_consolidation_rejects_conflicting_surfaces():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
    from tum_rgb_stereo import reduce_native_depth_for_fusion
    native = np.zeros((480, 640), np.float32)
    native[0:2, 0:2] = [[2.0, 2.01], [0, 0]]
    native[0:2, 2:4] = [[1.0, 3.0], [0, 0]]
    native[2:4, 0:2] = [[0, 0], [0, 1.2]]
    result = reduce_native_depth_for_fusion(native)
    assert result.shape == (240, 320)
    assert result[0, 0] == pytest.approx(2.0)
    assert result[0, 1] == 0
    assert result[1, 0] == pytest.approx(1.2)
    assert np.count_nonzero(result) == 2
    native[0, 0] = np.nan
    with pytest.raises(DataError, match='finite'):
        reduce_native_depth_for_fusion(native)


def test_all_training_anchor_budget_preserves_frozen_holdouts():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
    from tum_rgb_stereo import select_pairs, HOLDOUT, TOTAL
    camera, _, _ = setup()
    frames = []
    for i in range(TOTAL):
        pose = np.eye(4)
        pose[0, 3] = i * .01
        frames.append({'pose': pose})
    pairs = select_pairs(frames, camera, max_pairs=48, backups_per_left=3)
    assert len({left for left, _ in pairs}) <= 48
    assert len(pairs) <= 48 * 3
    assert pairs
    assert all(a not in HOLDOUT and b not in HOLDOUT for a, b in pairs)
