"""Synthetic independent counterexamples for train-only mocap pose diagnostic."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
spec = importlib.util.spec_from_file_location('desk_external_pose_audit', Path(__file__).resolve().parents[2] / 'tools' / 'desk_external_pose_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
from echo4d.model import DataError


def fixture():
    names = [f'frame_{i:05d}.png' for i in range(60)]
    train = [n for i, n in enumerate(names) if i not in audit.HOLDOUT_INDICES]
    held = [names[i] for i in audit.HOLDOUT_INDICES]
    rotation = Rotation.from_euler('xyz', [11, -4, 17], degrees=True).as_matrix()
    shift = np.array([1.0, -0.2, 0.7])
    est, ref = {}, {}
    for i, name in enumerate(names):
        center = np.array([i * .02, np.sin(i*.12), np.cos(i*.13)])
        e, r = np.eye(4), np.eye(4)
        e[:3, 3] = center
        r[:3, :3] = rotation
        r[:3, 3] = rotation @ center + shift
        est[name], ref[name] = e, r
    return train, held, est, ref


def test_train_only_alignment_exact_and_heldout_not_fitted():
    train, held, est, ref = fixture()
    before = audit.align_train_only(est, ref, train, held)
    assert before['scale'] == 1
    assert before['train']['position_rmse_m'] < 1e-12
    assert before['holdout']['position_rmse_m'] < 1e-12
    # A large reference error on ONE holdout cannot change the fitted transform.
    ref[held[0]][:3, 3] += [1, 0, 0]
    after = audit.align_train_only(est, ref, train, held)
    np.testing.assert_allclose(before['rotation_rowmajor'], after['rotation_rowmajor'], atol=0, rtol=0)
    np.testing.assert_allclose(before['translation_m'], after['translation_m'], atol=0, rtol=0)
    assert after['holdout']['position_rmse_m'] == pytest.approx((1/12)**.5)


def test_rejects_wrong_split_and_degenerate_training_trajectory():
    train, held, est, ref = fixture()
    with pytest.raises(DataError, match='48/12'):
        audit.align_train_only(est, ref, train[:-1], held)
    with pytest.raises(DataError, match='48/12'):
        audit.align_train_only(est, ref, train, held[:-1]+[train[0]])
    for name in train:
        est[name][:3, 3] = 0
    with pytest.raises(DataError, match='Degenerate'):
        audit.align_train_only(est, ref, train, held)


def test_original_timestamp_bracketing_not_artificial_mp4_time():
    rgb = [[str(1305031000. + i*.01), f'rgb/{i}.png'] for i in range(600)]
    gt = [[str(1305031000. + i*.05), '0', '0', '0', '0', '0', '0', '1'] for i in range(120)]
    poses, ledger = audit.reference_poses(rgb, gt, ['frame_00004.png'])
    assert ledger[0]['original_rgb_timestamp_s'] == pytest.approx(1305031000.32)
    assert ledger[0]['interpolated'] is True
    assert poses['frame_00004.png'].shape == (4, 4)
    with pytest.raises(DataError, match='Noncanonical'):
        audit.reference_poses(rgb, gt, ['bad-name.png'])
    # Entire 100ms gap is exceeded; interpolation MUST fail closed.
    sparse = [gt[0], gt[-1]]
    with pytest.raises(DataError, match='No bracketed'):
        audit.reference_poses(rgb, sparse, ['frame_00004.png'])


def test_rejects_nonfinite_or_nonunit_mocap_quaternion():
    rgb = [[str(1000. + i*.01), f'rgb/{i}.png'] for i in range(600)]
    gt = [[str(1000. + i*.05), '0', '0', '0', '0', '0', '0', '1'] for i in range(120)]
    gt[0][-1] = '2'
    with pytest.raises(DataError, match='quaternion'):
        audit.reference_poses(rgb, gt, ['frame_00000.png'])


def test_missing_pose_even_with_extra_names_fails_closed():
    train, held, est, ref = fixture()
    del est[held[0]]
    est['frame_99999.png'] = np.eye(4)
    with pytest.raises(DataError, match='Missing'):
        audit.align_train_only(est, ref, train, held)
