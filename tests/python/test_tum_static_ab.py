"""Checks for preregistered same-input A/B and anti-cheating guards."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from tum_static_ab import (verify_v2_split,strict_truth,check_gate,
                           FAIR_AB_V2_SELECTED_INDICES, FAIR_AB_V2_HOLDOUT_INDICES)
from echo4d.model import DataError


def _split():
    selected=[f'frame_{i:05d}.png' for i in FAIR_AB_V2_SELECTED_INDICES]
    hold=[f'frame_{i:05d}.png' for i in FAIR_AB_V2_HOLDOUT_INDICES]
    return {'train':[{'name':s} for s in selected if s not in hold],
            'holdout':hold,
            'protocol':{'ab_version':'AB-V2-final',
                        'selected_source_indices':list(FAIR_AB_V2_SELECTED_INDICES),
                        'holdout_indices':list(FAIR_AB_V2_HOLDOUT_INDICES),
                        'missing_gt_source_index_predeclared_excluded':25}}


def test_exact_48_12_no_missing_gt_frame25():
    train,held=verify_v2_split(_split())
    assert len(train)==48 and len(held)==12 and len(set(train+held))==60
    assert 'frame_00025.png' not in train+held
    assert 'frame_00060.png' in train


@pytest.mark.parametrize('mutation',["wrong_split", "missing_view", "train_duplicate", "wrong_version", "leaked_missing_gt"])
def test_reject_ad_hoc_cohort_after_results(mutation):
    s=_split()
    if mutation=='wrong_split':s['holdout'][0]='frame_00001.png'
    if mutation=='missing_view':s['holdout'].pop()
    if mutation=='train_duplicate':s['train'][0]['name']=s['train'][1]['name']
    if mutation=='wrong_version':s['protocol']['ab_version']='legacy-B-V1'
    if mutation=='leaked_missing_gt':s['train'][0]['name']='frame_00025.png'
    with pytest.raises(DataError,match='protocol'):
        verify_v2_split(s)


def test_all_sixty_real_gt_poses_required_no_silent_eleven():
    train,held=verify_v2_split(_split())
    ident=np.eye(4).flatten(order='F').tolist()
    truth={name:ident for name in train+held}
    assert len(strict_truth(truth,train,held))==12
    del truth[held[0]]
    with pytest.raises(DataError,match='unavailable'):
        strict_truth(truth,train,held)


def test_gate_accounts_for_uncovered_pixels():
    good={'holdout_frames':12,'valid_measured_pixels':100,
          'overlap_pixels':85,'coverage':.85,'rmse_m':.12}
    assert check_gate('B',good)['passed']
    assert not check_gate('A',{**good,'coverage':.79,'overlap_pixels':79})['passed']
    assert not check_gate('A',{**good,'rmse_m':.2})['passed']
    with pytest.raises(DataError):check_gate('B',{**good,'overlap_pixels':84})
