import numpy as np
import pytest
from tum_xyz_six_da6_local import correct_local_bias, compare_withheld

def test_smooth_bias_correction():
    y,x=np.indices((100,140))
    truth=(1.5+.002*x+.001*y).astype(np.float32)
    pred=truth+(.08*np.sin(x/30.)).astype(np.float32)
    obs=np.full_like(truth,np.inf)
    obs[::3,::3]=truth[::3,::3]
    out,info=correct_local_bias(pred,obs)
    assert info["inliers"]>1000
    assert np.median(np.abs(out-truth)) < .45*np.median(np.abs(pred-truth))

def test_occlusion_and_missing_depth():
    pred=np.full((80,90),1.8,np.float32);pred[:,:8]=0
    obs=np.full_like(pred,np.inf);obs[::3,::3]=1.72;obs[::7,::7]=3.5
    out,_=correct_local_bias(pred,obs)
    assert not out[:,:8].any()
    assert 1.68 < np.median(out[:,15:]) < 1.8

def test_insufficient_anchors_fail_closed():
    with pytest.raises(RuntimeError,match="Insufficient"):
        correct_local_bias(np.ones((40,50),np.float32),np.full((40,50),np.inf))

def test_evaluation_is_read_only():
    a=np.full((20,20),1.5,np.float32)
    r=compare_withheld(a,a+.03,a+.01)
    assert r["global"]["median_abs_error_m"] < r["local"]["median_abs_error_m"]
    assert np.all(a==1.5)
