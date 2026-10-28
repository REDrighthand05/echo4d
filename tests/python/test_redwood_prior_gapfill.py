"""Synthetic geometry tests; real Bedroom reconstruction is an Actions job."""
import numpy as np
import pytest
from redwood_prior_gapfill import missing_prior_faces,safe_registration


def test_prior_only_faces_fill_missing_geometry_not_observed_surface():
    observed=np.array([[0.,0,0],[.1,0,0],[0,.1,0],[.1,.1,0]])
    prior=np.array([[0,0,0],[.1,0,0],[0,.1,0],
                    [1,0,0],[1.1,0,0],[1,.1,0]],dtype=float)
    faces=np.array([[0,1,2],[3,4,5]],dtype=np.int32)
    keep,distance=missing_prior_faces(observed,prior,faces,.08)
    assert keep.tolist()==[False,True]
    assert distance[0]==0 and distance[3]>.08


def test_gap_mask_rejects_nonfinite_or_out_of_range():
    with pytest.raises(ValueError):
        missing_prior_faces(np.zeros((3,3)),np.zeros((3,3)),
                            np.array([[0,1,2]]),.005)
    with pytest.raises(ValueError):
        missing_prior_faces(np.zeros((3,3)),np.zeros((3,3)),
                            np.array([[0,1,3]]),.09)


def test_alignment_guard_blocks_false_priors():
    assert np.array_equal(safe_registration(.75,.025,np.eye(4)),np.eye(4))
    with pytest.raises(ValueError):
        safe_registration(.08,.02,np.eye(4))
    with pytest.raises(ValueError):
        safe_registration(.8,.30,np.eye(4))
    scaled=np.eye(4)
    scaled[0,0]=1.3
    with pytest.raises(ValueError):
        safe_registration(.8,.02,scaled)
