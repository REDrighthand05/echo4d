import numpy as np
import pytest
from redwood_poisson_supported_holes import supported_faces


def test_only_locally_supported_gap_is_inferred():
    vertices=np.array([[0,0,0],[.06,0,0],[0,.06,0],
                       [1,0,0],[1.06,0,0],[1,.06,0],
                       [2,0,0],[2.06,0,0],[2,.06,0]],float)
    faces=np.array([[0,1,2],[3,4,5],[6,7,8]])
    distances=np.array([.02,.06,.06,.03,.20,.04,.01,.02,.01])
    result=supported_faces(vertices,faces,distances,np.ones(9),
                           [-.1,-.1,-.1],[2.1,.1,.1])
    assert result.tolist()==[True,False,False]


def test_no_unbounded_extrapolation():
    vertices=np.array([[0,0,0],[.1,0,0],[0,.1,0]],float)
    faces=np.array([[0,1,2]])
    distances=np.array([.05,.06,.07])
    density=np.ones(3)
    assert not supported_faces(vertices,faces,distances,density,[1,1,1],[2,2,2]).any()
    assert not supported_faces(vertices,faces,distances,density,[-1,-1,-1],[1,1,1],
                               max_edge=.05).any()


@pytest.mark.parametrize("distances",[[0,.1],[0,-.2,.1],[0,float("nan"),.1]])
def test_invalid_support_fails_closed(distances):
    with pytest.raises(ValueError):
        supported_faces(np.zeros((3,3)),np.array([[0,1,2]]),
                        np.array(distances),np.ones(3),[-1,-1,-1],[1,1,1])
