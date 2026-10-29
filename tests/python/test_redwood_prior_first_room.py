"""No made-up colors on unsupported reference surfaces."""
import numpy as np
import pytest
from redwood_prior_first_room import transfer_observed_colors


class Surface:
    def __init__(self,vertices,colors=None,normals=None):
        self.vertices=vertices
        self.vertex_colors=(np.array(colors,dtype=float) if colors is not None
                            else np.zeros((len(vertices),3)))
        self.vertex_normals=(np.array(normals,dtype=float) if normals is not None
                             else np.tile([0.,0.,1.],(len(vertices),1)))


def test_prior_geometry_kept_and_unknown_color_marked():
    measured=Surface([[0,0,0],[1,0,0],[0,1,0]],
                     colors=[[1,0,0],[0,1,0],[0,0,1]])
    prior=Surface([[.01,0,0],[1.01,0,0],[0,1.01,0],[5,5,5]])
    colors,labels=transfer_observed_colors(prior,measured,max_distance_m=.09)
    assert labels.tolist()==[1,1,1,2]
    assert np.allclose(colors[:3],[[1,0,0],[0,1,0],[0,0,1]])
    assert np.allclose(colors[3],[.53,.56,.59])


def test_opposite_normal_is_not_a_color_match():
    measured=Surface([[0,0,0],[1,0,0],[0,1,0]],
                     colors=[[1,0,0],[1,0,0],[1,0,0]])
    prior=Surface([[0,0,.001],[1,0,.001],[0,1,.001]],
                  normals=[[0,0,-1]]*3)
    colors,labels=transfer_observed_colors(prior,measured)
    assert np.all(labels==2)
    assert np.allclose(colors,[.53,.56,.59])


def test_invalid_mesh_is_rejected():
    with pytest.raises(ValueError):
        transfer_observed_colors(Surface([[0,0,0]]),
                                 Surface([[0,0,0],[1,0,0],[0,1,0]]))
