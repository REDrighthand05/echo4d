import numpy as np
from stanford_prior_additive_shell import boundary_shell_mask,unsupported_prior_faces
from test_stanford_observed_prior_shell import grid_plane


def test_only_outer_planes_not_interior_table():
    floor,ff=grid_plane(2,0)
    wall,wf=grid_plane(0,0)
    table,tf=grid_plane(2,.8,length=1.2,width=.8,steps=12)
    xyz=np.concatenate((floor,wall,table))
    faces=np.concatenate((ff,wf+len(floor),tf+len(floor)+len(wall)))
    mask,planes=boundary_shell_mask(xyz,faces)
    assert len(planes)>=2
    assert mask[-len(tf):].sum()==0


def test_observed_overlap_is_never_removed_and_prior_gap_kept():
    observed,of=grid_plane(0,0)
    prior=np.array([[0,0,0],[0,1,0],[0,0,1],[.5,0,0],[.5,1,0],[.5,0,1]],dtype=float)
    faces=np.array([[0,1,2],[3,4,5]])
    keep,discarded=unsupported_prior_faces(observed,prior,faces,gap_m=.075)
    assert keep.tolist()==[False,True]
    assert discarded==1
