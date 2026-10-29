import numpy as np
import pytest
from stanford_observed_prior_shell import broad_prior_shell,retain_observed_faces


def grid_plane(axis, value, length=4., width=3., steps=24):
    other=[i for i in range(3) if i!=axis]
    uv=np.array([(i*length/steps,j*width/steps) for i in range(steps+1) for j in range(steps+1)])
    xyz=np.zeros((len(uv),3))
    xyz[:,axis]=value
    xyz[:,other]=uv
    faces=[]
    for i in range(steps):
        for j in range(steps):
            a=i*(steps+1)+j
            faces.extend([[a,a+1,a+steps+1],[a+1,a+steps+2,a+steps+1]])
    return xyz,np.asarray(faces,dtype=int)


def test_large_floor_wall_selected_but_furniture_table_not():
    floor,ff=grid_plane(2,0)
    wall,wf=grid_plane(0,0)
    table,tf=grid_plane(2,.8,length=1.2,width=.8,steps=12)
    faces=np.concatenate([ff,wf+len(floor),tf+len(floor)+len(wall)])
    xyz=np.concatenate([floor,wall,table])
    mask,planes=broad_prior_shell(xyz,faces,min_area=1.5)
    assert len(planes)>=2
    assert mask[:len(ff)].sum()>0
    assert mask[len(ff):len(ff)+len(wf)].sum()>0
    assert mask[-len(tf):].sum()==0


def test_only_coincident_wall_removed_not_table_in_front():
    wall,wf=grid_plane(0,0)
    table,tf=grid_plane(0,.5)
    obs=np.concatenate([wall,table])
    of=np.concatenate([wf,tf+len(wall)])
    keep,n=retain_observed_faces(obs,of,wall,wf)
    assert n==len(wf)
    assert np.all(~keep[:len(wf)])
    assert np.all(keep[len(wf):])
