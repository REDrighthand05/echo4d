"""Realistic geometry guards; no fake dataset required for unit checks."""
import numpy as np
import pytest
from scipy.spatial import cKDTree
from stanford_planar_support import broad_planes,planar_gap_tiles


def synthetic_flat_support(n=55,missing=True):
    grid=np.arange(n,dtype=float)*.07
    ii,jj=np.meshgrid(np.arange(n),np.arange(n),indexing="ij")
    keep=np.ones((n,n),dtype=bool)
    if missing:
        keep[n//2-2:n//2+2,n//2-2:n//2+2]=False
    coords=np.column_stack((grid[ii[keep]],grid[jj[keep]],np.zeros(np.sum(keep))))
    normals=np.tile([0.,0.,1.],(len(coords),1))
    return coords,normals


def test_measured_support_ransac_recovers_large_floor():
    vertices,normals=synthetic_flat_support()
    planes=broad_planes(vertices,normals,trials=80,rounds=2)
    assert len(planes)>=1
    assert len(planes[0][3])>2000
    assert np.abs(planes[0][1][2])>.99


def test_planar_tiles_fill_interior_but_never_extrapolate_room_boundary():
    vertices,normals=synthetic_flat_support()
    center=vertices.mean(axis=0)
    plane=(center,np.array([0.,0.,1.]),np.array([[1.,0.,0.],[0.,1.,0.]]),
           np.arange(len(vertices)),np.array([3.8,3.8]))
    rgb=np.tile([.5,.5,.5],(len(vertices),1))
    out,faces,colors,cells=planar_gap_tiles(vertices,rgb,plane,tree=cKDTree(vertices))
    assert cells>0 and len(faces)==2*cells
    assert colors.shape==(len(out),3)
    assert out.min(axis=0)[0]>=vertices[:,0].min()-1e-5
    assert out.max(axis=0)[0]<=vertices[:,0].max()+1e-5
    assert out.min(axis=0)[1]>=vertices[:,1].min()-1e-5
    assert out.max(axis=0)[1]<=vertices[:,1].max()+1e-5


def test_rejects_nonfinite_or_insufficient_real_points():
    with pytest.raises(ValueError):
        broad_planes(np.ones((5,3)),np.ones((5,3)))
    pts,normals=synthetic_flat_support()
    pts[0,0]=np.nan
    with pytest.raises(ValueError):
        broad_planes(pts,normals)
