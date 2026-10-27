"""Independent Moller-Trumbore ray/triangle oracle for frozen desk train z-buffers."""
import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from echo4d.holdout import read_echo_ply, rasterize_mesh_depth
from echo4d.model import Camera, rigid_matrix

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--evidence-zip', required=True, type=Path)
a = p.parse_args()
with zipfile.ZipFile(a.evidence_zip) as z:
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp)/'mesh.ply'
        f.write_bytes(z.read('frozen-inputs/mesh/mesh.ply'))
        vertices, faces = read_echo_ply(f)
    traj = {x['name']:rigid_matrix(x['camera_to_metric_sfm_world'])
            for x in json.loads(z.read('frozen-inputs/metric-trajectory-from-sensor.json'))['frames']}
    split = json.loads(z.read('frozen-inputs/desk-48train-12holdout.json'))
    c = z.read('frozen-inputs/cameras.txt').decode().splitlines()[-1].split()
    cam = Camera(160, 120, *[float(v)/4 for v in c[4:]])
    count, maxdiff = 0, 0.
    for name in [split['train'][i]['name'] for i in (0,11,23,35,47)]:
        pose = traj[name]
        rendered, _ = rasterize_mesh_depth(vertices, faces, cam, pose, far=4.)
        yy, xx = np.nonzero(rendered > 0)
        rng = np.random.default_rng(20261027 + len(name) + int(name[6:11]))
        pick = rng.choice(len(xx), size=8, replace=False)
        cv = (vertices-pose[:3,3])@pose[:3,:3]
        tri = cv[faces]
        tri = tri[np.all(tri[:,:,2]>.15,axis=1)]
        v0=tri[:,0,:]; e1=tri[:,1,:]-v0; e2=tri[:,2,:]-v0
        for ind in pick:
            x,y=int(xx[ind]),int(yy[ind])
            ray=np.array([(x-cam.cx)/cam.fx,(y-cam.cy)/cam.fy,1.])
            h=np.cross(np.broadcast_to(ray,e2.shape),e2)
            det=np.einsum('ij,ij->i',e1,h)
            good=np.abs(det)>1e-12
            inv=np.zeros_like(det);inv[good]=1/det[good]
            s=-v0
            u=inv*np.einsum('ij,ij->i',s,h)
            q=np.cross(s,e1)
            v=inv*(q@ray)
            t=inv*np.einsum('ij,ij->i',e2,q)
            hit=good&(u>=-1e-10)&(v>=-1e-10)&(u+v<=1+1e-10)&(t>=.15)&(t<=4)
            assert hit.any(),(name,x,y)
            diff=abs(float(t[hit].min())-float(rendered[y,x]))
            maxdiff=max(maxdiff,diff);count+=1
    print('independent_ray_oracle_count',count,'max_z_disagreement_m',maxdiff)
    assert count==40 and maxdiff<1e-4
