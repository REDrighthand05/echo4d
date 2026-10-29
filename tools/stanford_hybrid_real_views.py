"""Actual Stanford Lounge photo/observed/hybrid, plus two room-scale views."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def review(out:Path,cache:Path,observed_path:Path,observed_report:Path):
    import open3d as o3d
    import trimesh
    from open3d_official_room_baseline import spaced_real_frames
    from stanford_lounge_second_room import lounge_recorded_poses
    from redwood_dense_native_review import raycast,barycentric_rgb

    data=o3d.data.LoungeRGBDImages(data_root=str(cache))
    paths=list(data.color_paths)
    trajectory=o3d.io.read_pinhole_camera_trajectory(data.trajectory_log_path)
    k,poses,mapping=lounge_recorded_poses(trajectory,paths,data.trajectory_log_path)
    frames=spaced_real_frames(len(paths),360)
    measured=trimesh.load(str(observed_path),force='mesh',process=False)
    n=int(json.loads(observed_report.read_text())['observed_triangles'])
    measured.update_faces(np.arange(n))
    measured.remove_unreferenced_vertices()
    hybrid=trimesh.load(str(out/'HYBRID-observed-plus-PRIOR-shell.glb'),force='mesh',process=False)

    def legacy(mesh):
        converted=o3d.geometry.TriangleMesh()
        converted.vertices=o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        converted.triangles=o3d.utility.Vector3iVector(np.asarray(mesh.faces,dtype=np.int32))
        converted.vertex_colors=o3d.utility.Vector3dVector(
            np.asarray(mesh.visual.vertex_colors)[:,:3].astype(float)/255.)
        return converted

    old,new=legacy(measured),legacy(hybrid)
    scenes=[]
    for mesh in (old,new):
        scene=o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        scenes.append(scene)
    center=np.asarray(hybrid.bounds).mean(axis=0)
    span=np.diff(np.asarray(hybrid.bounds),axis=0)[0]
    views=[('recorded',i,poses[i]) for i in (frames[0],frames[len(frames)//2],frames[-1])]
    views.extend([('free',1,center+span*np.array([-.50,-.90,-.70])),
                  ('free',2,center+span*np.array([.80,-1.20,.05]))])

    def free_raycast(scene,mesh,eye):
        rays=o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            fov_deg=78.,center=o3d.core.Tensor(center.astype(np.float32)),
            eye=o3d.core.Tensor(np.asarray(eye,dtype=np.float32)),
            up=o3d.core.Tensor(np.array([0.,-1.,0.],dtype=np.float32)),
            width_px=320,height_px=240)
        hits=scene.cast_rays(rays)
        dist=hits['t_hit'].numpy().reshape(-1)
        ids=hits['primitive_ids'].numpy().reshape(-1)
        uv=hits['primitive_uvs'].numpy().reshape(-1,2)
        faces=np.asarray(mesh.triangles)
        good=np.isfinite(dist)&(dist>.15)&(dist<16.)&(ids<len(faces))
        pixels=np.zeros((320*240,3),dtype=np.uint8)
        if np.any(good):
            pixels[good]=barycentric_rgb(ids[good].astype(np.int64),uv[good],
                                         np.asarray(mesh.vertex_colors),faces)
        return pixels.reshape(240,320,3),int(good.sum())

    target=out/'three-real-two-free'
    target.mkdir(exist_ok=False)
    records=[]
    for kind,index,pose in views:
        if kind=='recorded':
            before,n0=raycast(o3d,scenes[0],old,k,pose)
            after,n1=raycast(o3d,scenes[1],new,k,pose)
        else:
            before,n0=free_raycast(scenes[0],old,pose)
            after,n1=free_raycast(scenes[1],new,pose)
        panels=[Image.fromarray(before),Image.fromarray(after)]
        if kind=='recorded':
            with Image.open(paths[index]) as photo:
                panels.insert(0,photo.convert('RGB').resize((320,240)))
            name=f'photo-vs-mesh-{index:05d}.png'
        else:
            name=f'free-view-{index}.png'
        canvas=Image.new('RGB',(320*len(panels),268),(24,24,24))
        ImageDraw.Draw(canvas).text((8,8),
            'REAL RGB / MEASURED / MEASURED + PRIOR SHELL' if kind=='recorded' else
            'ROOM-SCALE FREE / MEASURED / MEASURED + PRIOR SHELL',fill='white')
        for j,panel in enumerate(panels):
            canvas.paste(panel,(j*320,28))
        canvas.save(target/name)
        records.append({'kind':kind,'file':'three-real-two-free/'+name,
                        'before_pixels':n0,'hybrid_pixels':n1,
                        'room_scale_free_camera':kind=='free'})
    report=json.loads((out/'result.json').read_text())
    report['three_real_two_free_views']=records
    report['pose_mapping']=mapping
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--observed',type=Path,required=True)
    p.add_argument('--observed-report',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(review(a.output,a.cache,a.observed,a.observed_report)))
