"""Stanford Lounge 3 recorded RGB cameras + 2 ROOM-SCALE free cameras, full rays."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw


def render(out:Path,cache:Path):
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
    measured=trimesh.load(str(out/"OBSERVED-plus-INFERRED.glb"),force="mesh",process=False)
    observed_faces=json.loads((out/"result.json").read_text())["observed_triangles"]
    baseline=measured.copy()
    baseline.update_faces(np.arange(observed_faces))
    baseline.remove_unreferenced_vertices()
    def convert(mesh):
        result=o3d.geometry.TriangleMesh()
        result.vertices=o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        result.triangles=o3d.utility.Vector3iVector(np.asarray(mesh.faces,dtype=np.int32))
        result.vertex_colors=o3d.utility.Vector3dVector(
            np.asarray(mesh.visual.vertex_colors)[:,:3]/255.)
        return result
    original,new=convert(baseline),convert(measured)
    scenes=[]
    for mesh in (original,new):
        scene=o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        scenes.append(scene)
    views=[(i,"recorded",poses[i]) for i in (frames[0],frames[len(frames)//2],frames[-1])]
    bounds=measured.bounds
    center=bounds.mean(axis=0)
    span=bounds[1]-bounds[0]
    # Room-scale third-person viewpoints, NOT 35cm shifts beside input cameras.
    views.extend([(1,"free",center+span*np.array([-.50,-.90,-.70])),
                  (2,"free",center+span*np.array([.80,-1.20,.05]))])

    def wide_free(scene,mesh,eye):
        rays=o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            fov_deg=78.,
            center=o3d.core.Tensor(center.astype(np.float32)),
            eye=o3d.core.Tensor(np.asarray(eye,dtype=np.float32)),
            up=o3d.core.Tensor(np.array([0.,-1.,0.],dtype=np.float32)),
            width_px=320,height_px=240)
        hits=scene.cast_rays(rays)
        dist=hits["t_hit"].numpy().reshape(-1)
        face_ids=hits["primitive_ids"].numpy().reshape(-1)
        uv=hits["primitive_uvs"].numpy().reshape(-1,2)
        faces=np.asarray(mesh.triangles)
        valid=np.isfinite(dist)&(dist>.15)&(dist<16.)&(face_ids<len(faces))
        pixels=np.zeros((320*240,3),dtype=np.uint8)
        if np.any(valid):
            pixels[valid]=barycentric_rgb(face_ids[valid].astype(np.int64),uv[valid],
                                          np.asarray(mesh.vertex_colors),faces)
        return pixels.reshape(240,320,3),int(valid.sum())

    target=out/"three-real-two-free"
    target.mkdir(exist_ok=False)
    records=[]
    for index,kind,extrinsic in views:
        if kind=="recorded":
            before,n0=raycast(o3d,scenes[0],original,k,extrinsic)
            after,n1=raycast(o3d,scenes[1],new,k,extrinsic)
        else:
            before,n0=wide_free(scenes[0],original,extrinsic)
            after,n1=wide_free(scenes[1],new,extrinsic)
        panels=[Image.fromarray(before),Image.fromarray(after)]
        if kind=="recorded":
            with Image.open(paths[index]) as img:
                panels.insert(0,img.convert("RGB").resize((320,240)))
            name=f"photo-vs-mesh-{index:05d}.png"
        else:
            name=f"free-view-{index}.png"
        canvas=Image.new("RGB",(320*len(panels),268),(24,24,24))
        ImageDraw.Draw(canvas).text((8,8),
             "REAL RGB / MEASURED / MEASURED + INFERRED" if kind=="recorded" else
             "ROOM-SCALE FREE / MEASURED / MEASURED + INFERRED",fill="white")
        for j,panel in enumerate(panels):
            canvas.paste(panel,(j*320,28))
        canvas.save(target/name)
        records.append({"kind":kind,"file":"three-real-two-free/"+name,
                        "before_hit_pixels":n0,"after_hit_pixels":n1,
                        "free_camera":"room-scale, 78 degree FOV, 16m far" if kind=="free" else None})
    report=json.loads((out/"result.json").read_text())
    report["three_real_two_free_views"]=records
    report["recorded_pose_mapping"]=mapping
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
    return records


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--cache",required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(render(args.output,args.cache)))
