"""Fixed Stanford Lounge 3 recorded RGB camera poses + 2 free poses, full mesh rays."""
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
    from redwood_dense_native_review import raycast
    from tum_graph_visual import look_at
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
    for j,(i,dx) in enumerate(((frames[len(frames)//3],.35),
                               (frames[2*len(frames)//3],-.35)),1):
        pose=np.linalg.inv(poses[i])
        eye=pose[:3,3]+dx*pose[:3,0]
        views.append((j,"free",np.linalg.inv(look_at(eye,pose[:3,3]+1.8*pose[:3,2]))))
    target=out/"three-real-two-free"
    target.mkdir(exist_ok=False)
    records=[]
    for index,kind,extrinsic in views:
        before,n0=raycast(o3d,scenes[0],original,k,extrinsic)
        after,n1=raycast(o3d,scenes[1],new,k,extrinsic)
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
             "FREE CAMERA / MEASURED / MEASURED + INFERRED",fill="white")
        for j,panel in enumerate(panels):
            canvas.paste(panel,(j*320,28))
        canvas.save(target/name)
        records.append({"kind":kind,"file":"three-real-two-free/"+name,
                        "before_hit_pixels":n0,"after_hit_pixels":n1})
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
