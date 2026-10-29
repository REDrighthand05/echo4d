"""Independent Stanford Lounge 300-frame reconstruction: COLMAP RGB SfM + measured depth TSDF.

Unlike the failing Open3D independent-300/600 pose graphs, COLMAP uses SIFT,
sequential matching and bundle adjustment. Metric scale is recovered ONLY from
the same 300 selected measured depth frames. No previous trajectory or Mesh.
"""
import argparse, hashlib, json, os, shutil, subprocess, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

def command(argv, log, env):
    with open(log,"w") as f:
        f.write("COMMAND: "+" ".join(map(str,argv))+"\n")
        f.flush()
        subprocess.run(list(map(str,argv)),stdout=f,stderr=subprocess.STDOUT,env=env,check=True)

def original_pairs(data):
    cdir=Path(data.color_paths[0]).parent
    ddir=Path(data.depth_paths[0]).parent
    colors=sorted(p for p in cdir.iterdir() if p.stem.isdecimal() and p.suffix.lower() in (".jpg",".png"))
    depths=sorted(p for p in ddir.iterdir() if p.stem.isdecimal() and p.suffix.lower()==".png")
    if len(colors)!=3000 or len(depths)!=3000 or [p.stem for p in colors]!=[p.stem for p in depths]:
        raise RuntimeError("Expected original 3000 aligned Stanford RGB-D pairs")
    return list(zip(colors,depths))

def read_model(directory):
    points={}
    for line in (directory/"points3D.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            v=line.split()
            points[int(v[0])]=np.array(list(map(float,v[1:4])))
    lines=[x.strip() for x in (directory/"images.txt").read_text().splitlines()
           if x.strip() and not x.startswith("#")]
    if len(lines)%2:
        raise RuntimeError("Malformed COLMAP images.txt")
    images=[]
    for i in range(0,len(lines),2):
        v=lines[i].split()
        qw,qx,qy,qz=map(float,v[1:5])
        ext=np.eye(4)
        ext[:3,:3]=Rotation.from_quat([qx,qy,qz,qw]).as_matrix()
        ext[:3,3]=list(map(float,v[5:8]))
        obs=lines[i+1].split()
        matches=[]
        for j in range(0,len(obs),3):
            pid=int(obs[j+2])
            if pid in points:
                matches.append((float(obs[j]),float(obs[j+1]),points[pid]))
        images.append((v[9],ext,matches))
    return images

def depth_scale(images, depth_paths):
    ratios=[]
    for name,ext,observations in images:
        with Image.open(depth_paths[name]) as im:
            depth=np.asarray(im)
        h,w=depth.shape[:2]
        for x,y,point in observations:
            ix,iy=round(x),round(y)
            if not (0<=ix<w and 0<=iy<h):
                continue
            z=float(depth[iy,ix])/1000.
            sfm=float(ext[2,:3]@point+ext[2,3])
            if .25<z<3.0 and sfm>1e-4:
                ratios.append(z/sfm)
    a=np.asarray(ratios)
    a=a[np.isfinite(a)&(a>0)]
    if len(a)<100:
        raise RuntimeError("Too few matched real depth and COLMAP 3D points")
    m=np.median(a)
    a=a[(a>m*.5)&(a<m*2)]
    if len(a)<100:
        raise RuntimeError("Metric depth scale is not supported")
    return float(np.median(a)),len(a)

def compose(out, n):
    names=[f"recorded-{i:05d}.png" for i in (0,n//2,n-1)]+["free-1.png","free-2.png"]
    image=Image.new("RGB",(640,5*268+34),(20,20,20))
    draw=ImageDraw.Draw(image)
    draw.text((6,10),"real Stanford photos / free views",fill="white")
    draw.text((330,10),"COLMAP independent RGB poses + real depth",fill="white")
    for i,name in enumerate(names):
        with Image.open(out/"views"/name) as im:
            im=im.convert("RGB")
            y=34+i*268
            if im.width==640:
                image.paste(im.crop((0,28,320,268)),(0,y+28))
                image.paste(im.crop((320,28,640,268)),(320,y+28))
            else:
                image.paste(im.crop((0,28,320,268)),(320,y+28))
            draw.text((6,y+5),name[:-4],fill="white")
    image.save(out/"source-colmap300-3plus2.png")

def run(output, count):
    import open3d as o3d
    import trimesh
    from run_official_room import render_views
    output=output.resolve()
    if output.exists():
        raise FileExistsError("Fresh output required")
    output.mkdir(parents=True)
    report=dict(status="RUNNING",dataset="Stanford Lounge original RGB-D",
                selected_rgb=count,selected_measured_depth=count,
                pose_source="COLMAP SIFT RGB SfM, depth only for metric scale and TSDF",
                full_trajectory_prior=False,existing_mesh_prior=False,
                predicted_depth=False,visual_acceptance="NOT REVIEWED")
    def save():
        (output/"receipt.json").write_text(json.dumps(report,indent=2)+"\n")
    save()
    start=time.time()
    try:
        source=original_pairs(o3d.data.LoungeRGBDImages())
        selected=np.linspace(0,len(source)-1,count,dtype=int)
        rgb_dir=output/"colmap_images"
        rgb_dir.mkdir()
        depths={}
        ledger=[]
        for i,index in enumerate(selected):
            rgb,dep=source[index]
            name=f"{i:06d}{rgb.suffix.lower()}"
            dest=rgb_dir/name
            try: os.link(rgb,dest)
            except OSError: shutil.copy2(rgb,dest)
            depths[name]=dep
            ledger.append(dict(sample=i,original_index=int(index),
                               rgb=rgb.name,depth=dep.name,
                               rgb_sha256=hashlib.sha256(rgb.read_bytes()).hexdigest(),
                               depth_sha256=hashlib.sha256(dep.read_bytes()).hexdigest()))
        (output/"inputs.json").write_text(json.dumps(ledger,indent=2)+"\n")
        env=os.environ.copy()
        env.update(QT_QPA_PLATFORM="offscreen",OMP_NUM_THREADS="2")
        db=output/"colmap.db"
        command(["colmap","feature_extractor","--database_path",db,"--image_path",rgb_dir,
                 "--ImageReader.single_camera","1","--ImageReader.camera_model","PINHOLE",
                 "--ImageReader.camera_params","525,525,319.5,239.5",
                 "--SiftExtraction.use_gpu","0","--SiftExtraction.max_num_features","8192"],
                output/"01-features.log",env)
        command(["colmap","sequential_matcher","--database_path",db,
                 "--SequentialMatching.overlap","30","--SiftMatching.use_gpu","0"],
                output/"02-matching.log",env)
        sparse=output/"sparse"
        sparse.mkdir()
        command(["colmap","mapper","--database_path",db,"--image_path",rgb_dir,
                 "--output_path",sparse,"--Mapper.ba_refine_focal_length","0",
                 "--Mapper.ba_refine_principal_point","0","--Mapper.ba_refine_extra_params","0"],
                output/"03-mapper.log",env)
        models=[]
        for model in sparse.iterdir():
            if not model.is_dir():
                continue
            txt=output/f"model-txt-{model.name}"
            txt.mkdir()
            command(["colmap","model_converter","--input_path",model,"--output_path",txt,
                     "--output_type","TXT"],output/f"04-convert-{model.name}.log",env)
            views=read_model(txt)
            models.append((len(views),views,txt.name))
        if not models:
            raise RuntimeError("COLMAP could not recover a room camera model")
        _,views,model_name=max(models,key=lambda v:v[0])
        report.update(registered_frames=len(views),model=model_name)
        save()
        if len(views)<3:
            raise RuntimeError("Too few COLMAP registered frames")
        scale,witnesses=depth_scale(views,depths)
        report.update(metric_scale_m_per_colmap_unit=scale,scale_depth_witnesses=witnesses)
        volume=o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=.025,sdf_trunc=.10,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        intr=o3d.camera.PinholeCameraIntrinsic(640,480,525.,525.,319.5,239.5)
        photos=[]
        poses=[]
        for name,w2c,_ in sorted(views,key=lambda v:v[0]):
            rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.io.read_image(str(rgb_dir/name)),
                o3d.io.read_image(str(depths[name])),
                depth_scale=1000.,depth_trunc=3.,convert_rgb_to_intensity=False)
            metric=w2c.copy()
            metric[:3,3]*=scale
            volume.integrate(rgbd,intr,metric)
            photos.append(rgb_dir/name)
            poses.append(np.linalg.inv(metric))
        mesh=volume.extract_triangle_mesh()
        if not mesh.has_triangles() or not mesh.has_vertex_colors():
            raise RuntimeError("No real colored Mesh")
        xyz=np.asarray(mesh.vertices)
        faces=np.asarray(mesh.triangles)
        rgb=np.clip(np.asarray(mesh.vertex_colors)*255,0,255).astype(np.uint8)
        trimesh.Trimesh(xyz,faces,vertex_colors=rgb,process=False).export(output/"room-colmap300.glb")
        preview=mesh.simplify_quadric_decimation(300000) if len(faces)>300000 else mesh
        trimesh.Trimesh(np.asarray(preview.vertices),np.asarray(preview.triangles),
            vertex_colors=np.clip(np.asarray(preview.vertex_colors)*255,0,255).astype(np.uint8),
            process=False).export(output/"room-colmap300-preview.glb")
        render_views(o3d,mesh,photos,poses,output)
        compose(output,len(photos))
        report.update(status="REAL_MESH_GENERATED_VISUAL_REVIEW_REQUIRED",
                      visual_acceptance="FAIL until image review",
                      mesh="room-colmap300.glb",image="source-colmap300-3plus2.png")
    except Exception as exc:
        report.update(status="FAIL",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"]=round(time.time()-start)
        save()
        shutil.rmtree(output/"colmap_images",ignore_errors=True)
        if (output/"colmap.db").exists():
            (output/"colmap.db").unlink()

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--samples",type=int,default=300)
    a=p.parse_args()
    run(a.output,a.samples)
