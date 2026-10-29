"""Run Open3D's complete pinned reconstruction system, without custom pose edges."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

UPSTREAM_SHA = "1e7b17438687a0b0c1e5a7187321ac7044afe275"

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def render_views(o3d, mesh, colors, poses, out):
    import numpy as np
    from PIL import Image, ImageDraw
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    faces = np.asarray(mesh.triangles)
    rgb = np.asarray(mesh.vertex_colors)
    K = np.array([[525., 0, 319.5], [0, 525., 239.5], [0, 0, 1.]])
    # Original pixels are resized for display. Ray calibration matches centers.
    small = K.copy()
    small[:2] /= 2
    small[0, 2] = (K[0, 2] + .5) / 2 - .5
    small[1, 2] = (K[1, 2] + .5) / 2 - .5
    center = (mesh.get_min_bound() + mesh.get_max_bound()) / 2
    span = mesh.get_max_bound() - mesh.get_min_bound()
    views = []
    for i in (0, len(colors)//2, len(colors)-1):
        ext = np.linalg.inv(poses[i])
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(small.astype(np.float32)),
            o3d.core.Tensor(ext.astype(np.float32)), 320, 240)
        views.append((f"recorded-{i:05d}.png", rays, colors[i]))
    for j, offset in enumerate(([-.5,-.9,-.7],[.8,-1.2,.05]), 1):
        eye = center + span * np.asarray(offset)
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            78., o3d.core.Tensor(center.astype(np.float32)),
            o3d.core.Tensor(eye.astype(np.float32)),
            o3d.core.Tensor(np.array([0.,-1.,0.],np.float32)), 320, 240)
        views.append((f"free-{j}.png", rays, None))
    (out/"views").mkdir()
    for name, rays, photo in views:
        hit = scene.cast_rays(rays)
        ids = hit["primitive_ids"].numpy().reshape(-1)
        uv = hit["primitive_uvs"].numpy().reshape(-1,2)
        dist = hit["t_hit"].numpy().reshape(-1)
        ok = np.isfinite(dist) & (dist > .15) & (dist < 16.) & (ids < len(faces))
        pixels = np.zeros((320*240,3),np.uint8)
        if np.any(ok):
            weights = np.column_stack((1-uv[ok].sum(axis=1),uv[ok]))
            values = (rgb[faces[ids[ok].astype(int)]]*weights[:,:,None]).sum(axis=1)
            pixels[ok] = np.clip(values*255,0,255).astype(np.uint8)
        panels = [Image.fromarray(pixels.reshape(240,320,3))]
        if photo:
            with Image.open(photo) as im:
                panels.insert(0,im.convert("RGB").resize((320,240)))
        canvas = Image.new("RGB",(320*len(panels),268),(24,24,24))
        ImageDraw.Draw(canvas).text((8,8),"REAL PHOTO / OFFICIAL COMPLETE PIPELINE" if photo else "FREE CAMERA / ACTUAL RECONSTRUCTED MESH",fill="white")
        for j, panel in enumerate(panels):
            canvas.paste(panel,(320*j,28))
        canvas.save(out/"views"/name)

def run(upstream, output):
    import numpy as np
    import open3d as o3d
    import trimesh
    upstream, output = Path(upstream).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Fresh output directory required")
    actual = subprocess.check_output(["git","rev-parse","HEAD"],cwd=upstream,text=True).strip()
    if actual != UPSTREAM_SHA:
        raise ValueError("Unexpected upstream commit")
    output.mkdir(parents=True)
    system = upstream/"examples/python/reconstruction_system"
    sys.path.insert(0,str(system))
    from data_loader import lounge_data_loader
    from initialize_config import initialize_config
    config = lounge_data_loader()
    initialize_config(config)
    # Use upstream Lounge defaults; all original RGB-D, not 480 sampled frames.
    root = Path(config["path_dataset"])
    data = o3d.data.LoungeRGBDImages()
    colors = [Path(p) for p in data.color_paths]
    depths = [Path(p) for p in data.depth_paths]
    if not colors or len(colors) != len(depths):
        raise ValueError("Missing original RGB-D")
    if any(a.stem != b.stem for a,b in zip(colors,depths)):
        raise ValueError("RGB/depth identities differ")
    ledger = [{"index":i,"color":a.name,"depth":b.name,
               "color_sha256":digest(a),"depth_sha256":digest(b)}
              for i,(a,b) in enumerate(zip(colors,depths))]
    (output/"inputs.json").write_text(json.dumps(ledger,indent=2)+"\n")
    config_path = output/"config.json"
    config_path.write_text(json.dumps(config,indent=2)+"\n")
    receipt = {"upstream": "https://github.com/isl-org/Open3D",
               "upstream_commit":actual,"open3d_version":o3d.__version__,
               "real_frames":len(colors),"stage":"STARTED",
               "input":"native RGB-D; no logged GT trajectory or prior mesh used",
               "visual_acceptance":"NOT REVIEWED"}
    start=time.time()
    try:
        subprocess.run([sys.executable,str(system/"run_system.py"),
                        "--config",str(config_path),"--make","--register",
                        "--refine","--integrate"],check=True)
        scene=root/"scene"
        shutil.copytree(scene,output/"scene")
        # Preserve all estimated fragment graphs, without duplicating RGB-D.
        graphs=output/"fragment-graphs"
        graphs.mkdir()
        for graph in (root/"fragments").glob("*.json"):
            shutil.copyfile(graph,graphs/graph.name)
        mesh=o3d.io.read_triangle_mesh(str(scene/"integrated.ply"))
        if not mesh.has_triangles() or not mesh.has_vertex_colors():
            raise RuntimeError("No colored native mesh")
        xyz=np.asarray(mesh.vertices)
        faces=np.asarray(mesh.triangles)
        rgba=np.clip(np.asarray(mesh.vertex_colors)*255,0,255).astype(np.uint8)
        model=trimesh.Trimesh(xyz,faces,vertex_colors=rgba,process=False)
        model.export(output/"room-full.glb")
        preview=mesh.simplify_quadric_decimation(300000) if len(faces)>300000 else mesh
        trimesh.Trimesh(np.asarray(preview.vertices),np.asarray(preview.triangles),
            vertex_colors=np.clip(np.asarray(preview.vertex_colors)*255,0,255).astype(np.uint8),
            process=False).export(output/"room-preview.glb")
        lines=(scene/"trajectory.log").read_text().splitlines()
        poses=[np.array([float(v) for row in lines[i+1:i+5] for v in row.split()]).reshape(4,4)
               for i in range(0,len(lines),5)]
        if len(poses)!=len(colors):
            raise RuntimeError("Estimated pose/frame count differs")
        render_views(o3d,mesh,colors,poses,output)
        receipt.update(stage="COMPLETE",faces=len(faces),vertices=len(xyz),
                       full_glb_sha256=digest(output/"room-full.glb"),
                       preview_glb_sha256=digest(output/"room-preview.glb"),
                       seconds=time.time()-start)
    except Exception as exc:
        receipt.update(stage="FAILED",error=f"{type(exc).__name__}: {exc}",
                       seconds=time.time()-start)
        raise
    finally:
        (output/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    run(args.upstream,args.output)
