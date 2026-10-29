"""Recover only presentation/export from the completed upstream reconstruction."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

def run(source, output, upstream):
    import numpy as np
    import open3d as o3d
    from run_official_room import digest, render_views, UPSTREAM_SHA
    source, output, upstream = map(lambda p: Path(p).resolve(), (source,output,upstream))
    if output.exists():
        raise FileExistsError("Fresh recovery directory required")
    expected = {
      "room-full.glb":"957a00a5a7e664b5285cca2896b9142798650cd929e91123580e2017de59e0b9",
      "room-preview.glb":"e8e53dd1a80a6d01e970b8fa572c08833f755a23776f1617235f9f8f00df5bfb",
      "scene/integrated.ply":"d429e2de5deea9733b4d9a021dd9a5433d0c8a766dcffcb9df24f32d9f401d19"}
    for rel,wanted in expected.items():
        if digest(source/rel)!=wanted:
            raise ValueError("Completed reconstruction bytes changed: "+rel)
    sys.path.insert(0,str(upstream/"examples/python"))
    from open3d_example import get_rgbd_file_lists
    data=o3d.data.LoungeRGBDImages()
    colors,depths=get_rgbd_file_lists(data.extract_dir)
    colors,depths=list(map(Path,colors)),list(map(Path,depths))
    if len(colors)!=len(depths) or any(a.stem!=b.stem for a,b in zip(colors,depths)):
        raise ValueError("Upstream RGB-D inventory differs")
    original=json.loads((source/"inputs.json").read_text())
    lookup={a.name:(a,b) for a,b in zip(colors,depths)}
    for row in original:
        a,b=lookup[row["color"]]
        if b.name!=row["depth"] or digest(a)!=row["color_sha256"] or digest(b)!=row["depth_sha256"]:
            raise ValueError("Original SDK subset bytes changed")
    lines=(source/"scene/trajectory.log").read_text().splitlines()
    if len(lines)%5:
        raise ValueError("Truncated estimated trajectory")
    poses=[]
    for frame,i in enumerate(range(0,len(lines),5)):
        if list(map(int,lines[i].split()))!=[frame,frame,frame+1]:
            raise ValueError("Estimated trajectory identities differ")
        pose=np.array([float(v) for row in lines[i+1:i+5] for v in row.split()]).reshape(4,4)
        if not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1]):
            raise ValueError("Invalid estimated pose")
        poses.append(pose)
    if len(colors)!=len(poses):
        raise ValueError(f"Upstream inventory {len(colors)} != poses {len(poses)}")
    shutil.copytree(source,output)
    (output/"receipt.json").rename(output/"receipt.original-failed.json")
    (output/"inputs.json").rename(output/"inputs.original-sdk.json")
    ledger=[{"index":i,"color":a.name,"depth":b.name,
             "color_sha256":digest(a),"depth_sha256":digest(b)}
            for i,(a,b) in enumerate(zip(colors,depths))]
    (output/"inputs.json").write_text(json.dumps(ledger,indent=2)+"\n")
    mesh=o3d.io.read_triangle_mesh(str(output/"scene/integrated.ply"))
    render_views(o3d,mesh,colors,poses,output)
    for rel,wanted in expected.items():
        if digest(output/rel)!=wanted:
            raise RuntimeError("Recovery modified original model bytes")
    report={
        "stage":"COMPLETE_POSTPROCESS_RECOVERY",
        "original_run":"https://github.com/REDrighthand05/echo4d/actions/runs/40865675012",
        "original_tested_checkout":"b4e3428442f55cfdf5b217a60ecd1d654fd81f99",
        "upstream_commit":UPSTREAM_SHA,
        "reconstruction_rerun":False,
        "original_sdk_inventory":len(original),
        "actual_upstream_inventory":len(colors),
        "estimated_poses":len(poses),
        "cause":"SDK inventory omits a file present in upstream folder inventory",
        "input":"native measured RGB-D, estimated poses; GT trajectory not read",
        "faces":len(mesh.triangles),"vertices":len(mesh.vertices),
        "model_sha256":expected,
        "views":{p.name:digest(p) for p in (output/"views").glob("*.png")},
        "visual_acceptance":"NOT REVIEWED"}
    (output/"receipt.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--upstream",required=True)
    args=parser.parse_args()
    run(args.source,args.output,args.upstream)
