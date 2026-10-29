"""Original full-resolution Stanford scan: test real missing geometry, never replace measured surfaces.

The prior replacement experiment removed 74,641 observed triangles and visibly
worsened some recorded views. This variant NEVER removes measured geometry.
The already-aligned reference is used only where observed vertices do not
support a broad outer wall/floor/ceiling. It remains explicitly prior geometry.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from stanford_observed_prior_shell import broad_prior_shell,merge_colored


def boundary_shell_mask(vertices,faces,*,margin_m=.38):
    """Select broad prior planes close to the scan's external bounds.

    This excludes large interior horizontal surfaces such as tables; it is a
    deliberately conservative shell, not a complete semantic room model.
    """
    raw,planes=broad_prior_shell(vertices,faces,min_area=1.5)
    mins=np.min(vertices,axis=0)
    maxs=np.max(vertices,axis=0)
    eligible=[]
    keep=np.zeros(len(faces),dtype=bool)
    tris=np.asarray(vertices)[np.asarray(faces)]
    for plane in planes:
        axis=plane['axis']
        pos=plane['coordinate_m']
        if min(abs(pos-mins[axis]),abs(pos-maxs[axis]))>margin_m:
            continue
        on_plane=np.max(np.abs(tris[:,:,axis]-pos),axis=1)<=.065
        keep|=raw&on_plane
        eligible.append(plane)
    return keep,eligible


def unsupported_prior_faces(observed_vertices,prior_vertices,prior_faces,*,gap_m=.075):
    """Require all three existing-scan vertices to lack nearby measured support.

    Reject overlap from the prior, not observed faces. This avoids removing
    furniture/wall observations even if the registration is only 4 cm RMS.
    """
    if not .04<=gap_m<=.15:
        raise ValueError('Unsafe overlap tolerance')
    observed=np.asarray(observed_vertices,dtype=float)
    prior=np.asarray(prior_vertices,dtype=float)
    faces=np.asarray(prior_faces)
    if not len(faces):
        return np.empty(0,dtype=bool),0
    if not np.isfinite(observed).all() or not np.isfinite(prior).all():
        raise ValueError('Nonfinite scene geometry')
    distances=cKDTree(observed).query(prior,k=1,workers=2)[0]
    keep=np.all(distances[faces]>gap_m,axis=1)
    return keep,int(np.count_nonzero(~keep))



def run(observed_path,prior_report_path,observed_report_path,cache,output):
    """Use the ORIGINAL official scan, not the 169999-face preview."""
    import open3d as o3d
    import trimesh
    from redwood_prior_first_room import transfer_observed_colors,export_colored
    if output.exists():
        raise FileExistsError("Use new output")
    info=json.loads(observed_report_path.read_text())
    pinned=json.loads(prior_report_path.read_text())["prior"]
    if (pinned["state"]!="ALIGNED_PRIOR_ONLY" or
            pinned["registration"]["fitness"]<.85 or
            pinned["registration"]["rmse_m"]>.09):
        raise ValueError("No reliable existing scan registration")
    observed=trimesh.load(str(observed_path),force="mesh",process=False)
    n=int(info["observed_triangles"])
    if n<1000 or len(observed.faces)!=info["output_triangles"]:
        raise ValueError("Native observed geometry boundary changed")
    observed.update_faces(np.arange(n))
    observed.remove_unreferenced_vertices()
    data=o3d.data.LoungeRGBDImages(data_root=str(cache))
    scan=o3d.io.read_triangle_mesh(data.reconstruction_path)
    original_faces=len(scan.triangles)
    if original_faces!=pinned["original_faces"] or original_faces<2000000:
        raise ValueError("Original un-decimated official scan unavailable")
    transform=np.asarray(pinned["registration"]["prior_to_measured"],dtype=float)
    if transform.shape!=(4,4) or not np.isfinite(transform).all():
        raise ValueError("Invalid pinned original-scan alignment")
    scan.transform(transform)
    scan.compute_vertex_normals()
    measured=o3d.geometry.TriangleMesh()
    measured.vertices=o3d.utility.Vector3dVector(np.asarray(observed.vertices))
    measured.triangles=o3d.utility.Vector3iVector(np.asarray(observed.faces,dtype=np.int32))
    measured.vertex_colors=o3d.utility.Vector3dVector(
        np.asarray(observed.visual.vertex_colors)[:,:3].astype(float)/255.)
    measured.compute_vertex_normals()
    output.mkdir(parents=True)
    # Always export a complete, explicitly labeled PRE-EXISTING reference.
    colors,source=transfer_observed_colors(scan,measured,max_distance_m=.09)
    xyz=np.asarray(scan.vertices)
    faces=np.asarray(scan.triangles)
    export_colored(output/"PREEXISTING-original-full-RGB.glb",xyz,faces,colors)
    mask,planes=boundary_shell_mask(xyz,faces)
    candidates=np.flatnonzero(mask)
    keep,discarded=unsupported_prior_faces(
        np.asarray(observed.vertices),xyz,faces[candidates],gap_m=.075)
    chosen=np.zeros(len(faces),dtype=bool)
    chosen[candidates[keep]]=True
    # Preserve every measured triangle; never replace furniture.
    hybrid_file=None
    if int(chosen.sum())>=100:
        scan_part=trimesh.Trimesh(vertices=xyz,faces=faces,
            vertex_colors=np.column_stack((
                np.rint(np.clip(colors,0,1)*255).astype(np.uint8),
                np.full(len(xyz),255,dtype=np.uint8))),process=False)
        scan_part.update_faces(chosen)
        scan_part.remove_unreferenced_vertices()
        combined=trimesh.util.concatenate((observed,scan_part))
        hybrid_file="OBSERVED-plus-original-PRIOR-shell.glb"
        combined.export(str(output/hybrid_file))
        origin=np.zeros((len(combined.vertices),4),dtype=np.uint8)
        origin[:len(observed.vertices)]=[36,181,207,255]
        origin[len(observed.vertices):]=[244,143,55,255]
        combined.visual.vertex_colors=origin
        combined.export(str(output/"OBSERVED-vs-PREEXISTING-source.glb"))
    report={"scene":"REAL Stanford Lounge 640x480 RGB-D and recorded camera poses",
        "original_prior_geometry":"PREEXISTING full official LoungeRGBDImages scan",
        "prior_independently_recovered_from_video":False,
        "observed_faces_preserved":n,
        "original_prior_faces":original_faces,
        "prior_preview_faces_last_round":pinned["preview_faces"],
        "prior_colored_vertex_fraction":float(np.mean(source==1)),
        "outer_planes":planes,"outer_shell_candidates":len(candidates),
        "original_scan_overlap_rejected":discarded,
        "original_scan_missing_shell_faces":int(chosen.sum()),
        "hybrid_file":hybrid_file,
        "full_prior_file":"PREEXISTING-original-full-RGB.glb",
        "visual_review":"NOT REVIEWED",
        "free_navigation":"NOT ACCEPTED",
        "issue_74":"FAIL unchanged",
        "desk_v2":"FAIL 16.7779cm > 15cm unchanged"}
    (output/"result.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"original_faces":original_faces,
        "missing_prior_faces":int(chosen.sum()),"hybrid":hybrid_file}),flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("observed","prior-report","observed-report","cache","output"):
        p.add_argument("--"+key,required=True,type=Path)
    a=p.parse_args()
    try:run(a.observed,a.prior_report,a.observed_report,a.cache,a.output)
    except Exception as e:
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/"FAILURE.txt").write_text(f"{type(e).__name__}: {e}\n")
        raise
