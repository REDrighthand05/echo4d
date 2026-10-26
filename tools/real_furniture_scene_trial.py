"""Real ECHO mesh trial + public Stanford 'single-chair' RGB-D fallback audit.

Attempts don't certify chairs without photo-specific object masks. Never treats
retrieved ShapeNet CAD as measured 3D. Reports PASS-for-audit / BLOCKED-for-object.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile
from hi3d_candidate_intake import inventory

ECHO={
 'a':('docs/evidence/tum-static-ab-v2/a-known-pose-mesh.ply',
      '7c83972f794dc1ad90041d36475be9d1a46d9916c576f8c754efd132310170b4'),
 'b':('docs/evidence/tum-static-ab-v2/b-estimated-pose-mesh.ply',
      'a5273c0d38117d6f2063b6edc9c49adb9c71f8e38c2f2bc35cfb3d63a33d5897')
}
STANFORD='https://graphics.stanford.edu/projects/objectsensing/data/single-chair/single-chair.zip'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):
            h.update(chunk)
    return h.hexdigest()

def alternate(path,out):
    r={'source_url':STANFORD,
       'dataset':'Stanford Object Sensing 2015 / single-chair real Kinect RGB-D',
       'dataset_license':'CC BY-NC-SA 4.0 (not ECHO MIT)',
       'mesh_source':'never ShapeNet retrieval; no CAD substitute accepted',
       'status':'running'}
    try:
        req=urllib.request.Request(STANFORD,headers={'User-Agent':'echo4d-research/1.0'})
        with urllib.request.urlopen(req,timeout=75) as response, path.open('wb') as file:
            size=0
            while True:
                b=response.read(1048576)
                if not b:break
                size+=len(b)
                if size>180*1024**2:
                    raise ValueError('Source ZIP exceeds frozen 180MiB input bound')
                file.write(b)
        r['zip_bytes']=path.stat().st_size
        r['zip_sha256']=digest(path)
        with zipfile.ZipFile(path) as z:
            files=z.infolist()
            if len(files)>20000 or sum(e.file_size for e in files)>3*1024**3:
                raise ValueError('ZIP decompression budget exceeded')
            names=set()
            for e in files:
                n=PurePosixPath(e.filename)
                if n.is_absolute() or '..' in n.parts or '\\' in e.filename:
                    raise ValueError('ZIP contains unsafe paths')
                if not e.is_dir():names.add(e.filename)
            photos=sorted(x for x in names if x.lower().endswith(('.color.png','.color.jpg','.color.jpeg')))
            depths=sorted(x for x in names if x.lower().endswith('.depth.png'))
            poses=sorted(x for x in names if x.lower().endswith('.pose.txt'))
            matched=[x for x in photos if x.rsplit('.color.',1)[0]+'.depth.png' in names
                          and x.rsplit('.color.',1)[0]+'.pose.txt' in names]
            r.update({'members':len(files),'photos':len(photos),'depths':len(depths),
                      'poses':len(poses),'paired_rgbd_pose':len(matched)})
            if matched:
                image=matched[0]
                if z.getinfo(image).file_size>8*1024**2:
                    raise ValueError('Sample source image too large')
                raw=z.read(image)
                target=out/('public-original-photo'+Path(image).suffix.lower())
                target.write_bytes(raw)
                r.update({'first_photo_member':image,'photo_sha256':hashlib.sha256(raw).hexdigest()})
        r['status']='real_chair_sequence_checked_not_instance_segmentation_verified'
        r['reason']='Dataset is labelled single-chair but source photo foreground pixels have not been verified across frames; background must not be included.'
    except Exception as e:
        r['status']='fallback_unavailable_or_invalid'
        r['failure']={'type':type(e).__name__,'message':str(e)[:600]}
    finally:
        path.unlink(missing_ok=True)
    return r

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fetch-public-fallback',action='store_true')
    a=p.parse_args()
    if a.out.exists():
        raise SystemExit('Output exists; refusing overwrite')
    a.out.mkdir(parents=True)
    report={'version':'echo-real-furniture-first-trial-1',
            'result':'running','verified_independent_furniture_count':0,
            'hi3d_used':False,'invented_geometry_faces':0,
            'echo_source':'PR34 real RGB-D measured-depth static meshes',
            'scenes':{},'fallback':None}
    try:
        for key,(path,expected) in ECHO.items():
            source=a.repo_root/path
            actual=digest(source)
            if actual!=expected:
                raise ValueError('Changed real ECHO source mesh hash: '+path)
            scanned=inventory(source)
            report['scenes'][key]={
                 'source_path':path,'sha256':actual,
                 'vertices':scanned['vertices'],'triangles':scanned['triangles'],
                 'connected_surface_components':scanned['face_connected_component_count'],
                 'bounds_as_encoded':scanned['bounds_as_encoded'],
                 'attempted_object':'one chair or desk',
                 'decision':'BLOCKED_no_original_photo_instance_mask_and_metric_depth_association',
                 'photo_to_object_identity_verified':False,
                 'explanation':'A mesh component is not a chair/table label. No multi-view human-reviewed mask matches these room meshes.'}
        report['fallback']=(alternate(a.out/'single-chair-original.zip',a.out)
            if a.fetch_public_fallback else
            {'status':'not_downloaded','source_url':STANFORD})
        report['result']='blocked_missing_furniture_instance_evidence'
    except Exception as exc:
        report['result']='failed_input_or_execution'
        report['failure']={'type':type(exc).__name__,'message':str(exc)}
    finally:
        (a.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        (a.out/'SUMMARY.md').write_text(
             '# First real furniture extraction attempt\n\n'
             f"- Result: **{report['result']}**\n"
             '- Verified standalone furniture: **0**\n'
             '- Both real ECHO source meshes preserved; source SHA256 in report.\n'
             '- Public single-chair alternative kept separate from ECHO world coordinates.\n'
             '- No unobserved structures or AI-generated shape promoted to real geometry.\n')
        print(json.dumps({'result':report['result'],'out':str(a.out/'report.json')}))
    return 1 if report['result']=='failed_input_or_execution' else 0

if __name__=='__main__':
    raise SystemExit(main())
