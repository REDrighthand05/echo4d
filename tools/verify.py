"""Numerical tests plus an end-to-end Python to JavaScript scene contract check."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    node = shutil.which('node')
    if node is None: raise SystemExit('Node.js 22+ required for browser contract tests')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT/'src') + os.pathsep + env.get('PYTHONPATH','')
    def run(args):
        print('+ '+' '.join(args),flush=True)
        subprocess.run(args,cwd=ROOT,env=env,check=True)
    run([sys.executable,'-m','pytest','-q'])
    with tempfile.TemporaryDirectory(prefix='echo4d-verify-') as directory:
        work=Path(directory);data=work/'data';result=work/'result';cli=[sys.executable,'-m','echo4d']
        run(cli+['demo','--output',str(data),'--frames','8'])
        run(cli+['validate',str(data/'recording.json')])
        run(cli+['reconstruct',str(data/'recording.json'),'--output',str(result),'--bounds','-1.8','-.2','-1.9','1.8','2.2','1.8','--voxel','.06','--quiet'])
        env['ECHO_NATIVE_SCENE']=str(result/'scene.echo.json')
        run([node,'--test',*[str(p) for p in sorted((ROOT/'tests').glob('*.test.mjs'))]])
        report=json.loads((result/'report.json').read_text())
        print(json.dumps({'verified':True,'smoke_mesh':report['mesh'],'mesh_sha256':report['outputs']['mesh.ply']},indent=2))


if __name__=='__main__':main()
