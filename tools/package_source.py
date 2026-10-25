"""Deterministic allowlisted source archive, excluding credentials and raw datasets."""
import argparse
import hashlib
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DIRS=('src','tests','tools','docs','dist','.github')
FILES=('README.md','THIRD_PARTY_NOTICES.md','LICENSE','pyproject.toml','requirements.lock','package.json','Dockerfile','.dockerignore','.gitignore')


def source_files():
    paths=[ROOT/name for name in FILES]
    for name in DIRS:paths.extend((ROOT/name).rglob('*'))
    return sorted(p for p in paths if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts
                  and p.suffix not in ('.pyc','.zip') and not p.name.startswith('.env'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/echo4d-source.zip');args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    paths=source_files()
    with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for p in paths:
            entry=zipfile.ZipInfo('echo4d/'+p.relative_to(ROOT).as_posix(),(2026,1,1,0,0,0))
            entry.compress_type=zipfile.ZIP_DEFLATED;entry.external_attr=0o100644<<16
            archive.writestr(entry,p.read_bytes())
    print(f'{args.output.resolve()}\n{len(paths)} files; sha256={hashlib.sha256(args.output.read_bytes()).hexdigest()}')


if __name__=='__main__':main()
