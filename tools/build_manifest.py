"""Build the hot-update manifest for a finished dist tree.

    python tools/build_manifest.py <dist_dir> <version> [asset_out]

Walks <dist_dir>, hashes every file (POSIX relative paths, sha256 +
size) except the .echo state directory, and writes:
  <dist_dir>/.echo/manifest.json   the install's initial truth
  [asset_out]                      same JSON, for the release page
The manifest format is the batch-5 kernel contract (see
echo_super/ds_batch5.py); the file is validated through
manifest_digest before anything is written.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echo_super.ds_batch5 import manifest_digest  # noqa: E402


def build_manifest(dist: Path, version: str) -> dict:
    files = {}
    for item in sorted(dist.rglob("*")):
        if not item.is_file():
            continue
        rel = item.relative_to(dist).as_posix()
        if rel.startswith(".echo/") or "__pycache__" in rel.split("/"):
            continue
        h = hashlib.sha256()
        with item.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        files[rel] = {"sha256": h.hexdigest(), "size": item.stat().st_size}
    manifest = {"version": version, "files": files}
    manifest_digest(manifest)  # raises on any malformed shape
    return manifest


def main(argv) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    dist, version = Path(argv[1]), argv[2]
    manifest = build_manifest(dist, version)
    text = json.dumps(manifest, ensure_ascii=False, indent=1)
    echo = dist / ".echo"
    echo.mkdir(parents=True, exist_ok=True)
    (echo / "manifest.json").write_text(text, encoding="utf-8")
    if len(argv) >= 4:
        Path(argv[3]).write_text(text, encoding="utf-8")
    print(f"manifest: {len(manifest['files'])} files, "
          f"digest {manifest_digest(manifest)[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
