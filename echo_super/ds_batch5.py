"""DS batch 5 (2026-11-02): hot-update pure kernels.

Written by DeepSeek Flash from workorders/WO-update-kernels.md;
reviewed line-by-line and bench-verified by Muse before
registration (see test_ds_batch5.py). Stdlib only, pure,
deterministic: manifest digest, update planning, tree
verification, apply-step ordering (version stamp flips last).
"""
import hashlib
import json
from typing import Any, Dict, List

_HEX = frozenset("0123456789abcdef")

def _validate_path(path: Any) -> None:
    """Validate a POSIX-style relative path."""
    if not isinstance(path, str) or path == "" or path.startswith("/") or "\\" in path:
        raise ValueError("invalid path")
    if any(part == ".." for part in path.split("/")):
        raise ValueError("invalid path")

def _validate_entry(entry: Any) -> None:
    """Validate a file entry with sha256 and size."""
    if not isinstance(entry, dict) or "sha256" not in entry or "size" not in entry:
        raise ValueError("invalid file entry")
    sha = entry["sha256"]
    if not isinstance(sha, str) or len(sha) != 64 or any(ch not in _HEX for ch in sha):
        raise ValueError("invalid sha256")
    size = entry["size"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("invalid size")

def _validate_file_map(files: Any) -> None:
    """Validate a mapping of paths to file entries."""
    if not isinstance(files, dict):
        raise ValueError("files must be a dict")
    for path, entry in files.items():
        _validate_path(path)
        _validate_entry(entry)

def _validate_manifest(manifest: Any) -> None:
    """Validate a manifest object."""
    if not isinstance(manifest, dict) or "version" not in manifest or "files" not in manifest:
        raise ValueError("invalid manifest")
    _validate_file_map(manifest["files"])

def manifest_digest(manifest: Dict[str, Any]) -> str:
    """Return the canonical SHA-256 digest of a manifest."""
    _validate_manifest(manifest)
    try:
        canonical = json.dumps(
            {"version": manifest["version"], "files": manifest["files"]},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest is not JSON serializable") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def plan_update(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Build a deterministic update plan from old and new manifests."""
    _validate_manifest(old)
    _validate_manifest(new)
    old_files = old["files"]
    new_files = new["files"]
    to_change: List[Dict[str, Any]] = []
    unchanged = 0
    for path in sorted(new_files):
        new_entry = new_files[path]
        old_entry = old_files.get(path)
        if (
            old_entry is not None
            and old_entry["sha256"] == new_entry["sha256"]
            and old_entry["size"] == new_entry["size"]
        ):
            unchanged += 1
        else:
            to_change.append(
                {
                    "path": path,
                    "sha256": new_entry["sha256"],
                    "size": new_entry["size"],
                }
            )
    to_delete = sorted(path for path in old_files if path not in new_files)
    total_bytes = sum(item["size"] for item in to_change)
    return {
        "from_version": old["version"],
        "to_version": new["version"],
        "to_change": to_change,
        "to_delete": to_delete,
        "unchanged": unchanged,
        "total_bytes": total_bytes,
    }

def verify_listing(manifest: Dict[str, Any], listing: Dict[str, Any]) -> Dict[str, Any]:
    """Compare a manifest against an on-disk listing."""
    _validate_manifest(manifest)
    _validate_file_map(listing)
    manifest_files = manifest["files"]
    missing = sorted(path for path in manifest_files if path not in listing)
    extra = sorted(path for path in listing if path not in manifest_files)
    mismatched = []
    for path in sorted(set(manifest_files) & set(listing)):
        manifest_entry = manifest_files[path]
        listing_entry = listing[path]
        if (
            manifest_entry["sha256"] != listing_entry["sha256"]
            or manifest_entry["size"] != listing_entry["size"]
        ):
            mismatched.append(path)
    return {
        "ok": not missing and not mismatched,
        "missing": missing,
        "mismatched": mismatched,
        "extra": extra,
    }

def order_apply_steps(plan: Dict[str, Any]) -> List[Dict[str, str]]:
    """Return deterministic replace-then-delete apply steps from a plan."""
    if not isinstance(plan, dict) or "to_change" not in plan or "to_delete" not in plan:
        raise ValueError("invalid plan")
    to_change = plan["to_change"]
    to_delete = plan["to_delete"]
    if not isinstance(to_change, list) or not isinstance(to_delete, list):
        raise ValueError("invalid plan")
    replace_paths: List[str] = []
    for item in to_change:
        if not isinstance(item, dict) or "path" not in item:
            raise ValueError("invalid to_change entry")
        path = item["path"]
        _validate_path(path)
        replace_paths.append(path)
    delete_paths: List[str] = []
    for path in to_delete:
        _validate_path(path)
        delete_paths.append(path)

    def is_special(path: str) -> bool:
        return path == "installer/version.inc" or path.endswith("/installer/version.inc")

    sorted_replace_paths = sorted(replace_paths)
    normal = [path for path in sorted_replace_paths if not is_special(path)]
    special = [path for path in sorted_replace_paths if is_special(path)]
    steps: List[Dict[str, str]] = []
    for path in normal + special:
        steps.append({"op": "replace", "path": path})
    for path in sorted(delete_paths):
        steps.append({"op": "delete", "path": path})
    return steps