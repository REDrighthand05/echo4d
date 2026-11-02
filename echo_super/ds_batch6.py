"""Batch-6 release kernels for DS Flash.

Pure, deterministic helpers for manifest merging, text detection, line-ending
alignment, and change selection. No I/O and no mutation of caller inputs.
"""


def _validate_path(path: object) -> None:
    if not isinstance(path, str) or path == "":
        raise ValueError("invalid path")
    if path.startswith("/") or "\\" in path:
        raise ValueError("invalid path")
    if ".." in path.split("/"):
        raise ValueError("invalid path")


def _validate_sha256(value: object) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("invalid sha256")
    for char in value:
        if char not in "0123456789abcdef":
            raise ValueError("invalid sha256")


def _validate_size(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("invalid size")


def _copy_entry(entry: object) -> dict:
    if not isinstance(entry, dict):
        raise ValueError("invalid file entry")
    if "sha256" not in entry or "size" not in entry:
        raise ValueError("invalid file entry")

    sha256 = entry["sha256"]
    size = entry["size"]
    _validate_sha256(sha256)
    _validate_size(size)
    return {"sha256": sha256, "size": size}


def _copy_file_map(files: object) -> dict:
    if not isinstance(files, dict):
        raise ValueError("invalid file map")

    result = {}
    for path, entry in files.items():
        _validate_path(path)
        result[path] = _copy_entry(entry)
    return result


def merge_manifest(base: dict, overrides: dict, version: str) -> dict:
    """Return a new manifest with overrides applied and version replaced."""
    if not isinstance(base, dict):
        raise ValueError("invalid base manifest")
    if "version" not in base or "files" not in base:
        raise ValueError("invalid base manifest")
    if not isinstance(base["version"], str):
        raise ValueError("invalid base version")
    if not isinstance(version, str):
        raise ValueError("invalid version")
    if not isinstance(overrides, dict):
        raise ValueError("invalid overrides")

    files = _copy_file_map(base["files"])

    for path, entry in overrides.items():
        _validate_path(path)
        if entry is None:
            files.pop(path, None)
        else:
            files[path] = _copy_entry(entry)

    return {"version": version, "files": files}


def looks_text(data: bytes) -> bool:
    """Return True iff data appears to be strict UTF-8 text without early NULs."""
    if not isinstance(data, bytes):
        raise ValueError("data must be bytes")
    if b"\x00" in data[:8192]:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def align_crlf(data: bytes) -> bytes:
    """Normalize all line endings to CRLF, idempotently."""
    if not isinstance(data, bytes):
        raise ValueError("data must be bytes")
    return (
        data.replace(b"\r\n", b"\n")
        .replace(b"\r", b"\n")
        .replace(b"\n", b"\r\n")
    )


def select_changes(base_files: dict, candidates: dict) -> dict:
    """Select changed, deleted, and shippable files between manifests."""
    base = _copy_file_map(base_files)

    if not isinstance(candidates, dict):
        raise ValueError("invalid candidates")

    candidate_map = {}
    for path, candidate in candidates.items():
        _validate_path(path)
        if not path.startswith("app/"):
            raise ValueError("invalid candidate path")
        if not isinstance(candidate, dict):
            raise ValueError("invalid candidate")
        if (
            "text" not in candidate
            or "lf" not in candidate
            or "crlf" not in candidate
        ):
            raise ValueError("invalid candidate")
        if not isinstance(candidate["text"], bool):
            raise ValueError("invalid candidate text")

        candidate_map[path] = {
            "text": candidate["text"],
            "lf": _copy_entry(candidate["lf"]),
            "crlf": _copy_entry(candidate["crlf"]),
        }

    changed = []
    deleted = []
    ship = {}

    for path, base_entry in base.items():
        candidate = candidate_map.get(path)
        if candidate is None:
            if path.startswith("app/"):
                deleted.append(path)
            continue

        if base_entry == candidate["lf"] or base_entry == candidate["crlf"]:
            continue

        changed.append(path)
        ship[path] = "crlf" if candidate["text"] else "raw"

    for path, candidate in candidate_map.items():
        if path not in base:
            changed.append(path)
            ship[path] = "crlf" if candidate["text"] else "raw"

    changed.sort()
    deleted.sort()

    return {"changed": changed, "deleted": deleted, "ship": ship}

