"""Index-only RGB/depth pairing audit for frozen Freiburg1 desk V1.

This does not decode depth images, run SfM, pick alternate frames or tune gates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np

from echo4d.model import DataError, file_sha256
from echo4d.tum import nearest_timestamp, read_rows
from tum_desk_full_mesh import (
    FRAME_COUNT, GEOMETRY_SELECTED, GEOMETRY_HELDOUT, SCALE_ANCHORS,
    SCALE_DIAGNOSTICS, MAX_GAP_S, frozen_protocol,
)
from tum_desk_sfm import (
    ROOT, SOURCE_STRIDE, PINNED_BYTES, PINNED_SHA256, URL,
    get_archive, validate_archive_name,
)


def read_archive_indexes(archive: Path, out: Path) -> tuple[Path, Path, set[str]]:
    """Extract ONLY rgb.txt/depth.txt while checking archive members' names."""
    out.mkdir(parents=True, exist_ok=False)
    found: dict[str, Path] = {}
    members: set[str] = set()
    total = count = 0
    with tarfile.open(archive, "r:gz") as source:
        for item in source:
            parts = validate_archive_name(
                item.name, "dir" if item.isdir() else
                "file" if item.isfile() else "other", item.size
            )
            if item.isdir():
                continue
            count += 1
            total += item.size
            if count > 25000 or total > 3 * 1024**3:
                raise DataError("Desk archive exceeds member resource limits")
            relative = "/".join(parts[1:])
            if relative in members:
                raise DataError("Duplicate archive member identity")
            members.add(relative)
            if relative not in ("rgb.txt", "depth.txt"):
                continue
            if item.size > 16 * 1024**2:
                raise DataError("Oversized source timestamp index")
            stream = source.extractfile(item)
            if stream is None:
                raise DataError("Cannot read source timestamp index")
            data = stream.read(item.size + 1)
            if len(data) != item.size:
                raise DataError("Short timestamp index")
            path = out / relative
            path.write_bytes(data)
            found[relative] = path
    if set(found) != {"rgb.txt", "depth.txt"}:
        raise DataError("Missing original RGB/depth timestamp indexes")
    return found["rgb.txt"], found["depth.txt"], members


def audit_indexes(rgb_index: Path, depth_index: Path,
                  members: set[str] | None = None) -> dict:
    """Evaluate exactly the V1 frozen roles. No depth PNG pixels are opened."""
    rgb = read_rows(rgb_index, 2)
    depth = read_rows(depth_index, 2)
    chosen = rgb[::SOURCE_STRIDE][:FRAME_COUNT]
    if len(chosen) != FRAME_COUNT:
        raise DataError("Not enough original RGB samples for fixed 75-frame cohort")
    source = [
        {"name": f"frame_{i:05d}.png", "t_tum_rgb_s": float(t)}
        for i, (t, _) in enumerate(chosen)
    ]
    frozen = frozen_protocol(source)
    times = np.asarray([float(x[0]) for x in depth], dtype=np.float64)
    if not np.isfinite(times).all() or not len(times):
        raise DataError("Invalid source depth times")
    roles = sorted(set(GEOMETRY_SELECTED) | set(SCALE_ANCHORS) |
                   set(SCALE_DIAGNOSTICS))
    used: set[str] = set()
    rows = []
    for i in roles:
        rgb_time, rgb_path = chosen[i]
        p_rgb = PurePosixPath(rgb_path)
        if (p_rgb.is_absolute() or len(p_rgb.parts) != 2 or
                p_rgb.parts[0] != "rgb" or ".." in p_rgb.parts or
                p_rgb.suffix.lower() != ".png"):
            raise DataError("Unsafe original RGB index path")
        if members is not None and rgb_path not in members:
            raise DataError("Indexed RGB image absent from official source archive")
        t = float(rgb_time)
        at = nearest_timestamp(times, t, float("inf"))
        closest = depth[at] if at is not None else None
        candidate = closest[1] if closest else None
        delta = float(times[at] - t) if at is not None else None
        if candidate is not None:
            p_depth = PurePosixPath(candidate)
            if (p_depth.is_absolute() or len(p_depth.parts) != 2 or
                    p_depth.parts[0] != "depth" or ".." in p_depth.parts or
                    p_depth.suffix.lower() != ".png"):
                raise DataError("Unsafe original depth index path")
        within = delta is not None and abs(delta) <= MAX_GAP_S
        exists = candidate in members if members is not None else True
        reused = bool(within and candidate in used)
        if within:
            used.add(candidate)
        reasons = []
        if not within:
            reasons.append("NO_DEPTH_WITHIN_FROZEN_30MS")
        if not exists:
            reasons.append("DEPTH_INDEX_PATH_NOT_IN_ARCHIVE")
        if reused:
            reasons.append("DUPLICATE_SELECTED_DEPTH_SOURCE")
        markers = []
        if i in GEOMETRY_HELDOUT:
            markers.append("geometry_holdout")
        elif i in GEOMETRY_SELECTED:
            markers.append("geometry_train")
        if i in SCALE_ANCHORS:
            markers.append("scale_anchor")
        if i in SCALE_DIAGNOSTICS:
            markers.append("scale_diagnostic")
        rows.append({
            "index": i, "frame": source[i]["name"], "roles": markers,
            "original_rgb_path": rgb_path, "rgb_time_s": t,
            "nearest_depth_path": candidate,
            "nearest_depth_time_s": float(times[at]) if at is not None else None,
            "signed_delta_s": delta,
            "absolute_delta_ms": abs(delta) * 1000 if delta is not None else None,
            "within_frozen_30ms": within, "depth_member_exists": exists,
            "already_used_in_frozen_roles": reused,
            "accepted_for_v1": not reasons, "reasons": reasons
        })
    blocked = [x for x in rows if not x["accepted_for_v1"]]
    return {
        "version": "echo-desk-time-index-preflight-v1",
        "source": "TUM freiburg1_desk original timestamp indexes",
        "original_rgb_index_sha256": file_sha256(rgb_index),
        "original_depth_index_sha256": file_sha256(depth_index),
        "frozen_protocol_version": frozen["version"],
        "frozen_max_rgb_depth_gap_s": MAX_GAP_S,
        "selected_rgb_source_stride": SOURCE_STRIDE,
        "selected_rgb_count": FRAME_COUNT,
        "required_role_count": len(roles),
        "required_geometry_train_count": len(frozen["train_names"]),
        "required_geometry_holdout_count": len(frozen["holdout_names"]),
        "required_anchor_count": len(SCALE_ANCHORS),
        "required_diagnostic_count": len(SCALE_DIAGNOSTICS),
        "accepted_role_count": len(rows) - len(blocked),
        "failed_role_count": len(blocked),
        "status": ("PASS_FROZEN_V1_INDEX_PAIRING" if not blocked
                   else "BLOCKED_FROZEN_V1_INDEX_PAIRING"),
        "uses_depth_image_pixels": False,
        "runs_sfm_or_mesh": False,
        "changes_frozen_frame_selection_or_thresholds": False,
        "rows": rows,
        "limitations": [
            "Index-only availability, not sensor calibration or verified image overlap",
            "Does not claim physical Mesh fidelity or independently scanned geometry",
            "V1 failures cannot be fixed by quietly replacing frames or changing 30ms"
        ]
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--archive", type=Path)
    args = p.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    output = args.report_dir / "original-timestamps-preflight.json"
    receipt: dict = {
        "status": "error_before_audit",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "predeclared_archive_url": URL,
        "pinned_archive_sha256": PINNED_SHA256,
        "pinned_archive_bytes": PINNED_BYTES,
        "uses_heldout_pixels_or_gt_poses": False
    }
    try:
        archive = args.work / "source.tgz"
        get_archive(archive, receipt, args.archive.resolve() if args.archive else None)
        index_dir = args.work / "index-only"
        rgb, depth, members = read_archive_indexes(archive, index_dir)
        result = audit_indexes(rgb, depth, members)
        result["source_archive_sha256"] = file_sha256(archive)
        result["source_archive_bytes"] = archive.stat().st_size
        receipt["status"] = result["status"]
        receipt["accepted_role_count"] = result["accepted_role_count"]
        receipt["failed_role_count"] = result["failed_role_count"]
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")
        return 0 if result["failed_role_count"] == 0 else 2
    except Exception as exc:
        receipt["status"] = "ERROR_INDEX_PREFLIGHT"
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
        return 1
    finally:
        (args.report_dir / "preflight-receipt.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
