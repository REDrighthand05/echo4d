"""ECHO GPT D: verify frozen real RGB/depth sources without inventing furniture truth.

Inventory mode uses immutable ECHO evidence already in Git; archive mode re-reads
actual original TUM data and reports nearest raw motion-capture pose samples.
No geometry refinement is run and no single furniture identity is inferred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import tarfile

from echo4d.model import DataError, file_sha256

EXPECTED_HOLDOUT = [4, 5, 9, 10, 14, 19, 24, 26, 29, 34, 35, 55]
SOURCE_ROOT = "rgbd_dataset_freiburg1_xyz"


def validate_ledger(ledger: dict) -> list[dict]:
    if ledger.get("version") != "echo-hi3d-furniture-source-prereg-v1":
        raise DataError("Unsupported frozen source ledger")
    frames = ledger.get("frames")
    if not isinstance(frames, list) or len(frames) != 60:
        raise DataError("Expected 60 pre-frozen TUM records")
    roles = ledger.get("preregistered_protocol", {})
    holdout = [f"frame_{i:05d}.png" for i in EXPECTED_HOLDOUT]
    selected = [f"frame_{i:05d}.png" for i in range(61) if i != 25]
    if (roles.get("holdout_names") != holdout or
            roles.get("train_names") != [n for n in selected if n not in holdout] or
            [f.get("name") for f in frames] != selected):
        raise DataError("TUM 48/12 preregistration was changed")
    if ledger.get("object_identity", {}).get("verified_individual_furniture_count") != 0:
        raise DataError("Unreviewed furniture identity must not be claimed")
    for f in frames:
        is_holdout = f["name"] in holdout
        if (f.get("role") != ("holdout_no_model_access" if is_holdout else "train_input")
                or f.get("object_mask_status") != "not_available" or f.get("object_id") is not None):
            raise DataError("Source frames are not verified object instances")
        for key in ("original_rgb_path", "raw_depth_path"):
            p = PurePosixPath(f[key])
            if (p.is_absolute() or ".." in p.parts or len(p.parts) != 2 or
                    p.parts[0] not in ("rgb", "depth")):
                raise DataError("Unsafe or mismatched sensor path")
        for key in ("original_rgb_sha256", "raw_depth_sha256"):
            h = f.get(key)
            if (key == "raw_depth_sha256" and not is_holdout and h is None):
                continue  # historical training-depth hash is of the processed array
            if not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
                raise DataError(f"Missing original file hash: {key}")
        dt = f["raw_depth_t_s"] - f["original_rgb_t_s"]
        if (not math.isfinite(dt) or abs(dt) > .03 or
                abs(dt - f["rgb_depth_delta_s"]) > 1e-5):
            raise DataError("Source RGB/depth clock association invalid")
    return frames


def inventory(ledger_path: Path) -> dict:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    frames = validate_ledger(ledger)
    return {
        "version": "echo-hi3d-furniture-real-evidence-report-1",
        "status": "BLOCKED_NO_VERIFIED_SINGLE_FURNITURE",
        "ledger_sha256": file_sha256(ledger_path),
        "source_archive_expected_sha256": ledger["source_archive"]["sha256"],
        "photo_depth_pairs_historically_recorded": len(frames),
        "model_input_pair_count": sum(f["role"] == "train_input" for f in frames),
        "independent_scene_holdout_pair_count": sum(f["role"] == "holdout_no_model_access" for f in frames),
        "holdout_names": ledger["preregistered_protocol"]["holdout_names"],
        "individual_furniture_identity": "UNKNOWN",
        "independent_furniture_dimensions_m": None,
        "photo_depth_object_mask_association": "UNKNOWN",
        "actual_sensor_depth_available": True,
        "actual_furniture_surface_isolation": False,
        "camera_trajectory": "TUM raw groundtruth mocap timestamps; NOT automatically image-time matched",
        "estimated_camera_poses_are_measured": False,
        "real_hi3d_gpu_inference": False,
        "metric_geometry_improvement_claim": "PROHIBITED",
        "blocking_requirements": [
            "GPT C: audited single-instance furniture mesh and same-furniture photo masks",
            "Independent furniture physical dimensions or scan, not AABB of generated mesh",
            "12 fixed withheld views must have the SAME independently reviewed furniture identity and masked sensor Z",
            "Measured capture poses/registration for those source views; holdout must not tune shape or scale",
            "GPU Hi3D generated candidate with pre-holdout frozen placement and input provenance"
        ],
    }


def raw_archive_checks(ledger_path: Path, archive: Path) -> dict:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    frames = validate_ledger(ledger)
    size_expected = ledger["source_archive"]["bytes"]
    if archive.stat().st_size != size_expected or file_sha256(archive) != ledger["source_archive"]["sha256"]:
        raise DataError("TUM source archive does not match pre-registered byte count and SHA256")
    withheld = [f for f in frames if f["role"] == "holdout_no_model_access"]
    # We inspect ALL 12 source photos and depths, not select only favourable views.
    want = {}
    for f in withheld:
        want[f"{SOURCE_ROOT}/{f['original_rgb_path']}"] = (f["name"], "photo", f["original_rgb_sha256"])
        want[f"{SOURCE_ROOT}/{f['raw_depth_path']}"] = (f["name"], "depth", f["raw_depth_sha256"])
    gt_path = f"{SOURCE_ROOT}/groundtruth.txt"
    gt_expected = ledger["source_archive"]["raw_groundtruth_sha256"]
    seen = {}
    track = None
    with tarfile.open(archive, "r:gz") as t:
        for member in t:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != SOURCE_ROOT:
                raise DataError("Unsafe TUM source archive member")
            if not member.isdir() and not member.isfile():
                raise DataError("Unexpected non-regular archive entry")
            if member.name not in want and member.name != gt_path:
                continue
            if member.name in seen or member.size > 30 * 1024**2:
                raise DataError("Duplicate or oversized requested source asset")
            stream = t.extractfile(member)
            if stream is None:
                raise DataError("Missing source member bytes")
            with stream:
                data = stream.read(30 * 1024**2 + 1)
            if len(data) != member.size or len(data) > 30 * 1024**2:
                raise DataError("Truncated source asset")
            digest = hashlib.sha256(data).hexdigest()
            expected = want[member.name][2] if member.name in want else gt_expected
            if digest != expected:
                raise DataError(f"TUM source RGB/depth/trajectory SHA mismatch: {member.name}")
            seen[member.name] = digest
            if member.name == gt_path:
                track = data.decode("utf-8")
    if len(seen) != len(want) + 1 or track is None:
        raise DataError("One or more frozen source RGB, depth or raw camera trajectory files missing")
    poses = []
    for line in track.splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        parts = t.split()
        if len(parts) != 8:
            raise DataError("Invalid source mocap pose line")
        nums = [float(x) for x in parts]
        if not all(math.isfinite(x) for x in nums):
            raise DataError("Nonfinite motion-capture pose")
        poses.append(nums)
    if len(poses) < 12 or any(b[0] <= a[0] for a, b in zip(poses, poses[1:])):
        raise DataError("Unusable TUM camera groundtruth time series")
    rows = []
    for f in withheld:
        nearest = min(poses, key=lambda p: abs(p[0] - f["original_rgb_t_s"]))
        delta = nearest[0] - f["original_rgb_t_s"]
        rows.append({
            "name": f["name"], "source_rgb_t_s": f["original_rgb_t_s"],
            "raw_rgb_file_sha256": f["original_rgb_sha256"],
            "raw_depth_t_s": f["raw_depth_t_s"],
            "raw_depth_file_sha256": f["raw_depth_sha256"],
            "mocap_pose_timestamp_s": nearest[0],
            "nearest_mocap_translation_m": nearest[1:4],
            "nearest_mocap_quaternion_xyzw": nearest[4:8],
            "pose_rgb_time_offset_s": delta,
            "pose_is_interpolated_to_image_time": False,
            "pose_is_camera_calibrated_image_extrinsic": False,
            "warning": "Nearest mocap sample is NOT automatically a calibrated camera-to-world pose at RGB timestamp",
        })
    return {
        **inventory(ledger_path),
        "actual_archive_rechecked": True,
        "verified_raw_holdout_photos": len(withheld),
        "verified_raw_holdout_depths": len(withheld),
        "verified_raw_groundtruth": True,
        "camera_pose_samples": rows,
        "status": "SOURCE_BYTES_VERIFIED_BUT_FURNITURE_IDENTITY_BLOCKED",
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--archive", type=Path, help="Optional complete, pinned original TUM .tgz")
    a = p.parse_args(argv)
    try:
        if a.out.exists():
            raise DataError("Refuse to overwrite existing validation report")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        try:
            report = raw_archive_checks(a.ledger, a.archive) if a.archive else inventory(a.ledger)
        except (Exception,) as error:
            report = {"version": "echo-hi3d-furniture-real-evidence-report-1",
                      "status": "SOURCE_VERIFICATION_FAILED",
                      "error_type": type(error).__name__, "error": str(error),
                      "furniture_verified": False, "hi3d_executed": False}
            a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({"status": report["status"]}))
            return 2
        a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"status": report["status"], "report": str(a.out)}))
        return 0  # An evidenced BLOCKED decision is a successful honest research report.
    except (OSError, ValueError, KeyError, TypeError) as error:
        import sys
        print(json.dumps({"status": "rejected", "error": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
