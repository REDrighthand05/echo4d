"""Import real first-person video + optional timed sensor streams for later ECHO reconstruction.

This stores synchronized *claims* and original measurements, not inferred camera poses
or surfaces. Only a real depth+calibrated pose adapter can enter the metric TSDF.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import shutil

from .model import DataError, file_sha256, finite_number
from .output import output_transaction, write_json

CAPTURE_VERSION = "echo-history-stream-0.1"
TIMELINE_VERSION = "echo-history-timeline-1"
AUX_VERSION = "echo-history-aux-0.1"
TIMEBASE = "seconds_since_recording_start_monotonic"
AUX_KINDS = ("measured_depth", "scan", "near_field_observation", "other")


def _read_json(path: Path, max_bytes: int = 8 * 1024 * 1024) -> dict:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > max_bytes:
        raise DataError(f"Missing, symlinked, or oversized JSON: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise DataError(f"Invalid JSON: {path.name}") from error
    if not isinstance(payload, dict):
        raise DataError("Expected a JSON object")
    return payload


def _safe_basename(value: object) -> str:
    if (not isinstance(value, str) or not 1 <= len(value) <= 160
            or value in (".", "..") or "/" in value or "\\" in value
            or Path(value).name != value or any(ord(ch) < 32 for ch in value)):
        raise DataError("Media filename must be a simple relative filename")
    return value


def _samples(items: object, duration: float, *, imu: bool) -> list[dict]:
    if not isinstance(items, list) or len(items) > 100000:
        raise DataError("Sensor samples must be a bounded list")
    previous = -1.0
    for sample in items:
        if not isinstance(sample, dict):
            raise DataError("Every sensor sample must be an object")
        t = finite_number(sample.get("t_s"), "sensor t_s")
        if t < 0 or t <= previous or t > duration + 1:
            raise DataError("Sensor timestamps must increase within recording duration (+1s tolerance)")
        previous = t
        if imu:
            for key in ("acceleration_including_gravity_m_s2", "acceleration_without_gravity_m_s2"):
                v = sample.get(key)
                if v is not None and (not isinstance(v, list) or len(v) != 3
                                      or any(type(a) not in (float, int) or not math.isfinite(a) for a in v)):
                    raise DataError(f"Invalid IMU vector: {key}")
            rate = sample.get("rotation_rate_deg_s")
            if rate is not None:
                if not isinstance(rate, dict) or any(
                        rate.get(axis) is not None and
                        (type(rate[axis]) not in (int, float) or not math.isfinite(rate[axis]))
                        for axis in ("alpha", "beta", "gamma")):
                    raise DataError("Invalid IMU angular rate")
        else:
            value = sample.get("values")
            if (not isinstance(value, list) or not 1 <= len(value) <= 4096
                    or any(type(a) not in (int, float) or not math.isfinite(a) for a in value)):
                raise DataError("Auxiliary sensor values must be finite numeric arrays")
    return items


def _validate_capture(capture: dict, media: Path) -> tuple[float, dict, dict]:
    if capture.get("version") != CAPTURE_VERSION or capture.get("time_base") != TIMEBASE:
        raise DataError("Unsupported capture format or recording time base")
    started = capture.get("created_utc")
    try:
        stamp = datetime.fromisoformat(started.replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError) as error:
        raise DataError("Missing ISO capture start timestamp") from error
    if stamp.utcoffset() is None:
        raise DataError("Capture start needs an explicit timezone")
    duration = finite_number(capture.get("duration_s"), "duration_s")
    if not 0 < duration <= 86400:
        raise DataError("Capture duration outside (0, 86400] seconds")
    meta = capture.get("media")
    if not isinstance(meta, dict) or _safe_basename(meta.get("file")) != media.name:
        raise DataError("Selected media filename does not match the capture")
    if not media.is_file() or media.is_symlink():
        raise DataError("Media must be an ordinary local file")
    size = meta.get("bytes")
    if type(size) is not int or size <= 0 or media.stat().st_size != size:
        raise DataError("Media byte count mismatch or empty recording")
    mime = meta.get("mime_type")
    if not isinstance(mime, str) or not re.fullmatch(r"video/(webm|mp4)(;[a-zA-Z0-9=,.-]+)?", mime):
        raise DataError("Only the browser-recorded WebM/MP4 media type is supported")
    imu = capture.get("imu")
    if not isinstance(imu, dict) or imu.get("calibrated") is not False or imu.get("actual_head_pose") != "unknown":
        raise DataError("Browser IMU must remain explicitly uncalibrated")
    _samples(imu.get("samples"), duration, imu=True)
    if capture.get("auxiliary_sensor_observations") != []:
        raise DataError("Embedded auxiliary data is not supported; use explicit --sensor files")
    if capture.get("geometry_status") != "not_reconstructed":
        raise DataError("Browser capture cannot assert previously reconstructed geometry")
    return duration, meta, imu


def _auxiliary(path: Path, duration: float) -> dict:
    data = _read_json(path)
    ident, kind = data.get("sensor_id"), data.get("kind")
    if (data.get("version") != AUX_VERSION or not isinstance(ident, str)
            or not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", ident)
            or kind not in AUX_KINDS):
        raise DataError("Auxiliary sensor needs a valid ID, version and kind")
    basis = data.get("time_base")
    if basis not in (TIMEBASE, "device_clock_unaligned"):
        raise DataError("Auxiliary stream must declare its time base")
    unit = data.get("units")
    if not isinstance(unit, str) or not 1 <= len(unit) <= 48:
        raise DataError("Auxiliary stream needs source units")
    if data.get("spatial_calibration") not in ("unknown", "externally_supplied"):
        raise DataError("Auxiliary stream must state whether extrinsics are available")
    samples = _samples(data.get("samples"), duration if basis == TIMEBASE else 86400, imu=False)
    if not samples:
        raise DataError("Empty auxiliary stream")
    return data


def import_history(
    capture_path: Path, media_path: Path, output: Path, sensor_paths: tuple[Path, ...] = (),
) -> dict:
    """Make a self-contained time-indexed archive; never infer geometry or motion."""
    capture_path, media_path, output = Path(capture_path), Path(media_path), Path(output)
    capture = _read_json(capture_path)
    duration, meta, imu = _validate_capture(capture, media_path)
    sensors: list[tuple[Path, dict]] = []
    seen_ids = set()
    for path in sensor_paths:
        path = Path(path)
        data = _auxiliary(path, duration)
        ident = data["sensor_id"]
        if ident in seen_ids:
            raise DataError("Repeated auxiliary sensor ID")
        seen_ids.add(ident)
        sensors.append((path, data))
    if output.exists():
        raise DataError("Output already exists")
    with output_transaction(output) as folder:
        (folder / "media").mkdir()
        (folder / "sensors").mkdir()
        dest_media = folder / "media" / media_path.name
        shutil.copyfile(media_path, dest_media)
        digest = file_sha256(dest_media)
        if dest_media.stat().st_size != meta["bytes"]:
            raise DataError("Media changed during import")
        write_json(folder / "capture-source.json", capture)
        exported_sensors = []
        for source, data in sensors:
            name = data["sensor_id"] + ".json"
            write_json(folder / "sensors" / name, data)
            basis = data["time_base"]
            exported_sensors.append({
                "sensor_id": data["sensor_id"], "kind": data["kind"],
                "time_base": basis, "units": data["units"],
                "spatial_calibration": data["spatial_calibration"],
                "clock_status": ("source_claim_same_clock_unverified"
                                 if basis == TIMEBASE else "unaligned"),
                "file": "sensors/" + name,
                "source_sha256": file_sha256(source),
                "samples": len(data["samples"]),
                "first_t_s": data["samples"][0]["t_s"],
                "last_t_s": data["samples"][-1]["t_s"],
            })
        timeline = {
            "version": TIMELINE_VERSION, "started_utc": capture["created_utc"],
            "duration_s": duration, "time_base": TIMEBASE,
            "media": {
                "file": media_path.name, "archive_relative_path": "media/" + media_path.name,
                "mime_type": meta["mime_type"], "bytes": meta["bytes"],
                "sha256": digest, "camera_pose": "unknown",
                "role": "recorded_first_person_view",
            },
            "imu": {
                "samples": imu["samples"], "calibrated": False,
                "clock_status": "browser_event_relative_unverified",
                "head_pose": "unknown",
                "acceleration_unit": "m/s^2", "rotation_rate_unit": "degrees/s",
            },
            "auxiliary_streams": exported_sensors,
            "geometry_status": "not_reconstructed",
            "notes": "Observed media+uncalibrated motion; no mesh or unknown geometry is invented.",
        }
        write_json(folder / "history.timeline.json", timeline)
    return {
        "output": str(output.resolve()), "version": TIMELINE_VERSION,
        "media_sha256": digest, "imu_samples": len(imu["samples"]),
        "auxiliary_streams": len(exported_sensors), "reconstruction": "not_attempted",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Adapt captured first-person media + IMU to ECHO history timeline")
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--sensor", type=Path, action="append", default=[],
                        help="Optional echo-history-aux-0.1 JSON stream; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = import_history(args.capture, args.media, args.output, tuple(args.sensor))
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (DataError, OSError, ValueError) as error:
        parser.exit(2, f"history import failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
