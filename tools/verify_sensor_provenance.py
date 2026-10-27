"""Fail-closed audit of SfM input RGB -> measured-depth B-path recording.

This does not certify original TUM camera files, sensor-depth provenance or
visual scene reconstruction. It only checks the specified SfM input files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from echo4d.metric_rgbd import verify_sensor_rgb_chain
from echo4d.model import DataError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--sfm-images", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(verify_sensor_rgb_chain(args.dataset, args.sfm_images),
                         ensure_ascii=False))
        return 0
    except (DataError, OSError, ValueError, KeyError, TypeError) as error:
        import sys
        print(json.dumps({"status": "BLOCKED", "error": str(error)},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
