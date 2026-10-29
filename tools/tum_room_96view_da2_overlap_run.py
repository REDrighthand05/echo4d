"""Run #104's real 96-view RGB-depth ladder with overlap-aware witnesses.

No new fusion engine, planner or frontend. Sensor depth remains unavailable to
the 0-sensor candidate until after its Mesh is fused. Historical FAIL retained.
"""
from __future__ import annotations

import tum_room_96view_da2_complete as base
from tum_room_pose_overlap_neighbors import choose_overlap_neighbors

_original_build = base.build_complete_rgb_prediction


def build_with_correct_receipts(*args, **kwargs):
    manifest, report = _original_build(*args, **kwargs)
    receipts = report["receipts"]
    report["new_cpu_inference_frames"] = sum(
        r["status"] not in ("frozen_DA2_RGB_prediction",
                            "no_two_independent_RGB_neighbors")
        for r in receipts)
    report["new_prediction_slots"] = sum(
        r["status"] != "frozen_DA2_RGB_prediction" for r in receipts)
    report["skipped_no_two_neighbors"] = sum(
        r["status"] == "no_two_independent_RGB_neighbors" for r in receipts)
    report["neighbor_policy"] = "projected-frustum-overlap-v1; no sensor depth"
    return manifest, report


def main():
    base.choose_neighbors = choose_overlap_neighbors
    base.build_complete_rgb_prediction = build_with_correct_receipts
    base.main()


if __name__ == "__main__":
    main()
