#!/usr/bin/env python3
"""Enumerate a finite planning board, NEVER run reconstruction or claim quality.

Standard library only. The frozen JSON catalog is independent of the draft
echo4d strategy CLI's ten legacy route labels. Output is deterministic JSONL.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs/strategy/chessboard-v1.json"
DOCUMENT = ROOT / "docs/STRATEGY_CHESSBOARD.md"
BEGIN = "<!-- generated-routes:begin -->"
END = "<!-- generated-routes:end -->"


def routes(catalog):
    """Every nonempty subset of geometry providers, with each pose and prior."""
    sources = catalog["geometry_sources"]
    for pose in catalog["poses"]:
        for count in range(1, len(sources) + 1):
            for geometry in itertools.combinations(sources, count):
                for prior in catalog["priors"]:
                    modules = catalog["common_modules"] + [pose["module"]]
                    modules += [g["module"] for g in geometry]
                    if len(geometry) > 1:
                        modules += ["M09"]
                    if prior["id"] == "K1":
                        modules += ["M10"]
                    groups = list(catalog["common_requires"]) + pose["requires"]
                    groups += [g for source in geometry for g in source["requires"]]
                    groups += prior["requires"]
                    groups = [list(g) for g in dict.fromkeys(tuple(g) for g in groups)]
                    source_ids = "".join(g["id"] for g in geometry)
                    yield {
                        "route_id": f'{pose["id"]}-{source_ids}-{prior["id"]}',
                        "pose": pose["id"],
                        "geometry": [g["id"] for g in geometry],
                        "prior": prior["id"],
                        "label": pose["label"] + " + " + "/".join(g["label"] for g in geometry)
                                 + " + " + prior["label"],
                        "modules": sorted(set(modules)),
                        "requires_all_groups": groups,
                        "development_state": "planned_composition",
                        "execution_state": "not_run",
                        "visual_recognizable": "not_evaluated",
                        "depth_screening": "not_evaluated",
                        "independent_geometry": "not_evaluated",
                        "overall_accepted": False,
                        "runtime_checks_required": [
                            "source_identity_and_permission", "calibration_and_sync",
                            "frame_count_overlap_and_observability", "units_and_world_alignment",
                            "provider_versions_and_interfaces", "budget_and_quality_gates"
                        ],
                    }


def case(catalog, route, mask):
    names = [entry["id"] for entry in catalog["input_catalog"]]
    available = {name for bit, name in enumerate(names) if mask & (1 << bit)}
    missing = [g for g in route["requires_all_groups"] if not available.intersection(g)]
    return {
        "board_version": catalog["version"],
        "case_id": f'{route["route_id"]}@{mask:04x}',
        "route_id": route["route_id"],
        "input_mask": mask,
        "declared_inputs": [name for name in names if name in available],
        "input_match": "missing_inputs" if missing else "declaration_candidate",
        "missing_required_groups": missing,
        "execution_state": "not_run",
        "overall_accepted": False,
    }


def markdown(rows):
    lines = [
        BEGIN, "",
        "| 路线ID | 位姿 + 几何来源 + 旧Mesh开关 | 共享模块组合 | 本组合状态 |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append("| " + " | ".join([
            row["route_id"], row["label"], " ".join(row["modules"]),
            "待构建/未执行，未验收"
        ]) + " |")
    lines += ["", END]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input-mask", type=int, help="Declared input presence mask, decimal.")
    group.add_argument("--all-inputs", action="store_true", help="Stream ALL input x route cases.")
    group.add_argument("--write-docs", action="store_true", help="Refresh the marked route table.")
    group.add_argument("--check-docs", action="store_true", help="Verify catalog and marked table agree.")
    args = parser.parse_args(argv)
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    rows = list(routes(catalog))
    if args.write_docs or args.check_docs:
        text = DOCUMENT.read_text(encoding="utf-8")
        before, rest = text.split(BEGIN, 1)
        _, after = rest.split(END, 1)
        expected = before + markdown(rows) + after
        if args.write_docs:
            DOCUMENT.write_text(expected, encoding="utf-8")
        elif text != expected:
            parser.error("route table differs from catalog; run --write-docs")
        print(f"route table: {len(rows)} rows verified")
        return 0
    limit = 1 << len(catalog["input_catalog"])
    if args.input_mask is not None and not 0 <= args.input_mask < limit:
        parser.error(f"input mask must be in [0, {limit - 1}]")
    if args.all_inputs:
        # Deliberately streamed: no million-row list or sensor data in memory.
        for mask in range(limit):
            for route in rows:
                print(json.dumps(case(catalog, route, mask), ensure_ascii=False, sort_keys=True))
    elif args.input_mask is not None:
        for route in rows:
            print(json.dumps(case(catalog, route, args.input_mask), ensure_ascii=False, sort_keys=True))
    else:
        for route in rows:
            print(json.dumps(route, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
