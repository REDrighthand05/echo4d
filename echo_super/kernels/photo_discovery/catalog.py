"""Catalogue loading for photo_discovery.

The catalogue is DATA, not code: a JSON object keyed by the canonical
(English) label, each entry carrying

    zh        Chinese display name
    queries   open-vocabulary query strings (English first — round 1
              found the Chinese terms trigger noticeably more duplicate
              boxes and false alarms; they stay as auxiliaries)
    typical_m [w, h, d] common-sense size in metres (prior only; this
              job class carries no absolute scale, sizes are judged
              relative inside the cloud)
    pitfalls  known traps for this entry, written by whoever verified
              it — read before trusting the output row

Entries can be added/removed by editing catalog.json (or by passing
another catalogue to the kernel); nothing in the pipeline hard-codes
the six pioneer entries.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent / "catalog.json"


def load_catalog(source=None) -> dict:
    """source: None -> bundled catalog.json; a path; a JSON string; or
    an already-parsed dict. Returns {label: entry} with defaults filled."""
    if source is None:
        raw = json.loads(DEFAULT_CATALOG_PATH.read_text(encoding="utf-8"))
    elif isinstance(source, dict):
        raw = source
    elif isinstance(source, (str, Path)) and Path(source).exists():
        raw = json.loads(Path(source).read_text(encoding="utf-8"))
    elif isinstance(source, str):
        raw = json.loads(source)
    else:
        raise ValueError(f"cannot load catalogue from {type(source).__name__}")
    out = {}
    for label, e in raw.items():
        out[label] = {
            "zh": e.get("zh", label),
            "queries": list(e.get("queries", [label])),
            "typical_m": e.get("typical_m"),
            "pitfalls": e.get("pitfalls", ""),
            "status_204141": e.get("status_204141", ""),
        }
    return out
