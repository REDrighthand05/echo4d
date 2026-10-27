"""A/B baseline gates: never fabricate a second real reconstruction."""
import json
from pathlib import Path

import pytest

from echo4d.model import DataError
from tum_graph_projective_ab import check_baseline, read_json


def test_bad_json_receipt_rejected(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text("[]")
    with pytest.raises(DataError, match="object receipt"):
        read_json(path)


@pytest.mark.parametrize("kind", ["missing", "wrong_source", "fake_holdout"])
def test_missing_or_unverifiable_baseline_cannot_be_compared(tmp_path, kind):
    work = tmp_path / "work"
    baseline = tmp_path / "baseline"
    work.mkdir()
    baseline.mkdir()
    if kind == "wrong_source":
        (baseline / "report.json").write_text(json.dumps({
            "status": "executed_unscored", "source": {"archive_sha256": "0" * 64}}))
    if kind == "fake_holdout":
        (baseline / "report.json").write_text("{}")
        (baseline / "graph-execution.json").write_text("{}")
        (baseline / "heldout-view-consistency.json").write_text(json.dumps({
            "status": "diagnostic_only", "diagnostic": {"heldout_frames": 12}}))
    with pytest.raises((DataError, FileNotFoundError, OSError)):
        check_baseline(work, baseline)
