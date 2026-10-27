"""Synthetic split-contract tests only; real TUM execution is separate."""
import copy
import pytest
from echo4d.model import DataError
from tum_composable_graph import HOLDOUT, TOTAL, split_training


def sample():
    return {"version": "echo-dataset-1", "source": "recorded", "units": "meters",
            "frames": [{"t": float(i), "depth": f"depth/{i}.npy"}
                       for i in range(TOTAL)]}


def test_exact_48_train_12_untouched_holdout():
    original = sample()
    result = split_training(original)
    assert len(result["frames"]) == 48 and len(original["frames"]) == TOTAL
    assert [int(f["t"]) for f in result["frames"]] == [
        i for i in range(TOTAL) if i not in HOLDOUT]
    assert len(HOLDOUT) == len(set(HOLDOUT)) == 12


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(source="synthetic"),
    lambda d: d.update(version="echo-1"),
    lambda d: d["frames"].pop(),
    lambda d: d["frames"].append(copy.deepcopy(d["frames"][-1])),
    lambda d: d.update(frames="bad"),
])
def test_reject_wrong_or_fabricated_source(mutate):
    data = sample()
    mutate(data)
    with pytest.raises(DataError):
        split_training(data)
