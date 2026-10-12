"""Tests for chronological split invariants in the data preparation step."""

from __future__ import annotations

from pathlib import Path
from runpy import run_path

import pandas as pd
import pytest

_prepare_data_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"
prepare = run_path(str(_prepare_data_path))["prepare"]


def test_prepare_keeps_timestamp_groups_in_one_split(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "Time": [timestamp for timestamp in range(10) for _ in range(2)],
            "V1": list(range(20)),
            "Class": [int(index in {2, 15}) for index in range(20)],
        }
    )
    source = tmp_path / "transactions.csv"
    output_dir = tmp_path / "processed"
    reference = tmp_path / "reference.csv"
    frame.to_csv(source, index=False)

    summary = prepare(source, output_dir, reference)

    partitions = {
        name: pd.read_csv(output_dir / f"{name}.csv")
        for name in ("train", "validation", "test")
    }
    for timestamp in frame["Time"].unique():
        matching = [
            name
            for name, partition in partitions.items()
            if timestamp in set(partition["Time"])
        ]
        assert len(matching) == 1

    assert partitions["train"]["Time"].max() < partitions["validation"]["Time"].min()
    assert partitions["validation"]["Time"].max() < partitions["test"]["Time"].min()
    assert list(pd.read_csv(reference).columns) == ["Time", "V1"]
    assert summary["shape"] == [20, 3]
    assert sum(item["rows"] for item in summary["splits"].values()) == 20


def test_prepare_rejects_missing_target_column(tmp_path) -> None:
    source = tmp_path / "invalid.csv"
    pd.DataFrame({"Time": [1.0], "V1": [0.0]}).to_csv(source, index=False)

    with pytest.raises(ValueError, match="Time and Class"):
        prepare(source, tmp_path / "processed", tmp_path / "reference.csv")
