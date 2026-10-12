"""Validate and chronologically split the Kaggle credit-card fraud dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "creditcard.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed"


def prepare(input_path: Path, output_dir: Path, reference_path: Path) -> dict[str, object]:
    """Create stable chronological 70/15/15 splits and a feature-only reference set."""
    frame = pd.read_csv(input_path)
    if "Class" not in frame.columns or "Time" not in frame.columns:
        raise ValueError("Expected the Kaggle credit-card dataset with Time and Class columns")
    if frame.empty:
        raise ValueError("Input dataset is empty")
    if frame["Class"].isna().any():
        raise ValueError("Target column Class contains null values")

    frame = frame.sort_values("Time", kind="mergesort").reset_index(drop=True)
    n = len(frame)
    # Choose approximate 70%/85% cut times, then move each cut to the start
    # of that timestamp group so identical transaction times never cross splits.
    times = frame["Time"]
    train_cut_time = times.iloc[min(int(n * 0.70), n - 1)]
    validation_cut_time = times.iloc[min(int(n * 0.85), n - 1)]
    train_end = int(times.searchsorted(train_cut_time, side="left"))
    validation_end = int(times.searchsorted(validation_cut_time, side="left"))
    validation_end = max(train_end, validation_end)
    splits = {
        "train": frame.iloc[:train_end].copy(),
        "validation": frame.iloc[train_end:validation_end].copy(),
        "test": frame.iloc[validation_end:].copy(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.to_csv(output_dir / f"{name}.csv", index=False)
    feature_columns = [col for col in frame.columns if col != "Class"]
    splits["train"][feature_columns].to_csv(reference_path, index=False)

    summary: dict[str, object] = {
        "source": str(input_path.relative_to(ROOT)) if input_path.is_relative_to(ROOT) else input_path.name,
        "shape": list(frame.shape),
        "null_cells": int(frame.isna().sum().sum()),
        "fraud_count": int(frame["Class"].sum()),
        "fraud_share": float(frame["Class"].mean()),
        "splits": {},
    }
    for name, split in splits.items():
        time_values = split["Time"]
        summary["splits"][name] = {
            "rows": len(split),
            "fraud_count": int(split["Class"].sum()),
            "fraud_share": float(split["Class"].mean()),
            "time_min": float(time_values.min()),
            "time_max": float(time_values.max()),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference", type=Path, default=ROOT / "data" / "reference.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "data" / "processed" / "summary.json")
    args = parser.parse_args()

    summary = prepare(args.input, args.output_dir, args.reference)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
