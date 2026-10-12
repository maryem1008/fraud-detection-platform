"""Compare a recent feature window with the training reference using Evidently."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "data" / "reference.csv"
DEFAULT_CURRENT = ROOT / "data" / "processed" / "test.csv"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts"


def drift_snapshot(reference_path: Path, current_path: Path, window_rows: int) -> tuple[Any, dict[str, Any]]:
    """Compute a DataDriftPreset report and a stable summary signal."""
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    if "Class" in current.columns:
        current = current.drop(columns=["Class"])
    if reference.empty or current.empty:
        raise ValueError("Reference and current datasets must both contain rows")
    if window_rows <= 0:
        raise ValueError("window_rows must be positive")

    missing = sorted(set(reference.columns) - set(current.columns))
    extra = sorted(set(current.columns) - set(reference.columns))
    if missing or extra:
        raise ValueError(f"Feature schema mismatch; missing={missing}, extra={extra}")
    current = current[reference.columns]
    # The default current file is chronologically sorted. Keep its most recent tail.
    current = current.tail(window_rows).reset_index(drop=True)
    reference = reference.reset_index(drop=True)

    snapshot = Report([DataDriftPreset()]).run(
        current_data=current,
        reference_data=reference,
    )
    data = snapshot.dict()
    aggregate = next(
        (
            metric
            for metric in data.get("metrics", [])
            if "DriftedColumnsCount" in metric.get("metric_name", "")
            and isinstance(metric.get("value"), dict)
        ),
        None,
    )
    if aggregate is None:
        raise RuntimeError("Evidently report did not return a DriftedColumnsCount aggregate metric")
    value = aggregate["value"]
    drift_share = float(value["share"])
    drifted_count = int(value["count"])
    summary = {
        "drift_score": drift_share,
        "drift_threshold": None,
        "drift_detected": None,
        "drifted_columns": drifted_count,
        "total_columns": len(reference.columns),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "reference_file": reference_path.name,
        "current_file": current_path.name,
        "current_window": "most recent rows in the supplied chronological file; held-out test tail is a production-window proxy until live data exists",
    }
    return snapshot, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--window-rows", type=int, default=5000)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--github-output", action="store_true", help="Write drift_detected/drift_score to GITHUB_OUTPUT")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    snapshot, summary = drift_snapshot(args.reference, args.current, args.window_rows)
    summary["drift_threshold"] = args.threshold
    summary["drift_detected"] = bool(summary["drift_score"] >= args.threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "drift_report.html"
    signal_path = args.output_dir / "drift_signal.json"
    snapshot.save_html(str(report_path))
    signal_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.github_output:
        output_path = os.getenv("GITHUB_OUTPUT")
        if not output_path:
            raise RuntimeError("--github-output requires the GITHUB_OUTPUT environment variable")
        with open(output_path, "a", encoding="utf-8") as output:
            output.write(f"drift_detected={str(summary['drift_detected']).lower()}\n")
            output.write(f"drift_score={summary['drift_score']:.8f}\n")
    print(json.dumps(summary, indent=2))
    print(f"HTML report: {report_path}")
    print(f"Machine-readable signal: {signal_path}")


if __name__ == "__main__":
    main()
