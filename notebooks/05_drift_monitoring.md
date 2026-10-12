# Evidently Drift Monitoring

## Signal definition

`scripts/check_drift.py` compares feature distributions in `data/reference.csv` (the training-window reference) against the most recent rows of a supplied current dataset. `Class` is excluded if present, and feature schemas must match exactly. Evidently's `DataDriftPreset` reports the **share of feature columns flagged as drifted**; the default trigger is `drift_score >= 0.30`. The threshold is configurable with `--threshold`.

The script saves `artifacts/drift_report.html` and `artifacts/drift_signal.json`. With `--github-output`, it writes `drift_detected=true|false` and `drift_score` to the GitHub Actions `GITHUB_OUTPUT` file for downstream conditional jobs.

## Local check

```bash
.venv/bin/python scripts/check_drift.py \
  --threshold 0.3 --window-rows 5000 --github-output
```

On this run, Evidently flagged **23/30 columns** in the last 5,000 rows of the chronologically held-out test file: drift share **0.7667**, threshold **0.30**, `drift_detected=true`. This is a **historical test-tail proxy**, not live or production traffic. In deployment, pass a recent production feature window with the same schema; do not interpret this offline signal as evidence of current production drift.

The HTML report is a build artifact and is not committed into the source repository. The JSON output records row counts, filenames, score, threshold, and signal without embedding transaction rows.
