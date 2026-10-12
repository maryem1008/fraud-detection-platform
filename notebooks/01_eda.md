# Exploratory Data Analysis — Credit Card Fraud Dataset

## Source and scope

- Source: Kaggle dataset `mlg-ulb/creditcardfraud` (downloaded from the dataset's official page).
- Input file: `data/raw/creditcard.csv` (kept out of Git).
- Shape: **284,807 rows × 31 columns**.
- Features: `Time`, `Amount`, and anonymized PCA columns `V1`–`V28`; target: `Class`.
- Null cells: **0**.
- Dataset-reported license: `DbCL-1.0`; review its terms before reuse beyond this portfolio/demo context.

## Class balance

| Class | Meaning | Rows | Share |
|---:|---|---:|---:|
| 0 | Non-fraud | 284,315 | 99.82725% |
| 1 | Fraud | 492 | 0.17275% |

This severe imbalance makes accuracy a misleading primary score. Model evaluation emphasizes recall, precision-recall AUC, false-positive rate, and threshold trade-offs.

## Split method and findings

Rows are stably sorted by `Time`; approximate 70% / 15% / 15% cut points are moved to timestamp-group boundaries so transactions with identical timestamps cannot straddle partitions. Splits are **not shuffled**.

| Split | Rows | Fraud cases | Fraud share | Time range (seconds) |
|---|---:|---:|---:|---:|
| Train | 199,364 | 384 | 0.19261% | 0–132,928 |
| Validation | 42,719 | 56 | 0.13109% | 132,929–151,327 |
| Test | 42,724 | 52 | 0.12171% | 151,328–172,792 |

The fraud rate declines across the chronological windows. This is recorded as a distribution shift, not corrected through random stratification, because the forward-in-time evaluation is intentional.

## Reproduction

```bash
.venv/bin/python scripts/prepare_data.py
```

Outputs (all local and Git-ignored): `data/processed/train.csv`, `validation.csv`, `test.csv`, `summary.json`, and feature-only `data/reference.csv`.
