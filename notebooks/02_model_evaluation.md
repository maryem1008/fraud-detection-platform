# Model Training and Evaluation

## Training design

- XGBoost binary classifier trained on the chronological training split with `scale_pos_weight = 518.1771`, PR-AUC early stopping, and 146 best boosting iterations.
- Isolation Forest trained on standardized features from the same training window.
- Both thresholds tuned only on the chronological validation split using F2, with a maximum per-component validation false-positive-rate budget of **0.05%**. The resulting ensemble validation FPR is below **0.10%**.
- The final chronological test split was used once for evaluation after both thresholds were locked.
- MLflow experiment: `fraud-detection-platform`, local SQLite tracking store. Initial exploratory run and final constrained run are both recorded.

## Final held-out test results

| Model | Precision | Recall | F1 | PR-AUC (average precision) | False-positive rate | False positives | True positives |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost component | 84.44% | 73.08% | 0.7835 | 0.7666 | 0.0164% | 7 | 38 |
| Isolation Forest component | 0.00% | 0.00% | 0.0000 | 0.0461 | 0.0023% | 1 | 0 |
| OR ensemble | 84.44% | 73.08% | 0.7835 | 0.7666 | 0.0164% | 7 | 38 |

The validation-tuned Isolation Forest threshold adds no true positives on the held-out test window. A first, unconstrained threshold experiment added hundreds of false positives for limited recall gain, so the final operating rule uses the declared validation FPR budget. The Isolation Forest artifact and score remain in the ensemble for auditability and future re-evaluation, but it does **not** improve this test result at the conservative threshold. This is reported rather than hidden.

## Validation metrics

The constrained OR ensemble achieved 95.65% precision, 78.57% recall, F1 0.8627, PR-AUC 0.8573, and 2 false positives (44 true positives) on validation.

## Wider Isolation Forest budget sensitivity

The threshold sweep `scripts/evaluate_if_sensitivity.py` selected each Isolation Forest threshold on validation only, then froze it before test scoring. At a **10× wider 0.5% component validation FPR budget**, the standalone IF found 7 validation fraud cases (207 false positives) and 4 test fraud cases (106 false positives). Every one of those fraud cases was already flagged by XGBoost, so the OR ensemble gained **zero recall**: validation recall stayed 78.57% and test recall stayed 73.08%. Meanwhile, combined false positives rose from 2 to 209 on validation and from 7 to 112 on test; test precision fell from 84.44% to 25.33%.

Thus, widening the IF budget is **not a worthwhile trade-off** for this dataset and trained model. The operational description should be **XGBoost-primary fraud detection with a separately tracked, auditable Isolation Forest anomaly-scoring component**, not a value-adding ensemble. The conservative default thresholds are retained. Sensitivity MLflow run: `54434866cab84d7aae2586e4e9c4ac8d`.

## Reproducibility and artifacts

```bash
.venv/bin/python scripts/train.py
.venv/bin/python scripts/evaluate_if_sensitivity.py
```

Saved local artifacts: `models/xgboost_model.json`, `models/isolation_forest.joblib`, `models/isolation_scaler.joblib`, `models/thresholds.json`, and `models/feature_schema.json`. The final MLflow run ID is `6bc5a1a768e84464ac005489f018844e`.

These figures are offline measurements on the provided Kaggle benchmark and do not establish production performance or business impact. The false-positive and deployment-cycle baseline comparison belongs to section M and has not yet been measured.
