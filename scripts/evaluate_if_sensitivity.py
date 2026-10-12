"""Evaluate validation-tuned Isolation Forest false-positive budgets."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from xgboost import XGBClassifier

from fraud_detection.training import binary_metrics, choose_fbeta_threshold

ROOT = Path(__file__).resolve().parents[1]

BUDGETS = (0.0005, 0.001, 0.0025, 0.005)


def main() -> None:
    manifest = json.loads((ROOT / "models" / "feature_schema.json").read_text())
    saved_thresholds = json.loads((ROOT / "models" / "thresholds.json").read_text())
    features = manifest["feature_columns"]
    validation = pd.read_csv(ROOT / "data" / "processed" / "validation.csv")
    test = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
    xgb = XGBClassifier()
    xgb.load_model(ROOT / "models" / "xgboost_model.json")
    isolation = joblib.load(ROOT / "models" / "isolation_forest.joblib")
    scaler = joblib.load(ROOT / "models" / "isolation_scaler.joblib")

    y_val = validation["Class"].to_numpy(dtype=int)
    y_test = test["Class"].to_numpy(dtype=int)
    xgb_val = xgb.predict_proba(validation[features])[:, 1]
    xgb_test = xgb.predict_proba(test[features])[:, 1]
    if_val = -isolation.decision_function(scaler.transform(validation[features]))
    if_test = -isolation.decision_function(scaler.transform(test[features]))
    xgb_threshold = float(saved_thresholds["xgb_probability"])
    baseline_test_pred = xgb_test >= xgb_threshold
    baseline_test = binary_metrics(y_test, baseline_test_pred, xgb_test)

    results: dict[str, Any] = {
        "method": "F2 threshold selected on validation under each stated IF FPR cap; thresholds then frozen for test",
        "xgb_threshold": xgb_threshold,
        "xgb_baseline_test": baseline_test,
        "budgets": {},
    }
    for budget in BUDGETS:
        threshold = choose_fbeta_threshold(y_val, if_val, max_fpr=budget)
        val_if_pred = if_val >= threshold
        val_ensemble_pred = (xgb_val >= xgb_threshold) | val_if_pred
        test_if_pred = if_test >= threshold
        test_ensemble_pred = (xgb_test >= xgb_threshold) | test_if_pred
        results["budgets"][str(budget)] = {
            "if_threshold": threshold,
            "validation_isolation_forest": binary_metrics(y_val, val_if_pred, if_val),
            "validation_ensemble": binary_metrics(y_val, val_ensemble_pred, xgb_val),
            "test_isolation_forest": binary_metrics(y_test, test_if_pred, if_test),
            "test_ensemble": binary_metrics(y_test, test_ensemble_pred, xgb_test),
        }

    output = ROOT / "artifacts" / "isolation_fpr_sensitivity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fraud-detection-platform")
    with mlflow.start_run(run_name="isolation-fpr-sensitivity"):
        mlflow.log_param("threshold_selection", results["method"])
        mlflow.log_param("xgb_threshold_held_fixed", xgb_threshold)
        for budget, metrics in results["budgets"].items():
            prefix = f"if_budget_{budget.replace('.', '_')}"
            mlflow.log_param(f"{prefix}_threshold", metrics["if_threshold"])
            for split in ("validation_ensemble", "test_ensemble"):
                for name, value in metrics[split].items():
                    mlflow.log_metric(f"{prefix}_{split}_{name}", value)
        mlflow.log_artifact(str(output), artifact_path="sensitivity")
        results["mlflow_run_id"] = mlflow.active_run().info.run_id
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    for budget, metrics in results["budgets"].items():
        val_if = metrics["validation_isolation_forest"]
        val_ens = metrics["validation_ensemble"]
        test_if = metrics["test_isolation_forest"]
        test_ens = metrics["test_ensemble"]
        print(
            f"budget={budget}: threshold={metrics['if_threshold']:.8f}; "
            f"val IF TP/FP={val_if['true_positives']:.0f}/{val_if['false_positives']:.0f}; "
            f"val ensemble recall/FPR={val_ens['recall']:.4f}/{val_ens['false_positive_rate']:.6f}; "
            f"test IF TP/FP={test_if['true_positives']:.0f}/{test_if['false_positives']:.0f}; "
            f"test ensemble TP/FP={test_ens['true_positives']:.0f}/{test_ens['false_positives']:.0f}; "
            f"test recall/precision={test_ens['recall']:.4f}/{test_ens['precision']:.4f}"
        )
    print(f"Sensitivity run: {results['mlflow_run_id']}")


if __name__ == "__main__":
    main()
