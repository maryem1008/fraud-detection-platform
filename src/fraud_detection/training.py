"""Training and evaluation helpers for the fraud detection ensemble."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "Class"
MAX_ENSEMBLE_VALIDATION_FPR = 0.001
MAX_COMPONENT_VALIDATION_FPR = MAX_ENSEMBLE_VALIDATION_FPR / 2

XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 2,
    "reg_lambda": 5.0,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "n_jobs": 2,
    "random_state": 42,
    "early_stopping_rounds": 40,
}
IF_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_samples": "auto",
    "contamination": "auto",
    "max_features": 1.0,
    "bootstrap": False,
    "n_jobs": 2,
    "random_state": 42,
}


def _read_split(name: str) -> pd.DataFrame:
    path = ROOT / "data" / "processed" / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run scripts/prepare_data.py first")
    frame = pd.read_csv(path)
    if TARGET not in frame:
        raise ValueError(f"{path} does not contain target column {TARGET}")
    return frame


def choose_fbeta_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    beta: float = 2.0,
    max_fpr: float | None = None,
) -> float:
    """Maximize F-beta, optionally subject to a validation false-positive-rate cap."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return 0.5
    precision = precision[:-1]
    recall = recall[:-1]
    beta_sq = beta**2
    fbeta = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + 1e-15)
    eligible = np.arange(thresholds.size)
    if max_fpr is not None:
        negative_scores = np.sort(np.asarray(scores)[np.asarray(y_true) == 0])
        false_positives = negative_scores.size - np.searchsorted(
            negative_scores, thresholds, side="left"
        )
        fpr = false_positives / max(negative_scores.size, 1)
        eligible = np.flatnonzero(fpr <= max_fpr)
        if eligible.size == 0:
            return float(np.nextafter(np.max(scores), np.inf))
    best_value = np.nanmax(fbeta[eligible])
    best = eligible[np.isclose(fbeta[eligible], best_value, rtol=1e-10, atol=1e-12)]
    # Among equal F-beta candidates, prefer precision, then the higher threshold.
    best = max(best.tolist(), key=lambda i: (precision[i], thresholds[i]))
    if fbeta[best] <= 0:
        return float(np.nextafter(np.max(scores), np.inf))
    return float(thresholds[best])


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    negatives = int((y_true == 0).sum())
    false_positives = int(((y_true == 0) & (y_pred == 1)).sum())
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "false_positive_rate": float(false_positives / negatives) if negatives else 0.0,
        "false_positives": float(false_positives),
        "true_positives": float(((y_true == 1) & (y_pred == 1)).sum()),
    }


def _log_metrics(prefix: str, metrics: dict[str, float]) -> None:
    mlflow.log_metrics({f"{prefix}_{key}": value for key, value in metrics.items()})


def _safe_tracking_uri(uri: str) -> str:
    """Return a display-safe tracking URI without userinfo, query, or fragments."""
    parsed = urlsplit(uri)
    if parsed.scheme == "sqlite":
        return "sqlite:///mlflow.db"
    if parsed.scheme in {"http", "https"} and parsed.hostname:
        host = parsed.hostname
        if parsed.port:
            host = f"{host}:{parsed.port}"
        return f"{parsed.scheme}://{host}{parsed.path}"
    return parsed.scheme or "configured"


def train_and_evaluate() -> dict[str, Any]:
    train = _read_split("train")
    validation = _read_split("validation")
    test = _read_split("test")
    features = [column for column in train.columns if column != TARGET]
    if features != [column for column in validation.columns if column != TARGET]:
        raise ValueError("Validation feature schema does not match training schema")
    if features != [column for column in test.columns if column != TARGET]:
        raise ValueError("Test feature schema does not match training schema")

    X_train = train[features]
    y_train = train[TARGET].astype(int).to_numpy()
    X_val = validation[features]
    y_val = validation[TARGET].astype(int).to_numpy()
    X_test = test[features]
    y_test = test[TARGET].astype(int).to_numpy()
    if y_train.sum() == 0 or y_val.sum() == 0 or y_test.sum() == 0:
        raise ValueError("Every chronological split must contain at least one fraud example")
    scale_pos_weight = float((y_train == 0).sum() / y_train.sum())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fraud-detection-platform")

    xgb_model = XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_val_probability = xgb_model.predict_proba(X_val)[:, 1]
    xgb_threshold = choose_fbeta_threshold(
        y_val, xgb_val_probability, max_fpr=MAX_COMPONENT_VALIDATION_FPR
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    isolation_model = IsolationForest(**IF_PARAMS)
    isolation_model.fit(X_train_scaled)
    if_val_score = -isolation_model.decision_function(X_val_scaled)
    if_threshold = choose_fbeta_threshold(
        y_val, if_val_score, max_fpr=MAX_COMPONENT_VALIDATION_FPR
    )

    val_combined_prediction = (xgb_val_probability >= xgb_threshold) | (if_val_score >= if_threshold)
    val_combined_metrics = binary_metrics(y_val, val_combined_prediction, xgb_val_probability)
    val_xgb_metrics = binary_metrics(y_val, xgb_val_probability >= xgb_threshold, xgb_val_probability)
    val_if_metrics = binary_metrics(y_val, if_val_score >= if_threshold, if_val_score)

    # Thresholds are locked before scoring the final chronological test window.
    xgb_test_probability = xgb_model.predict_proba(X_test)[:, 1]
    if_test_score = -isolation_model.decision_function(X_test_scaled)
    test_combined_prediction = (xgb_test_probability >= xgb_threshold) | (if_test_score >= if_threshold)
    test_combined_metrics = binary_metrics(y_test, test_combined_prediction, xgb_test_probability)
    test_xgb_metrics = binary_metrics(y_test, xgb_test_probability >= xgb_threshold, xgb_test_probability)
    test_if_metrics = binary_metrics(y_test, if_test_score >= if_threshold, if_test_score)

    thresholds = {
        "xgb_probability": xgb_threshold,
        "isolation_anomaly_score": if_threshold,
        "tuning_objective": "F2 with per-component validation FPR budget",
        "max_component_validation_fpr": MAX_COMPONENT_VALIDATION_FPR,
        "max_ensemble_validation_fpr": MAX_ENSEMBLE_VALIDATION_FPR,
    }
    manifest = {
        "feature_columns": features,
        "target_column": TARGET,
        "thresholds": thresholds,
        "xgboost_best_iteration": int(getattr(xgb_model, "best_iteration", XGB_PARAMS["n_estimators"] - 1)),
        "training_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
    }
    (MODEL_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")
    (MODEL_DIR / "feature_schema.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    xgb_model.save_model(MODEL_DIR / "xgboost_model.json")
    joblib.dump(isolation_model, MODEL_DIR / "isolation_forest.joblib", compress=3)
    joblib.dump(scaler, MODEL_DIR / "isolation_scaler.joblib", compress=3)

    results = {
        "tracking_uri": _safe_tracking_uri(tracking_uri),
        "scale_pos_weight": scale_pos_weight,
        "thresholds": thresholds,
        "best_iteration": manifest["xgboost_best_iteration"],
        "validation": {
            "xgboost": val_xgb_metrics,
            "isolation_forest": val_if_metrics,
            "ensemble": val_combined_metrics,
        },
        "test": {
            "xgboost": test_xgb_metrics,
            "isolation_forest": test_if_metrics,
            "ensemble": test_combined_metrics,
        },
    }
    metrics_path = ARTIFACT_DIR / "training_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    with mlflow.start_run(run_name="xgboost-isolation-forest"):
        mlflow.log_params({**XGB_PARAMS, **{f"if_{k}": v for k, v in IF_PARAMS.items()}})
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("threshold_tuning_objective", "F2 under validation FPR budget")
        mlflow.log_param("max_component_validation_fpr", MAX_COMPONENT_VALIDATION_FPR)
        mlflow.log_param("max_ensemble_validation_fpr", MAX_ENSEMBLE_VALIDATION_FPR)
        mlflow.log_param("xgb_probability_threshold", xgb_threshold)
        mlflow.log_param("isolation_anomaly_threshold", if_threshold)
        mlflow.log_param("feature_count", len(features))
        mlflow.log_param("train_rows", len(train))
        mlflow.log_param("validation_rows", len(validation))
        mlflow.log_param("test_rows", len(test))
        mlflow.log_param("xgboost_best_iteration", manifest["xgboost_best_iteration"])
        for model_name, metrics in (
            ("validation_xgb", val_xgb_metrics),
            ("validation_isolation_forest", val_if_metrics),
            ("validation_ensemble", val_combined_metrics),
            ("test_xgb", test_xgb_metrics),
            ("test_isolation_forest", test_if_metrics),
            ("test_ensemble", test_combined_metrics),
        ):
            _log_metrics(model_name, metrics)
        for artifact in ("xgboost_model.json", "isolation_forest.joblib", "isolation_scaler.joblib", "thresholds.json", "feature_schema.json"):
            mlflow.log_artifact(str(MODEL_DIR / artifact), artifact_path="models")
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
        results["mlflow_run_id"] = mlflow.active_run().info.run_id

    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results
