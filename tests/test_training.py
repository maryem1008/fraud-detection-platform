"""Tests for threshold-selection guardrails and tracking-URI redaction."""

from __future__ import annotations

import numpy as np

from fraud_detection.training import _safe_tracking_uri, choose_fbeta_threshold


def test_threshold_respects_validation_false_positive_budget() -> None:
    y_true = np.asarray([0] * 100 + [1] * 4)
    scores = np.concatenate(
        (np.linspace(0.0, 0.2, 100), np.asarray([0.8, 0.85, 0.9, 0.95]))
    )

    threshold = choose_fbeta_threshold(y_true, scores, max_fpr=0.01)
    predictions = scores >= threshold
    false_positive_rate = float(predictions[:100].mean())

    assert false_positive_rate <= 0.01
    assert predictions[100:].sum() == 4


def test_tracking_uri_redacts_credentials_and_query_parameters() -> None:
    uri = "https://analyst:secret@example.test:8443/mlflow?token=hidden#fragment"

    assert _safe_tracking_uri(uri) == "https://example.test:8443/mlflow"
    assert "secret" not in _safe_tracking_uri(uri)
    assert "hidden" not in _safe_tracking_uri(uri)


def test_sqlite_tracking_uri_is_displayed_as_relative_store() -> None:
    assert _safe_tracking_uri("sqlite:////private/path/mlflow.db") == "sqlite:///mlflow.db"
