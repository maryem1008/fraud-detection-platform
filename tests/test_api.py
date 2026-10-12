"""Integration tests for the inference API using the checked-in model bundle."""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from fraud_detection.api.main import app


def sample_transaction() -> dict[str, float]:
    return {
        "time": 42_000.0,
        "amount": 149.62,
        **{f"v{index}": 0.01 * index for index in range(1, 29)},
    }


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_loaded_models(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "models_loaded": True}


def test_predict_returns_finite_scores(client: TestClient) -> None:
    response = client.post("/predict", json=sample_transaction())

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "fraud_probability",
        "is_fraud",
        "latency_ms",
        "anomaly_score",
    }
    assert isinstance(payload["is_fraud"], bool)
    assert 0.0 <= payload["fraud_probability"] <= 1.0
    assert payload["latency_ms"] >= 0.0
    assert math.isfinite(payload["anomaly_score"])


def test_predict_rejects_extra_fields(client: TestClient) -> None:
    transaction = sample_transaction() | {"card_number": "not-accepted"}

    response = client.post("/predict", json=transaction)

    assert response.status_code == 422


def test_predict_rejects_negative_amount(client: TestClient) -> None:
    transaction = sample_transaction() | {"amount": -0.01}

    response = client.post("/predict", json=transaction)

    assert response.status_code == 422
