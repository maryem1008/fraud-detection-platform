"""FastAPI inference service for the trained fraud detection model bundle."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = ROOT / "models"


class Transaction(BaseModel):
    """One transaction using the original Kaggle credit-card feature schema."""

    model_config = ConfigDict(extra="forbid")

    time: Annotated[float, Field(ge=0)]
    amount: Annotated[float, Field(ge=0)]
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float
    v6: float
    v7: float
    v8: float
    v9: float
    v10: float
    v11: float
    v12: float
    v13: float
    v14: float
    v15: float
    v16: float
    v17: float
    v18: float
    v19: float
    v20: float
    v21: float
    v22: float
    v23: float
    v24: float
    v25: float
    v26: float
    v27: float
    v28: float

    @field_validator("time", "amount", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28")
    @classmethod
    def require_finite_values(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("feature values must be finite numbers")
        return value

    def to_feature_row(self, feature_columns: list[str]) -> pd.DataFrame:
        values = self.model_dump()
        row = {"Time": values["time"], "Amount": values["amount"]}
        row.update({f"V{i}": values[f"v{i}"] for i in range(1, 29)})
        return pd.DataFrame([row], columns=feature_columns)


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    latency_ms: float
    anomaly_score: float


@dataclass(frozen=True)
class ModelBundle:
    xgb: XGBClassifier
    isolation_forest: object
    isolation_scaler: object
    feature_columns: list[str]
    thresholds: dict[str, float]


def load_models(model_dir: Path | None = None) -> ModelBundle:
    directory = model_dir or Path(os.getenv("FRAUD_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    required = (
        "xgboost_model.json",
        "isolation_forest.joblib",
        "isolation_scaler.joblib",
        "feature_schema.json",
        "thresholds.json",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model artifacts in {directory}: {', '.join(missing)}")

    manifest = json.loads((directory / "feature_schema.json").read_text(encoding="utf-8"))
    thresholds = json.loads((directory / "thresholds.json").read_text(encoding="utf-8"))
    xgb = XGBClassifier()
    xgb.load_model(directory / "xgboost_model.json")
    isolation_forest = joblib.load(directory / "isolation_forest.joblib")
    isolation_scaler = joblib.load(directory / "isolation_scaler.joblib")
    # Keep concurrent API requests from multiplying estimator-level worker threads.
    xgb.set_params(n_jobs=1)
    isolation_forest.set_params(n_jobs=1)
    return ModelBundle(
        xgb=xgb,
        isolation_forest=isolation_forest,
        isolation_scaler=isolation_scaler,
        feature_columns=list(manifest["feature_columns"]),
        thresholds={key: float(value) for key, value in thresholds.items() if isinstance(value, (int, float))},
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load artifacts once per worker process, not once per prediction."""
    app.state.model_bundle = load_models()
    yield
    app.state.model_bundle = None


app = FastAPI(
    title="Real-Time Fraud Detection API",
    version="0.1.0",
    description="XGBoost-primary transaction fraud scoring with an Isolation Forest anomaly score.",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict[str, object]:
    ready = getattr(request.app.state, "model_bundle", None) is not None
    if not ready:
        raise HTTPException(status_code=503, detail="Model bundle is not loaded")
    return {"status": "ok", "models_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction, request: Request) -> PredictionResponse:
    bundle: ModelBundle | None = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model bundle is not loaded")

    started = time.perf_counter()
    features = transaction.to_feature_row(bundle.feature_columns)
    probability = float(bundle.xgb.predict_proba(features)[0, 1])
    scaled = bundle.isolation_scaler.transform(features)
    anomaly_score = float(-bundle.isolation_forest.decision_function(scaled)[0])
    is_fraud = (
        probability >= bundle.thresholds["xgb_probability"]
        or anomaly_score >= bundle.thresholds["isolation_anomaly_score"]
    )
    latency_ms = (time.perf_counter() - started) * 1000
    return PredictionResponse(
        fraud_probability=probability,
        is_fraud=bool(is_fraud),
        latency_ms=latency_ms,
        anomaly_score=anomaly_score,
    )
