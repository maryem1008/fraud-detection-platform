# FastAPI Inference Service and Latency Check

## API contract

- `GET /health` returns readiness only after the trained model bundle is loaded.
- `POST /predict` accepts `time`, `amount`, and `v1`–`v28`; unknown fields, negative time/amount, and non-finite values are rejected.
- Response: `fraud_probability` (XGBoost probability), `is_fraud` (locked OR threshold rule), `latency_ms`, and `anomaly_score` (for transparent audit/review).
- Model and preprocessing artifacts load once per worker at FastAPI lifespan startup, not per request.
- Runtime prediction uses one inference thread per estimator to limit CPU oversubscription; the deployment benchmark used **2 Uvicorn workers**.
- The 2-worker figure above is the configuration measured in this API benchmark. The Render-oriented Docker image now defaults to one worker for the 512 MB free-instance limit; its separate measurements are in [the Docker report](04_docker.md) and are not covered by the sub-100 ms local API result.

## Measured benchmark

Command:

```bash
.venv/bin/python scripts/benchmark_api.py \
  --url http://127.0.0.1:8001 --requests 250 --concurrency 8 \
  --output artifacts/api_benchmark.json
```

Observed on the Manus sandbox (6 logical CPUs, Intel Xeon Processor @ 2.10 GHz):

| Requests | Concurrency | Median inference latency | p95 | Maximum |
|---:|---:|---:|---:|---:|
| 250 | 8 | **73.19 ms** | 133.68 ms | 169.55 ms |

The stated **sub-100 ms median** criterion passed in this local benchmark. The p95 exceeds 100 ms, so no sub-100 ms tail-latency or production SLA claim is made. Results are hardware- and load-specific and should be re-measured on the target host.

This result is specific to the two-worker FastAPI process on the 6-logical-CPU sandbox. It must not be represented as the latency of the one-worker free-tier container or of a deployed service.

## Reproduction

Start the API with two workers and run the benchmark from another terminal:

```bash
.venv/bin/uvicorn fraud_detection.api.main:app --host 0.0.0.0 --port 8000 --workers 2
.venv/bin/python scripts/benchmark_api.py --url http://127.0.0.1:8000 --requests 250 --concurrency 8
```
