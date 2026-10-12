# Docker Containerization and Local Verification

## Image design

- Multi-stage `python:3.11-slim` build: the builder installs API dependencies into a virtual environment; the runtime copies only that environment, application source, and model artifacts.
- Serving dependencies use the official [XGBoost CPU-only package](https://xgboost.readthedocs.io/en/stable/install.html), avoiding GPU algorithms and their unneeded NCCL dependency. Training and drift-monitoring libraries are excluded from the inference image.
- Runtime installs `libgomp1`, runs as the unprivileged `app` user, and has a Docker healthcheck against `/health`.
- Uvicorn defaults to one worker with `WEB_CONCURRENCY` configurable; the server port follows `PORT` (default 8000, tested at Render's 10000).
- Build context excludes data, notebooks, training scripts, MLflow files, and local development tools.

## Local validation

- Image built successfully: `fraud-detection-api:local`.
- Image size: **178,432,704 bytes (~170.2 MiB)**. The initial full-XGBoost image was 658,836,845 bytes (~628.2 MiB); the CPU-only image is about **72.9% smaller**.
- Container runs as user `app`; the final image reached Docker health status `healthy` with both the default port and `PORT=10000`; it was not OOM-killed at 512 MB.
- `GET /health` returned `{"status":"ok","models_loaded":true}`; `POST /predict` returned the expected response on a sample transaction.
- Final one-worker container benchmark on the 6-logical-CPU sandbox: 250 test transactions, concurrency 8 — **118.80 ms median**, 239.88 ms p95, 331.74 ms maximum. This final container configuration did **not** meet the sub-100 ms median target; do not transfer the earlier two-worker API result to this image.
- With the same image capped locally at 0.1 CPU / 512 MB to approximate Render Free, 30 sequential warm requests measured **203.78 ms median**, 297.55 ms p95, and 298.99 ms max. This is a sandbox CPU-quota simulation, not a measurement on Render. The loaded container used about 256.7 MiB during the full-host benchmark and 248.8 MiB under the CPU-quota simulation.

## Reproduction

On a typical Docker host, use port publishing (the image defaults to one worker; `PORT` is read at startup):

```bash
docker build -t fraud-detection-api:local .
docker run --rm --name fraud-detection-api -p 8000:8000 fraud-detection-api:local
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @examples/sample_request.json
```

In this sandbox only, Docker's default bridge networking could not initialize because the kernel lacks the iptables `raw` table. The successful local build used `docker build --network=host`, and the test container used `--network=host`; this is an environment-specific workaround, not a requirement on a normal Docker host.
