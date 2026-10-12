# MLflow Tracking

## Local portfolio setup

Training reads `MLFLOW_TRACKING_URI`. Without it, runs are stored in the project-local SQLite database `mlflow.db`. Start the UI/API against that store with:

```bash
.venv/bin/mlflow server \
  --backend-store-uri sqlite:////absolute/path/to/fraud-detection-platform/mlflow.db \
  --default-artifact-root file:///absolute/path/to/fraud-detection-platform/mlartifacts \
  --host 127.0.0.1 --port 5000
```

Then open `http://127.0.0.1:5000` locally. The database, model artifacts, and MLflow artifact directory are local experiment state and are not committed to Git.

## Remote tracking

For a persistent remote MLflow server, set its URI in the environment before training; no server address or credential is hard-coded:

```bash
export MLFLOW_TRACKING_URI="https://your-mlflow-host.example"
.venv/bin/python scripts/train.py
```

In automation, configure the URI and any required authentication via CI secrets. Use a remote artifact store with durable access for production; a local file/SQLite store on an ephemeral runner will not persist between CI jobs.

## Verification

A manual retraining run was executed with an explicit SQLite URI and appeared through the running HTTP tracking server. The experiment listed **4 runs total** at verification time (three model-training runs and one threshold-sensitivity evaluation run). The final constrained model run and the manual rerun had matching held-out test metrics:

| Run | Run ID | Test precision | Test recall | Test FPR | Test false positives |
|---|---|---:|---:|---:|---:|
| Final constrained training | `6bc5a1a768e84464ac005489f018844e` | 84.44% | 73.08% | 0.0164% | 7 |
| Manual retraining | `41884633abe64f0e93901d10cdf66d1c` | 84.44% | 73.08% | 0.0164% | 7 |

The manual retrain therefore demonstrates a new logged run with comparable evaluation metrics. A CI-triggered retrain will be verified after the workflow is connected to the repository and its required data/secrets are available.
