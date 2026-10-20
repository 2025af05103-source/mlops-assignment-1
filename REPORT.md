# Heart Disease Risk Prediction — MLOps Assignment 1 Report

**Course:** MLOps (AIMLCZG523) | **Student:** [Your Name] | **Date:** October 2026

## 1. Overview

This project builds an end-to-end MLOps pipeline for predicting heart-disease risk from
patient data: automated data acquisition, EDA, model training with experiment tracking,
packaging, CI/CD, containerization, Kubernetes deployment, and monitoring.

**Final model:** Random Forest — holdout ROC-AUC **0.942**, accuracy 0.833, precision 0.875, recall 0.750.

## 2. Setup / Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py   # data
python src/train.py               # train + MLflow
uvicorn src.api.app:app --port 8000
docker build -t heart-disease-api:latest -f docker/Dockerfile .   # container
kubectl apply -f k8s/ -f monitoring/                              # deploy
```

## 3. Data & EDA (Task 1)

Dataset: UCI Heart Disease (Cleveland, 303 raw → 297 clean rows after dropping 6 rows
with missing `ca`/`thal`; target binarized 0/1; 160 no-disease vs 137 disease — mild
imbalance, no resampling needed).

Key EDA findings (plots in `artifacts/eda/`):
- `thalach` (max heart rate) lower and `oldpeak` (ST depression) higher in disease patients.
- Chest-pain type `cp=0` (asymptomatic) has by far the highest disease rate.
- Correlations with target: `ca` (0.46), `thal` (0.44), `oldpeak` (0.42), `thalach` (-0.42).

## 4. Modelling (Task 2)

Two model families, both wrapped in a full sklearn `Pipeline` (imputation → scaling →
encoding → classifier), tuned via GridSearchCV with stratified 5-fold CV scored on ROC-AUC:

| Model | Best params | CV ROC-AUC | Test ROC-AUC | Test Acc | Precision | Recall |
|---|---|---|---|---|---|---|
| Logistic Regression | C=1.0, liblinear | 0.903 | 0.941 | 0.817 | 0.870 | 0.714 |
| **Random Forest** | 200 trees, depth 6, leaf 2 | 0.895 | **0.942** | 0.833 | 0.875 | 0.750 |

Selection criterion: highest holdout ROC-AUC → Random Forest. Both models performed
nearly identically; RF won marginally and offers feature-interpretability via importances.

## 5. Experiment Tracking (Task 3)

MLflow local tracking (`mlruns/`, experiment `heart-disease-classification`). Each run
logs: model family, full grid, best hyperparameters, CV + test metrics (accuracy,
precision, recall, ROC-AUC, F1), confusion-matrix and ROC-curve artifacts, and the
pickled pipeline. Summary in `artifacts/mlflow_run_summary.csv`.
View with: `mlflow ui` → http://localhost:5000

## 6. Packaging & Reproducibility (Task 4)

- Final pipeline (preprocessing + model) serialized as one `models/final_pipeline.joblib`.
- Pinned `requirements.txt`; training uses a fixed random seed (42).
- `scripts/download_data.py` fetches the raw dataset from a fixed URL, so the whole
  pipeline runs from a clean checkout: download → train → serve.
- The serving app loads the same artifact — no preprocessing code duplication.

## 7. Testing & CI/CD (Task 5)

16 pytest tests across three suites: preprocessing (shape, no-NaN, unknown-category
handling, end-to-end pipeline), training utilities + saved-model quality guardrail
(ROC-AUC > 0.80), and API contract (health, predict, validation errors, metrics).

GitHub Actions pipeline (`.github/workflows/ci.yml`), four jobs with dependency chain:
**lint** (flake8 + black) → **test** (pytest, uploads results) → **train** (trains with
MLflow, uploads model + runs as artifacts) → **docker** (builds image, smoke-tests
`/health` + `/predict` inside the container). Any failure fails the pipeline with logs.

## 8. Containerization (Task 6)

`docker/Dockerfile`: python:3.12-slim, layer-cached installs, healthcheck, uvicorn.
Verified locally: build → run → `curl /health` and `/predict` return identical results
to the local API (proof in `artifacts/docker_proof.txt`).

## 9. Deployment (Task 7)

Docker Desktop Kubernetes (local cluster). Manifests in `k8s/`:
- `deployment.yaml`: 2 replicas, liveness/readiness probes on `/health`, resource limits.
- `service.yaml`: LoadBalancer → port 80. Verified at `http://localhost` (see
  `artifacts/deployment_proof.txt` and `screenshots/`).

## 10. Monitoring & Logging (Task 8)

- Structured request logging in the API (method, path, status, latency ms).
- Prometheus metrics: request count by endpoint/status, latency histogram, prediction
  counts by class, latest confidence, model-loaded flag.
- Prometheus deployed in-cluster, scraping the API every 15s (target verified `up`);
  Grafana at `localhost:3000` with auto-provisioned Prometheus datasource.

## 11. Architecture

```
GitHub ──push──► GitHub Actions (lint → tests → train → docker build)
                      │ uploads: test results, MLflow runs, model
                      ▼
 scripts/download_data.py ─► data/heart_disease_clean.csv
        │
        ▼
 src/train.py ──MLflow──► mlruns/   +   models/final_pipeline.joblib
        │                                   │
        ▼                                   ▼
 docker/Dockerfile ─► image ─► k8s Deployment (2 replicas, probes)
                                   │ Service (LoadBalancer :80)
                                   ├─► /health  /predict  (FastAPI)
                                   └─► /metrics ──scrape──► Prometheus ─► Grafana
```

## 12. How to Verify

- API: `curl localhost/predict -H 'Content-Type: application/json' -d @data/sample_input.json`
- MLflow UI: `mlflow ui`
- Tests: `pytest tests/ -v`
- Metrics: `curl localhost/metrics`, Grafana: `localhost:3000` (admin/admin)
