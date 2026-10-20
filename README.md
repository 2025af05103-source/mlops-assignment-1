# Heart Disease Risk Prediction — MLOps Assignment 1

End-to-end MLOps pipeline: data acquisition → EDA → model training → MLflow tracking
→ packaging → CI/CD (GitHub Actions) → Dockerized FastAPI → Kubernetes deployment →
Prometheus monitoring.

## Quick Start

```bash
# 1. Create environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download & clean data
python scripts/download_data.py

# 3. EDA (optional, generates plots into artifacts/eda/)
jupyter execute notebooks/01_eda.ipynb   # or open in Jupyter

# 4. Train models with MLflow tracking
python src/train.py

# 5. Serve the API locally
uvicorn src.api.app:app --port 8000
# predict:
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d @data/sample_input.json

# 6. Docker
docker build -t heart-disease-api:latest -f docker/Dockerfile .
docker run -p 8000:8000 heart-disease-api:latest

# 7. Kubernetes (Docker Desktop K8s or Minikube)
kubectl apply -f k8s/

# 8. Monitoring
kubectl apply -f monitoring/
```

## Repo Layout

```
scripts/download_data.py    # dataset download + cleaning
src/preprocess.py           # reusable preprocessing pipeline
src/train.py                # training, tuning, MLflow logging
src/api/app.py              # FastAPI serving app
tests/                      # pytest unit tests
.github/workflows/ci.yml    # lint + test + train pipeline
docker/Dockerfile           # API container
k8s/                        # deployment + service + ingress manifests
monitoring/                 # Prometheus + Grafana configs
notebooks/                  # EDA notebook
artifacts/                  # generated plots, models, MLflow runs
```
