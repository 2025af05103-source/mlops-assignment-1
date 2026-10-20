"""FastAPI serving app for the heart-disease model.

Loads the fitted pipeline saved by src/train.py and exposes:
  GET  /health   -> liveness + model status
  GET  /metrics  -> Prometheus metrics
  POST /predict  -> JSON patient record in, prediction + confidence out
"""

import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

try:
    from preprocess import ALL_FEATURES
except ImportError:  # running as src.api.app; src/ not on sys.path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from preprocess import ALL_FEATURES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("heart_api")

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "final_pipeline.joblib"

# --- Prometheus metrics ---------------------------------------------------
REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Request latency",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
PREDICTIONS = Counter(
    "predictions_total", "Total predictions served", ["predicted_class"]
)
PREDICTION_CONFIDENCE = Gauge(
    "prediction_confidence", "Confidence of the most recent prediction"
)
MODEL_LOADED = Gauge("model_loaded", "1 if the model artifact loaded successfully")

app = FastAPI(title="Heart Disease Risk API", version="1.0.0")

try:
    pipeline = joblib.load(MODEL_PATH)
    MODEL_LOADED.set(1)
    log.info("Model loaded from %s", MODEL_PATH)
except Exception:
    pipeline = None
    MODEL_LOADED.set(0)
    log.exception("Failed to load model from %s", MODEL_PATH)


class PatientRecord(BaseModel):
    """One patient record; field names match the cleaned dataset columns."""

    age: float = Field(..., ge=20, le=100, examples=[63])
    sex: float = Field(..., ge=0, le=1, examples=[1])
    cp: float = Field(..., ge=0, le=3, examples=[3])
    trestbps: float = Field(..., ge=80, le=250, examples=[145])
    chol: float = Field(..., ge=80, le=700, examples=[233])
    fbs: float = Field(..., ge=0, le=1, examples=[1])
    restecg: float = Field(..., ge=0, le=2, examples=[0])
    thalach: float = Field(..., ge=60, le=250, examples=[150])
    exang: float = Field(..., ge=0, le=1, examples=[0])
    oldpeak: float = Field(..., ge=-2, le=10, examples=[2.3])
    slope: float = Field(..., ge=0, le=2, examples=[0])
    ca: float = Field(..., ge=0, le=3, examples=[0])
    thal: float = Field(..., ge=0, le=3, examples=[1])


@app.middleware("http")
async def log_and_track(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    REQUEST_COUNT.labels(request.method, request.url.path, response.status_code).inc()
    REQUEST_LATENCY.labels(request.url.path).observe(elapsed)
    log.info(
        "%s %s -> %s in %.1f ms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed * 1000,
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(record: PatientRecord):
    if pipeline is None:
        return Response(
            content='{"error": "model not loaded"}',
            status_code=503,
            media_type="application/json",
        )
    X = pd.DataFrame([record.model_dump()])[ALL_FEATURES]
    proba = float(pipeline.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)
    PREDICTIONS.labels(str(pred)).inc()
    PREDICTION_CONFIDENCE.set(proba)
    return {
        "prediction": pred,
        "label": "heart disease" if pred else "no heart disease",
        "confidence": round(max(proba, 1 - proba), 4),
        "risk_score": round(proba, 4),
    }
