"""Unit tests for the FastAPI app using the TestClient (no server needed)."""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.app import app

client = TestClient(app)

VALID_RECORD = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_contract():
    r = client.post("/predict", json=VALID_RECORD)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0.0 <= body["risk_score"] <= 1.0


def test_predict_rejects_missing_field():
    bad = {k: v for k, v in VALID_RECORD.items() if k != "age"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_out_of_range_age():
    r = client.post("/predict", json={**VALID_RECORD, "age": 150})
    assert r.status_code == 422


def test_metrics_endpoint_exposes_prometheus_format():
    client.get("/health")  # generate traffic
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"api_requests_total" in r.content
    assert b"api_request_latency_seconds" in r.content
