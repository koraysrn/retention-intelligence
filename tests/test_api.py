"""Online API tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.config import settings
from src.features.ecommerce import FEATURE_COLUMNS
from src.serving.api import app

MODEL_COLUMNS = FEATURE_COLUMNS
ADMIN = {"X-API-Key": settings.api_key}
READ = {"X-API-Key": settings.api_read_key}

HAS_RAW_DATA = settings.ecommerce_data.exists()
HAS_SCORES = (settings.data_processed / "scores.parquet").exists()
HAS_METRICS = (settings.artifacts_dir / "model_ecommerce_ensemble" / "metrics.json").exists()
HAS_FRONTEND = (Path(__file__).resolve().parents[1] / "frontend" / "dist" / "index.html").exists()


def test_health() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_returns_risk() -> None:
    from src.config import settings

    path = settings.data_processed / "customer_features.parquet"
    if not path.exists():
        pytest.skip("customer_features.parquet not found")

    row = pd.read_parquet(path).iloc[0]
    features: dict = {}
    for col in MODEL_COLUMNS:
        value = row[col]
        if pd.isna(value):
            features[col] = "MISSING" if col == "country" else 0.0
        elif isinstance(value, np.integer):
            features[col] = int(value)
        elif isinstance(value, np.floating):
            features[col] = float(value)
        else:
            features[col] = str(value)

    client = TestClient(app)
    resp = client.post(
        "/predict", json={"customer_id": "TEST-1", "features": features}, headers=ADMIN
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["customer_id"] == "TEST-1"
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_tier"] in {"low", "medium", "high"}


@pytest.mark.skipif(not HAS_FRONTEND, reason="frontend build not present")
def test_root_serves_html() -> None:
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


@pytest.mark.skipif(not HAS_SCORES, reason="scores.parquet not found")
def test_summary_endpoint() -> None:
    client = TestClient(app)
    resp = client.get("/api/summary", headers=ADMIN)
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["total_customers"] > 0
    assert set(body["risk_distribution"]).issubset({"low", "medium", "high"})


@pytest.mark.skipif(not HAS_METRICS, reason="model metrics not found")
def test_model_metrics_endpoint() -> None:
    client = TestClient(app)
    resp = client.get("/api/model-metrics", headers=ADMIN)
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("ensemble") is True
    assert "test_full" in body


@pytest.mark.skipif(not HAS_RAW_DATA, reason="ecommerce_data.csv not found")
def test_predict_customer_endpoint() -> None:
    client = TestClient(app)
    resp = client.get("/api/predict/1000", headers=ADMIN)
    assert resp.status_code == 200
    body = resp.json()
    assert body["customer_id"] == "1000"
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_tier"] in {"low", "medium", "high"}


@pytest.mark.skipif(not HAS_RAW_DATA, reason="ecommerce_data.csv not found")
def test_customer_detail_endpoint() -> None:
    client = TestClient(app)
    resp = client.get("/api/customers/1000", headers=ADMIN)
    assert resp.status_code == 200
    body = resp.json()
    assert body["customer_id"] == "1000"
    assert body["name"]
    assert body["country"]
    assert "risk_explanation" in body
    assert body["risk_tier"] in {"low", "medium", "high"}
    assert 0.0 <= body["churn_probability"] <= 1.0


@pytest.mark.skipif(not HAS_RAW_DATA, reason="ecommerce_data.csv not found")
def test_customer_detail_unknown_404() -> None:
    client = TestClient(app)
    resp = client.get("/api/customers/NONEXISTENT-ID", headers=ADMIN)
    assert resp.status_code == 404


@pytest.mark.skipif(not HAS_RAW_DATA, reason="ecommerce_data.csv not found")
def test_predict_customer_unknown_404() -> None:
    client = TestClient(app)
    resp = client.get("/api/predict/NONEXISTENT-ID", headers=ADMIN)
    assert resp.status_code == 404


def test_chat_endpoint() -> None:
    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={"message": "Why is the risk high?", "customer_id": None},
        headers=ADMIN,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"]
    assert "risk" in body["reply"].lower() or "Risk" in body["reply"]


@pytest.mark.skipif(not HAS_RAW_DATA, reason="ecommerce_data.csv not found")
def test_chat_endpoint_with_customer() -> None:
    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={"message": "Why is the risk high?", "customer_id": "1000"},
        headers=ADMIN,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"]


@pytest.mark.skipif(not HAS_SCORES, reason="scores.parquet not found")
def test_chat_coupon_action() -> None:
    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={
            "message": "send a 15% discount coupon to high-risk customers for 7 days",
            "confirm": False,
        },
        headers=ADMIN,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "confirmation required" in body["reply"].lower()
