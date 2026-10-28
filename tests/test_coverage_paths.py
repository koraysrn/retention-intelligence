"""Additional coverage for API read paths, batch scoring and promotion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.config import settings
from src.serving.api import app

ADMIN = {"X-API-Key": settings.api_key}


@pytest.mark.parametrize(
    "path",
    [
        "/api/summary",
        "/api/model-metrics",
        "/api/business-metrics",
        "/api/experiments",
        "/api/population",
        "/api/insights",
        "/api/predict/1000",
    ],
)
def test_authenticated_read_endpoints_return_200(path: str) -> None:
    client = TestClient(app)
    resp = client.get(path, headers=ADMIN)
    assert resp.status_code == 200
    assert isinstance(resp.json(), dict)


def test_batch_score_main_writes_scores(tmp_path: Path) -> None:
    import sys

    from src.serving import batch_score

    out = tmp_path / "scores.parquet"
    argv = [
        "batch_score",
        "--features",
        str(settings.data_processed / "customer_features.parquet"),
        "--model",
        str(settings.artifacts_dir / "model_ecommerce_ensemble" / "pipeline.joblib"),
        "--serving-config",
        str(settings.artifacts_dir / "model_ecommerce_ensemble" / "serving_config.json"),
        "--out",
        str(out),
    ]
    old = sys.argv
    try:
        sys.argv = argv
        batch_score.main()
    finally:
        sys.argv = old
    assert out.exists()
    df = pd.read_parquet(out)
    assert "model_version" in df.columns


def test_retrain_promotes_candidate(tmp_path: Path) -> None:
    from src.monitoring.retrain import _promote_candidate

    candidate = tmp_path / "candidate"
    target = tmp_path / "target"
    candidate.mkdir()
    (candidate / "metrics.json").write_text(
        json.dumps({"model_version": "v-test", "ensemble": True}), encoding="utf-8"
    )
    for name in (
        "pipeline.joblib",
        "serving_config.json",
        "shap_importance.csv",
        "explanations.json",
    ):
        (candidate / name).write_text("x", encoding="utf-8")
    version = _promote_candidate(candidate, target)
    assert version == "v-test"
    assert (target / "metrics.json").exists()
    assert (target / "current").is_symlink()


def test_evaluate_extra_metrics() -> None:
    from src.models.evaluate import (
        bootstrap_ci,
        find_best_threshold_f1,
        find_best_threshold_youden,
    )

    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, 200))
    score = pd.Series(np.clip(0.2 + 0.5 * y + rng.normal(0, 0.2, 200), 0, 1))
    assert 0 <= find_best_threshold_f1(y, score) <= 1
    assert 0 <= find_best_threshold_youden(y, score) <= 1
    ci = bootstrap_ci(y, score, metric="roc_auc", n_boot=20)
    assert ci["low"] <= ci["high"]
    with pytest.raises(ValueError):
        bootstrap_ci(y, score, metric="unknown")


def test_feature_aggregate_fallback_without_orders() -> None:
    from src.features.ecommerce import build_customer_features

    customers = pd.DataFrame(
        {
            "customer_id": ["1", "2"],
            "country": ["TR", "US"],
            "age": [30, 40],
            "age_group": ["25-34", "35-44"],
            "signup_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "marketing_opt_in": [True, False],
            "first_order_date": pd.to_datetime(["2024-02-01", "2024-03-01"]),
            "last_order_date": pd.to_datetime(["2024-02-01", "2024-04-01"]),
            "total_sessions": [3, 4],
            "preferred_device_sess": ["mobile", "desktop"],
            "preferred_source_sess": ["direct", "email"],
            "first_session_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "last_session_date": pd.to_datetime(["2024-02-01", "2024-03-01"]),
            "has_abandoned_cart": [0, 1],
            "avg_rating_given": [4.0, None],
        }
    )
    features = build_customer_features(customers, "2024-03-01", 30, orders=None)
    assert features["has_purchase"].tolist() == [1, 1]
    # Customer 1's last order is pre-cutoff -> exact recency; customer 2 has a
    # post-cutoff last order, so the aggregate fallback leaves recency unknown.
    assert features.loc[0, "recency_days"] == 29
    assert pd.isna(features.loc[1, "recency_days"])


def test_actions_empty_scores() -> None:
    from src.agents.actions import list_risk_segments, send_coupon_to_risk_segment

    empty = pd.DataFrame(columns=["customer_id", "churn_probability", "risk_tier"])
    assert list_risk_segments(scores=empty) == {"available": False, "segments": {}}
    result = send_coupon_to_risk_segment(
        discount_pct=10, validity_days=7, scores=empty, confirm=True
    )
    assert result["sent"] == 0
