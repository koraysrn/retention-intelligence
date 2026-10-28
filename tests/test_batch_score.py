"""Batch scoring tests."""

import numpy as np
import pandas as pd
import pytest
from src.serving.batch_score import assign_risk_tier, load_model, score_customers


class _FakePipeline:
    def predict_proba(self, x):
        n = len(x)
        return np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])


def test_assign_risk_tier() -> None:
    proba = pd.Series([0.05, 0.45, 0.75, 0.9, 0.4, 0.70])
    assert list(assign_risk_tier(proba)) == [
        "low",
        "medium",
        "high",
        "high",
        "medium",
        "high",
    ]


def test_score_customers_assigns_tiers() -> None:
    features = pd.DataFrame({"customer_id": ["A", "B", "C"], "x": [1.0, 2.0, 3.0]})
    config = {
        "model_version": "test",
        "threshold": 0.5,
        "medium_risk": 0.4,
        "high_risk": 0.7,
        "feature_schema": [],
    }
    scores = score_customers(features, model=_FakePipeline(), serving_config=config)
    assert list(scores["customer_id"]) == ["A", "B", "C"]
    assert (scores["churn_probability"] == 0.6).all()
    assert list(scores["risk_tier"]) == ["medium", "medium", "medium"]


def test_load_model_raises_when_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.joblib")
