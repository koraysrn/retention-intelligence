"""Training pipeline (train) tests.

Synthetic data is used; small n_estimators values are used for speed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.models.prepare import categorical_features, numeric_features, split_features_target
from src.models.train import (
    build_pipeline,
    compute_shap_importance,
    make_model,
    walk_forward_cv,
)
from src.serving.batch_score import assign_risk_tier

LGB_FAST_PARAMS = {
    "n_estimators": 20,
    "num_leaves": 7,
    "learning_rate": 0.1,
    "verbose": -1,
    "random_state": 42,
}


def _synthetic_data(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    # Chronological order
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    x1 = rng.normal(size=n)
    x2 = rng.uniform(0, 100, size=n)
    gender = rng.choice(["Female", "Male"], size=n)
    country = rng.choice(["TR", "US", "DE"], size=n)

    # x1 is a strong signal; churn probability
    logit = 2.0 * x1 + 0.01 * x2
    proba = 1.0 / (1.0 + np.exp(-logit))
    target = (rng.random(n) < proba).astype(int)

    return pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(n)],
            "x1": x1,
            "x2": x2,
            "gender": gender,
            "country": country,
            "last_order_date": dates,
            "churn": target,
            "constant_col": [1] * n,
        }
    )


def test_make_model_types() -> None:
    assert hasattr(make_model("lightgbm"), "predict_proba")
    assert hasattr(make_model("xgboost"), "predict_proba")
    assert hasattr(make_model("catboost"), "predict_proba")
    assert hasattr(make_model("logistic"), "predict_proba")


def test_make_model_unknown_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model type"):
        make_model("svm")


def test_build_pipeline_fit_predict() -> None:
    df = _synthetic_data()
    x, y = split_features_target(df)
    pipe = build_pipeline("lightgbm", LGB_FAST_PARAMS, numeric_features(x), categorical_features(x))
    pipe.fit(x.iloc[:120], y.iloc[:120])
    proba = pipe.predict_proba(x.iloc[120:])[:, 1]
    assert len(proba) == 120
    assert (proba >= 0).all() and (proba <= 1).all()


def test_walk_forward_cv_returns_folds() -> None:
    df = _synthetic_data()
    x, y = split_features_target(df)
    results = walk_forward_cv(
        x,
        y,
        "lightgbm",
        n_splits=3,
        params=LGB_FAST_PARAMS,
        numeric_cols=numeric_features(x),
        categorical_cols=categorical_features(x),
    )
    assert len(results) == 3
    for r in results:
        assert "fold" in r
        assert "pr_auc" in r
        assert 0.0 <= r["pr_auc"] <= 1.0


def test_shap_importance_sorted_and_named() -> None:
    df = _synthetic_data(n=120)
    x, y = split_features_target(df)
    pipe = build_pipeline("lightgbm", LGB_FAST_PARAMS, numeric_features(x), categorical_features(x))
    pipe.fit(x, y)

    importance = compute_shap_importance(pipe, x.iloc[:60], "lightgbm")
    assert {"feature", "mean_abs_shap"}.issubset(importance.columns)
    assert len(importance) == 4  # x1, x2, gender, country
    assert importance["mean_abs_shap"].is_monotonic_decreasing


def test_risk_tier_assignment() -> None:
    proba = pd.Series([0.05, 0.45, 0.75, 0.9, 0.4])
    tiers = assign_risk_tier(proba)
    assert list(tiers) == ["low", "medium", "high", "high", "medium"]
