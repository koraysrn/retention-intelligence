"""Feature preparation (prepare) tests — ecommerce schema."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.models.prepare import (
    build_preprocessor,
    categorical_features,
    numeric_features,
    split_features_target,
)


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "total_orders": [2, 1, 1],  # leaky -> must always be dropped
            "total_spend_usd": [10.0, 20.0, 30.0],  # leaky -> dropped
            "recency_days": [5.0, 30.0, 120.0],
            "total_sessions": [5, 30, 120],
            "country": ["Canada", "USA", "UK"],
            "signup_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "churn": [0, 1, 0],
        }
    )


def test_split_drops_identifiers_labels_leaky_datetimes_constants() -> None:
    x, y = split_features_target(_training_frame())
    assert "customer_id" not in x.columns
    assert "churn" not in x.columns
    assert "total_orders" not in x.columns
    assert "signup_date" not in x.columns
    assert list(y) == [0, 1, 0]


def test_split_raises_on_missing_target() -> None:
    df = _training_frame().drop(columns="churn")
    with pytest.raises(ValueError, match="Target column"):
        split_features_target(df)


def test_feature_type_lists() -> None:
    x, _ = split_features_target(_training_frame())
    assert set(numeric_features(x)) == {"recency_days", "total_sessions"}
    assert set(categorical_features(x)) == {"country"}


def test_preprocessor_imputes_and_encodes() -> None:
    x, _ = split_features_target(_training_frame())
    pre = build_preprocessor(numeric_features(x), categorical_features(x))

    x_missing = x.copy()
    x_missing.loc[0, "recency_days"] = np.nan
    x_missing.loc[1, "country"] = np.nan

    out = pre.fit_transform(x_missing)
    assert not np.isnan(out).any()
    assert out.shape[1] == 3  # 2 numeric + 1 categorical


def test_preprocessor_handles_unknown_category() -> None:
    x, _ = split_features_target(_training_frame())
    pre = build_preprocessor(numeric_features(x), categorical_features(x))
    pre.fit(x)

    unknown = pd.DataFrame(
        {
            "recency_days": [5.0],
            "total_sessions": [10],
            "country": ["Mars"],
        }
    )
    out = pre.transform(unknown)
    assert out.shape == (1, 3)
    assert not np.isnan(out).any()
