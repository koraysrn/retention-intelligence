"""As-of e-commerce feature engineering and label tests."""

from __future__ import annotations

import pandas as pd
from src.features.ecommerce import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    build_churn_labels,
    build_customer_features,
    build_dataset,
)

AS_OF = pd.Timestamp("2025-01-31")
HORIZON = 30


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["1", "2", "3"],
            "name": ["A", "B", "C"],
            "email": ["a@x.com", "b@x.com", "c@x.com"],
            "country": ["TR", "US", "DE"],
            "age": [25, 40, 55],
            "age_group": ["18-24", "35-44", "55+"],
            "signup_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "marketing_opt_in": [True, False, True],
            "total_orders": [1, 1, 3],
            "total_spend_usd": [50.0, 50.0, 300.0],
            "avg_order_value": [50.0, 50.0, 100.0],
            "avg_discount_pct": [5.0, 5.0, 10.0],
            "first_order_date": pd.to_datetime(["2025-02-10", "2024-12-01", "2024-11-01"]),
            "last_order_date": pd.to_datetime(["2025-02-10", "2024-12-01", "2025-02-05"]),
            "preferred_payment": ["card", "card", "paypal"],
            "preferred_device_ord": ["desktop", "desktop", "mobile"],
            "preferred_source": ["email", "email", "organic"],
            "top_category_bought": ["Beauty", "Beauty", "Sports"],
            "avg_rating_given": [4.0, None, 5.0],
            "total_sessions": [2, 5, 8],
            "preferred_device_sess": ["mobile", "desktop", "mobile"],
            "preferred_source_sess": ["direct", "email", "organic"],
            "first_session_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "last_session_date": pd.to_datetime(["2024-06-01", "2024-07-01", "2025-01-20"]),
            "has_abandoned_cart": [0, 1, 0],
            "clv_tier": ["no_purchase", "low", "medium"],
            "is_repeat_customer": [0, 0, 1],
        }
    )


def _orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["O1", "O2", "O3", "O4"],
            "customer_id": ["1", "2", "3", "3"],
            "order_date": pd.to_datetime(["2025-02-10", "2024-12-01", "2024-11-01", "2025-02-05"]),
            "amount_usd": [50.0, 50.0, 100.0, 100.0],
        }
    )


def test_churn_is_outcome_window_not_inverse_repeat() -> None:
    labels = build_churn_labels(_sample(), AS_OF, HORIZON, _orders())
    # C1 bought in the outcome window (churn=0), C2 did not (churn=1),
    # C3 bought in the outcome window (churn=0).
    assert list(labels[TARGET_COLUMN]) == [0, 1, 0]
    assert labels["ordered_in_outcome_window"].tolist() == [1, 0, 1]


def test_churn_label_works_without_legacy_repeat_column() -> None:
    df = _sample().drop(columns="is_repeat_customer")
    labels = build_churn_labels(df, AS_OF, HORIZON, _orders())
    assert "churn_legacy_repeat" not in labels.columns
    assert labels[TARGET_COLUMN].tolist() == [0, 1, 0]


def test_features_exclude_pii_label_and_leaky_columns() -> None:
    features = build_customer_features(_sample(), AS_OF, HORIZON, _orders())
    for col in ("name", "email", "is_repeat_customer", "total_orders"):
        assert col not in features.columns
    for col in (
        "total_spend_usd",
        "avg_order_value",
        "avg_discount_pct",
        "order_span_days",
        "clv_tier",
        "preferred_payment",
        "top_category_bought",
        "ordered_in_outcome_window",
    ):
        assert col not in features.columns
    for col in (
        "signup_date",
        "first_order_date",
        "last_order_date",
        "first_session_date",
        "last_session_date",
    ):
        assert col not in features.columns


def test_features_have_deterministic_schema() -> None:
    features = build_customer_features(_sample(), AS_OF, HORIZON, _orders())
    assert list(features.columns) == [ID_COLUMN] + FEATURE_COLUMNS + ["pre_asof_segment"]


def test_customer_id_present_and_unique() -> None:
    features = build_customer_features(_sample(), AS_OF, HORIZON, _orders())
    assert features[ID_COLUMN].is_unique
    assert set(features[ID_COLUMN]) == {"1", "2", "3"}


def test_has_purchase_uses_pre_cutoff_orders() -> None:
    features = build_customer_features(_sample(), AS_OF, HORIZON, _orders())
    # Customer 1 only bought after the cutoff -> has_purchase = 0.
    assert features["has_purchase"].tolist() == [0, 1, 1]
    # Customer 3 has only one pre-cutoff order (2024-11-01), so it is one_time
    # until the outcome-window order arrives.
    assert features["pre_asof_segment"].tolist() == ["zero_order", "one_time", "one_time"]


def test_recency_uses_last_pre_cutoff_order() -> None:
    features = build_customer_features(_sample(), AS_OF, HORIZON, _orders())
    row = features.loc[features[ID_COLUMN] == "3"].iloc[0]
    # Last pre-cutoff order for customer 3 is 2024-11-01 -> 91 days.
    assert row["recency_days"] == 91
    # The post-cutoff order (2025-02-05) must not affect the feature.
    assert row["order_session_gap_days"] == -80


def test_build_dataset_merges_on_customer_id() -> None:
    dataset = build_dataset(_sample(), AS_OF, HORIZON, _orders())
    assert {"features", "labels", "training_set"}.issubset(dataset)
    training = dataset["training_set"]
    assert TARGET_COLUMN in training.columns
    assert len(training) == len(_sample())
    assert training[ID_COLUMN].is_unique
    assert set(FEATURE_COLUMNS).issubset(training.columns)
    assert "ordered_in_outcome_window" in training.columns


def test_build_dataset_churn_distribution() -> None:
    dataset = build_dataset(_sample(), AS_OF, HORIZON, _orders())
    assert dataset["training_set"][TARGET_COLUMN].value_counts().to_dict() == {0: 2, 1: 1}
