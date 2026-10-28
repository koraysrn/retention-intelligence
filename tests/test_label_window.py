"""Outcome-window label boundary tests."""

from __future__ import annotations

import pandas as pd
from src.features.ecommerce import build_churn_labels

AS_OF = pd.Timestamp("2025-01-31")
HORIZON = 30


def _labels(order_dates: list[pd.Timestamp | None]) -> list[int]:
    customers = pd.DataFrame(
        {
            "customer_id": [str(i) for i in range(len(order_dates))],
            "first_order_date": order_dates,
            "last_order_date": order_dates,
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": [f"O{i}" for i in range(len(order_dates))],
            "customer_id": [str(i) for i in range(len(order_dates))],
            "order_date": order_dates,
        }
    )
    return build_churn_labels(customers, AS_OF, HORIZON, orders)["churn"].tolist()


def test_window_boundaries_are_strict_left_closed_right() -> None:
    labels = _labels(
        [
            AS_OF,  # exactly at cutoff -> not in outcome
            AS_OF + pd.Timedelta(days=1),  # first outcome day
            AS_OF + pd.Timedelta(days=HORIZON),  # last outcome day (inclusive)
            AS_OF + pd.Timedelta(days=HORIZON + 1),  # after window
            None,  # never purchased
        ]
    )
    assert labels == [1, 0, 0, 1, 1]


def test_zero_order_with_outcome_purchase_is_not_churn() -> None:
    customers = pd.DataFrame(
        {
            "customer_id": ["Z"],
            "first_order_date": [None],
            "last_order_date": [None],
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": ["OZ"],
            "customer_id": ["Z"],
            "order_date": [AS_OF + pd.Timedelta(days=3)],
        }
    )
    labels = build_churn_labels(customers, AS_OF, HORIZON, orders)
    assert int(labels["churn"].iloc[0]) == 0
    assert int(labels["ordered_in_outcome_window"].iloc[0]) == 1
