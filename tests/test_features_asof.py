"""Cutoff leakage tests: post-as-of edits must not change features."""

from __future__ import annotations

import pandas as pd
from src.config import PROJECT_ROOT
from src.data.loader import load_ecommerce_data, load_orders
from src.features.ecommerce import build_churn_labels, build_customer_features

AS_OF = pd.Timestamp("2025-10-02")
HORIZON = 90
CUSTOMERS = PROJECT_ROOT / "ecommerce_data.csv"
ORDERS = PROJECT_ROOT / "data" / "fixtures" / "orders_sample.csv"


def _fixture():
    customers = load_ecommerce_data(CUSTOMERS)
    orders = load_orders(ORDERS)
    return customers, orders


def test_post_cutoff_date_edits_do_not_change_features() -> None:
    customers, orders = _fixture()
    base = build_customer_features(customers, AS_OF, HORIZON, orders)

    edited = customers.copy()
    post_cutoff = pd.Timestamp(AS_OF) + pd.Timedelta(days=1)
    # Move every post-cutoff last order/session to a different post-cutoff date.
    order_mask = edited["last_order_date"].notna() & (edited["last_order_date"] > AS_OF)
    session_mask = edited["last_session_date"].notna() & (edited["last_session_date"] > AS_OF)
    edited.loc[order_mask, "last_order_date"] = post_cutoff
    edited.loc[session_mask, "last_session_date"] = post_cutoff

    after = build_customer_features(edited, AS_OF, HORIZON, orders)
    pd.testing.assert_frame_equal(base, after)


def test_outcome_order_only_changes_label_not_features() -> None:
    customers, orders = _fixture()
    base_features = build_customer_features(customers, AS_OF, HORIZON, orders)
    base_labels = build_churn_labels(customers, AS_OF, HORIZON, orders)

    # Find a customer with no order in the outcome window.
    outcome = orders[
        (orders["order_date"] > AS_OF)
        & (orders["order_date"] <= AS_OF + pd.Timedelta(days=HORIZON))
    ]
    candidates = customers.loc[
        ~customers["customer_id"].astype(str).isin(outcome["customer_id"].astype(str)),
        "customer_id",
    ]
    assert not candidates.empty
    cid = str(candidates.iloc[0])

    new_order = pd.DataFrame(
        {
            "order_id": ["LEASE-TEST"],
            "customer_id": [cid],
            "order_date": [AS_OF + pd.Timedelta(days=30)],
            "amount_usd": [100.0],
        }
    )
    orders_after = pd.concat([orders, new_order], ignore_index=True)

    features_after = build_customer_features(customers, AS_OF, HORIZON, orders_after)
    labels_after = build_churn_labels(customers, AS_OF, HORIZON, orders_after)

    pd.testing.assert_frame_equal(base_features, features_after)

    before = int(base_labels.loc[base_labels["customer_id"] == cid, "churn"].iloc[0])
    after = int(labels_after.loc[labels_after["customer_id"] == cid, "churn"].iloc[0])
    assert before == 1
    assert after == 0
