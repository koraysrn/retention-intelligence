"""Python <-> dbt parity for as-of features and labels."""

from __future__ import annotations

import json

import duckdb
import pandas as pd
from src.config import PROJECT_ROOT, settings
from src.data.loader import load_ecommerce_data, load_orders
from src.features.ecommerce import (
    FEATURE_COLUMNS,
    build_churn_labels,
    build_customer_features,
)

WAREHOUSE = settings.data_processed / "warehouse.duckdb"
ORDERS = PROJECT_ROOT / "data" / "fixtures" / "orders_sample.csv"


def _metadata() -> dict:
    path = settings.data_processed / "feature_metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def test_dbt_and_python_features_match() -> None:
    meta = _metadata()
    as_of = meta.get("as_of_date", "2025-10-02")
    horizon = int(meta.get("horizon_days", 90))

    py = (
        build_customer_features(
            load_ecommerce_data(settings.ecommerce_data),
            as_of,
            horizon,
            load_orders(ORDERS),
        )
        .sort_values("customer_id")
        .reset_index(drop=True)
    )

    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    dbt = con.execute("select * from main_mart.customer_features order by customer_id").fetchdf()
    con.close()
    py["customer_id"] = py["customer_id"].astype(str)
    dbt["customer_id"] = dbt["customer_id"].astype(str)

    for col in FEATURE_COLUMNS:
        if pd.api.types.is_numeric_dtype(py[col]):
            left = py[col].astype(float).fillna(-999)
            right = dbt[col].astype(float).fillna(-999)
            assert (left - right).abs().max() < 1e-9, f"numeric parity failed: {col}"
        else:
            assert py[col].astype(str).tolist() == dbt[col].astype(str).tolist(), col
    assert py["pre_asof_segment"].tolist() == dbt["pre_asof_segment"].tolist()


def test_dbt_and_python_labels_match() -> None:
    meta = _metadata()
    as_of = meta.get("as_of_date", "2025-10-02")
    horizon = int(meta.get("horizon_days", 90))
    py = (
        build_churn_labels(
            load_ecommerce_data(settings.ecommerce_data),
            as_of,
            horizon,
            load_orders(ORDERS),
        )
        .sort_values("customer_id")
        .reset_index(drop=True)
    )

    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    dbt = con.execute(
        "select customer_id, churn from main_mart.churn_labels order by customer_id"
    ).fetchdf()
    con.close()
    assert dbt["churn"].astype(int).tolist() == py["churn"].astype(int).tolist()
