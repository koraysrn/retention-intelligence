"""dbt mart-layer integration + Python parity tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest
from src.config import PROJECT_ROOT, settings
from src.data.loader import load_ecommerce_data, load_orders
from src.features.ecommerce import build_churn_labels

WAREHOUSE = settings.data_processed / "warehouse.duckdb"


@pytest.fixture(scope="module")
def warehouse() -> Path:
    """Ensure the fixture warehouse + dbt marts exist (self-contained tests)."""
    if not WAREHOUSE.exists():
        subprocess.run([sys.executable, "-m", "scripts.ingest"], cwd=PROJECT_ROOT, check=True)
    dbt = Path(sys.executable).parent / "dbt"
    if dbt.exists():
        subprocess.run(
            [str(dbt), "build", "--project-dir", "dbt", "--profiles-dir", "dbt"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    if not WAREHOUSE.exists():
        pytest.skip("dbt unavailable and warehouse could not be built")
    return WAREHOUSE


def _connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(WAREHOUSE), read_only=True)


def test_raw_customers_row_count(warehouse: Path) -> None:
    n_expected = len(load_ecommerce_data(settings.ecommerce_data))
    con = _connect()
    n = con.execute("SELECT count(*) FROM main.customers").fetchone()[0]
    con.close()
    assert n == n_expected


def test_mart_customer_features_schema(warehouse: Path) -> None:
    con = _connect()
    df = con.execute("SELECT * FROM main_mart.customer_features").fetchdf()
    con.close()

    expected_cols = {
        "customer_id",
        "tenure_days",
        "recency_days",
        "has_abandoned_cart",
        "has_purchase",
        "country",
        "pre_asof_segment",
    }
    assert expected_cols.issubset(set(df.columns))
    assert df["customer_id"].is_unique
    assert not df["tenure_days"].isna().any()
    assert len(df) == len(load_ecommerce_data(settings.ecommerce_data))


def test_mart_churn_labels_binary_and_consistent(warehouse: Path) -> None:
    con = _connect()
    labels = con.execute("SELECT * FROM main_mart.churn_labels").fetchdf()
    con.close()

    assert labels["customer_id"].is_unique
    assert set(labels["churn"].unique()).issubset({0, 1})
    assert labels["churn"].nunique() == 2
    assert len(labels) == len(load_ecommerce_data(settings.ecommerce_data))


def test_churn_label_matches_python_definition(warehouse: Path) -> None:
    meta_path = settings.data_processed / "feature_metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    as_of = metadata.get("as_of_date", "2025-10-02")
    horizon = int(metadata.get("horizon_days", 90))
    orders_path = PROJECT_ROOT / "data" / "fixtures" / "orders_sample.csv"

    con = _connect()
    dbt_labels = con.execute(
        "SELECT customer_id, churn FROM main_mart.churn_labels ORDER BY customer_id"
    ).fetchdf()
    con.close()

    py_labels = build_churn_labels(
        load_ecommerce_data(settings.ecommerce_data),
        as_of,
        horizon,
        load_orders(orders_path) if orders_path.exists() else None,
    )
    py_labels["customer_id"] = py_labels["customer_id"].astype("int64")
    py_labels = py_labels.sort_values("customer_id").reset_index(drop=True)

    assert dbt_labels["churn"].tolist() == py_labels["churn"].tolist()


def test_parquet_outputs_exist(warehouse: Path) -> None:
    for name in ("customer_features.parquet", "churn_labels.parquet", "training_set.parquet"):
        assert (settings.data_processed / name).exists()
