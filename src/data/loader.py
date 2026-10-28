"""Loading and DuckDB ingestion for ``ecommerce_data.csv``.

The source is the customer-level ``ecommerce_data.csv`` at the project root. In
an enterprise deployment this module is replaced by connectors that read from
Snowflake/BigQuery; DuckDB + CSV is sufficient for the prototype.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.features.ecommerce import RAW_DATE_COLUMNS

logger = logging.getLogger(__name__)

# Explicit schema for DuckDB ``read_csv``. Dates are read as VARCHAR and cast to
# TIMESTAMP in the dbt/SQL layer (to preserve the ISO-8601 format in the CSV).
ECOMMERCE_COLUMNS: dict[str, str] = {
    "customer_id": "BIGINT",
    "name": "VARCHAR",
    "email": "VARCHAR",
    "country": "VARCHAR",
    "age": "BIGINT",
    "age_group": "VARCHAR",
    "signup_date": "VARCHAR",
    "marketing_opt_in": "BOOLEAN",
    "total_orders": "BIGINT",
    "total_spend_usd": "DOUBLE",
    "avg_order_value": "DOUBLE",
    "avg_discount_pct": "DOUBLE",
    "first_order_date": "VARCHAR",
    "last_order_date": "VARCHAR",
    "preferred_payment": "VARCHAR",
    "preferred_device_ord": "VARCHAR",
    "preferred_source": "VARCHAR",
    "top_category_bought": "VARCHAR",
    "avg_rating_given": "DOUBLE",
    "total_sessions": "BIGINT",
    "preferred_device_sess": "VARCHAR",
    "preferred_source_sess": "VARCHAR",
    "first_session_date": "VARCHAR",
    "last_session_date": "VARCHAR",
    "has_abandoned_cart": "BIGINT",
    "clv_tier": "VARCHAR",
    "is_repeat_customer": "BIGINT",
}


def load_orders(path: Path | str) -> pd.DataFrame:
    """Load a transaction-level order history (Parquet or CSV)."""
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    if "customer_id" in df.columns:
        df["customer_id"] = df["customer_id"].astype(str)
    logger.info("Loaded order history: %d rows", len(df))
    return df


def load_ecommerce_data(csv_path: Path | str) -> pd.DataFrame:
    """Load ``ecommerce_data.csv`` with pandas and normalize the column types.

    - Date columns -> datetime (invalid values become NaT)
    - ``marketing_opt_in`` -> bool
    - ``customer_id`` -> str (used as the primary key)
    """
    df = pd.read_csv(csv_path)
    for col in RAW_DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "marketing_opt_in" in df.columns:
        df["marketing_opt_in"] = df["marketing_opt_in"].astype(bool)
    if "customer_id" in df.columns:
        df["customer_id"] = df["customer_id"].astype(str)
    logger.info("Loaded e-commerce data: %d rows, %d columns", len(df), df.shape[1])
    return df


def ingest_orders_to_duckdb(
    csv_path: Path | str,
    db_path: Path | str,
    table: str = "orders",
) -> None:
    """Load a transaction-level order history into DuckDB."""
    import duckdb

    path = Path(csv_path)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    csv_literal = path.resolve().as_posix()
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT
            order_id,
            CAST(customer_id AS BIGINT) AS customer_id,
            CAST(order_date AS TIMESTAMP) AS order_date,
            CAST(amount_usd AS DOUBLE) AS amount_usd,
            CAST(discount_pct AS DOUBLE) AS discount_pct,
            channel,
            device,
            category
        FROM read_csv('{csv_literal}', header = true)
        """
    )
    row = con.execute(f"SELECT count(*) FROM {table}").fetchone()
    con.close()
    logger.info("Written to DuckDB: %s -> %s (%s rows)", table, db_path, row[0] if row else 0)


def ingest_ecommerce_to_duckdb(
    csv_path: Path | str,
    db_path: Path | str,
    table: str = "customers",
) -> None:
    """Load ``ecommerce_data.csv`` into DuckDB using an explicit schema.

    Empty strings are converted to NULL; date columns are read as VARCHAR and
    cast to ``TIMESTAMP`` in the dbt layer.
    """
    import duckdb

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    columns_literal = ", ".join(f"'{name}': '{dtype}'" for name, dtype in ECOMMERCE_COLUMNS.items())
    csv_literal = Path(csv_path).resolve().as_posix()
    sql = f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT
            customer_id,
            name,
            email,
            country,
            age,
            age_group,
            signup_date,
            marketing_opt_in,
            total_orders,
            total_spend_usd,
            avg_order_value,
            avg_discount_pct,
            first_order_date,
            last_order_date,
            preferred_payment,
            preferred_device_ord,
            preferred_source,
            top_category_bought,
            avg_rating_given,
            total_sessions,
            preferred_device_sess,
            preferred_source_sess,
            first_session_date,
            last_session_date,
            has_abandoned_cart,
            clv_tier,
            is_repeat_customer
        FROM read_csv(
            '{csv_literal}',
            header = true,
            columns = {{ {columns_literal} }}
        )
    """
    con = duckdb.connect(str(db_path))
    con.execute(sql)
    row = con.execute(f"SELECT count(*) FROM {table}").fetchone()
    count = int(row[0]) if row is not None else 0
    con.close()
    logger.info("Written to DuckDB: %s -> %s (%d rows)", table, db_path, count)
