"""Phase 1 — Ingestion: load ecommerce_data.csv into DuckDB in raw form.

Usage: python -m scripts.ingest
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import PROJECT_ROOT, settings
from src.data.loader import ingest_ecommerce_to_duckdb, ingest_orders_to_duckdb

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CSV = settings.ecommerce_data


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV -> DuckDB raw ingestion")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--orders",
        type=Path,
        default=PROJECT_ROOT / "data" / "fixtures" / "orders_sample.csv",
    )
    parser.add_argument("--db", type=Path, default=settings.data_processed / "warehouse.duckdb")
    parser.add_argument("--table", default="customers")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {args.csv}")

    logger.info("Ingestion started: %s -> %s (%s)", args.csv, args.db, args.table)
    ingest_ecommerce_to_duckdb(args.csv, args.db, args.table)
    if args.orders.exists():
        ingest_orders_to_duckdb(args.orders, args.db, table="orders")
    else:
        logger.warning("Order fixture not found: %s; dbt order features will be empty", args.orders)
    logger.info("Ingestion completed.")


if __name__ == "__main__":
    main()
