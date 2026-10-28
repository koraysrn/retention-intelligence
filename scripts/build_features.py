"""Phase 1 — as-of feature set and outcome-window label generation.

Reads the customer-level ``ecommerce_data.csv`` plus (when available) the
transaction-level order history, builds a leakage-free feature snapshot at
``--as-of`` and writes the parquet artifacts consumed by training/serving.

Usage:
    python -m scripts.build_features
    python -m scripts.build_features --as-of 2025-10-02 --horizon-days 90
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.config import PROJECT_ROOT, settings
from src.data.loader import load_ecommerce_data, load_orders
from src.features.ecommerce import (
    DEFAULT_HORIZON_DAYS,
    FEATURE_COLUMNS,
    build_dataset,
    resolve_as_of,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ORDERS = PROJECT_ROOT / "data" / "fixtures" / "orders_sample.csv"


def _default_orders_path() -> Path | None:
    for candidate in (settings.data_processed / "orders.parquet", DEFAULT_ORDERS):
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="ecommerce data -> as-of parquet features")
    parser.add_argument("--csv", type=Path, default=settings.ecommerce_data)
    parser.add_argument("--orders", type=Path, default=None)
    parser.add_argument("--as-of", dest="as_of", type=str, default=None)
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("--out", type=Path, default=settings.data_processed)
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {args.csv}. Run `make data` first.")

    logger.info("Loading customer data: %s", args.csv)
    df = load_ecommerce_data(args.csv)

    orders_path = args.orders or _default_orders_path()
    orders = None
    if orders_path is not None and Path(orders_path).exists():
        logger.info("Loading order history: %s", orders_path)
        orders = load_orders(orders_path)
    else:
        logger.warning(
            "No order history found; using aggregate last_order_date as the outcome proxy. "
            "As-of order timing will be conservative. See docs/model_card.md."
        )

    as_of = resolve_as_of(df, args.as_of, args.horizon_days)
    logger.info("As-of date: %s | outcome horizon: %d days", as_of.date(), args.horizon_days)

    dataset = build_dataset(
        df,
        as_of_date=as_of,
        horizon_days=args.horizon_days,
        orders=orders,
    )
    features = dataset["features"]
    labels = dataset["labels"]
    training = dataset["training_set"]

    logger.info("Features: %d rows, %d columns", len(features), features.shape[1])
    logger.info("Labels: %d rows, churn=%s", len(labels), labels["churn"].value_counts().to_dict())

    args.out.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.out / "customer_features.parquet", index=False)
    labels.to_parquet(args.out / "churn_labels.parquet", index=False)
    training.to_parquet(args.out / "training_set.parquet", index=False)

    metadata = {
        "as_of_date": as_of.isoformat(),
        "horizon_days": int(args.horizon_days),
        "label_definition": "churn = 1 iff no purchase in (as_of, as_of + horizon_days]",
        "orders_source": str(orders_path) if orders_path else None,
        "rows": int(len(training)),
        "churn_distribution": {
            str(k): int(v) for k, v in training["churn"].value_counts().sort_index().items()
        },
        "segment_distribution": {
            str(k): int(v)
            for k, v in training["pre_asof_segment"].value_counts().sort_index().items()
        },
        "feature_columns": FEATURE_COLUMNS,
    }
    (args.out / "feature_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    schema = {
        "features": FEATURE_COLUMNS,
        "id": "customer_id",
        "label": "churn",
        "as_of_date": as_of.isoformat(),
        "horizon_days": int(args.horizon_days),
    }
    (settings.artifacts_dir).mkdir(parents=True, exist_ok=True)
    (settings.artifacts_dir / "feature_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )

    logger.info("Files written under %s", args.out)
    logger.info("Churn distribution: %s", metadata["churn_distribution"])
    logger.info("Segments: %s", metadata["segment_distribution"])


if __name__ == "__main__":
    main()
