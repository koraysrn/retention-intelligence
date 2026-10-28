"""Leakage prevention tests for the model feature matrix."""

from __future__ import annotations

import json

import pandas as pd
from src.config import PROJECT_ROOT, settings
from src.models.prepare import split_features_target

TRAINING = settings.data_processed / "training_set.parquet"
METRICS = settings.artifacts_dir / "model_ecommerce_ensemble" / "metrics.json"

LEAKY = {
    "total_orders",
    "total_spend_usd",
    "avg_order_value",
    "avg_discount_pct",
    "order_span_days",
    "clv_tier",
    "is_repeat_customer",
    "ordered_in_outcome_window",
}


def test_leaky_columns_never_enter_features() -> None:
    df = pd.read_parquet(TRAINING)
    x, y = split_features_target(df)
    assert not (LEAKY & set(x.columns))
    assert "churn" not in x.columns
    assert len(x) == len(y)


def test_no_merge_suffixed_audit_columns() -> None:
    df = pd.read_parquet(TRAINING)
    x, _ = split_features_target(df)
    assert not [c for c in x.columns if c.startswith("ordered_in_outcome_window")]
    assert not [c for c in x.columns if c.startswith("churn_")]


def test_metrics_report_recency_ablation() -> None:
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    ablation = payload["leakage_ablation"]
    assert {"with_recency_pr_auc", "without_recency_pr_auc", "delta", "verdict"}.issubset(ablation)
    assert ablation["primary_metric"] == "without_recency_pr_auc"


def test_metrics_record_asof_and_horizon() -> None:
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    assert payload["as_of_date"]
    assert int(payload["horizon_days"]) > 0
    assert payload["label_definition"].startswith("churn = 1 iff no purchase")
    assert (PROJECT_ROOT / "artifacts" / "feature_schema.json").exists()
