"""Serving contract: serving_config.json, feature schema and risk boundaries."""

from __future__ import annotations

import json

import pandas as pd
from src.config import settings
from src.features.ecommerce import FEATURE_COLUMNS
from src.serving.batch_score import assign_risk_tier, load_serving_config

CONFIG = settings.artifacts_dir / "model_ecommerce_ensemble" / "serving_config.json"


def test_serving_config_matches_feature_schema() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["feature_schema"] == FEATURE_COLUMNS
    assert 0 <= config["threshold"] <= 1
    assert config["medium_risk"] < config["high_risk"]
    assert config["model_version"]


def test_batch_score_reads_serving_config() -> None:
    config = load_serving_config(CONFIG)
    assert config["model_version"]
    tiers = assign_risk_tier(
        pd.Series([0.0, 0.399, 0.40, 0.699, 0.70, 1.0]),
        config["medium_risk"],
        config["high_risk"],
    )
    assert list(tiers) == ["low", "low", "medium", "medium", "high", "high"]


def test_calibration_and_profit_are_reported() -> None:
    metrics = json.loads(
        (settings.artifacts_dir / "model_ecommerce_ensemble" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert {"brier_score", "log_loss", "calibration_slope"}.issubset(metrics["calibration"])
    assert {
        "profit_at_serving_threshold",
        "optimal_profit_on_holdout_curve",
        "curve",
    }.issubset(metrics["profit"])
    assert 0 <= metrics["threshold"] <= 1
    assert "pr_auc" in metrics["ci"] and "roc_auc" in metrics["ci"]
