"""Model evaluation metric tests."""

from __future__ import annotations

import pandas as pd
from src.models.evaluate import (
    calibration_metrics,
    evaluate_binary,
    find_optimal_threshold,
    full_metrics,
    lift_at_top_decile,
    profit_curve,
    summarize_cv,
)


def test_evaluate_binary_perfect_classifier() -> None:
    y_true = pd.Series([0, 1, 0, 1, 1])
    y_score = pd.Series([0.1, 0.9, 0.2, 0.8, 0.95])
    m = evaluate_binary(y_true, y_score, threshold=0.5)
    assert m["pr_auc"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_evaluate_binary_random_scores() -> None:
    y_true = pd.Series([0, 1, 0, 1, 0, 1])
    y_score = pd.Series([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    m = evaluate_binary(y_true, y_score, threshold=0.5)
    assert 0.0 <= m["roc_auc"] <= 1.0


def test_profit_curve_known_small_example() -> None:
    """Hand-checkable example: n=2, one positive, one negative."""
    y_true = pd.Series([0, 1])
    y_score = pd.Series([0.2, 0.8])
    curve = profit_curve(y_true, y_score, ltv=100.0, incentive_cost=10.0, uplift=1.0)
    assert {"threshold", "profit", "precision", "recall", "tp", "fp", "fn"}.issubset(curve.columns)
    # At threshold 0.8: tp=1, fp=0, fn=0 -> profit 100.
    row = curve.loc[curve["threshold"] == 0.8].iloc[0]
    assert row["tp"] == 1
    assert row["fp"] == 0
    assert row["fn"] == 0
    assert row["profit"] == 100.0
    # At threshold 0.2: tp=1, fp=1, fn=0 -> profit 100 - 10 = 90.
    row = curve.loc[curve["threshold"] == 0.2].iloc[0]
    assert row["fp"] == 1
    assert row["profit"] == 90.0
    # Explicit "contact nobody" endpoint exists.
    assert curve["threshold"].max() == 1.0


def test_optimal_threshold_matches_hand_example() -> None:
    y_true = pd.Series([0, 1])
    y_score = pd.Series([0.2, 0.8])
    assert find_optimal_threshold(y_true, y_score, 100.0, 10.0, 1.0) == 0.8


def test_find_optimal_threshold_between_zero_and_one() -> None:
    y_true = pd.Series([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    y_score = pd.Series([0.1, 0.9, 0.15, 0.85, 0.7, 0.3, 0.6, 0.2, 0.8, 0.95])
    threshold = find_optimal_threshold(y_true, y_score, ltv=100.0, incentive_cost=10.0)
    assert 0.0 <= threshold <= 1.0


def test_lift_top_decile_greater_than_one_for_good_model() -> None:
    y_true = pd.Series([1] * 10 + [0] * 90)
    y_score = pd.Series([0.9] * 10 + [0.1] * 90)
    lift = lift_at_top_decile(y_true, y_score)
    assert lift > 1.0


def test_full_metrics_keys_include_calibration() -> None:
    y_true = pd.Series([0, 1, 0, 1])
    y_score = pd.Series([0.1, 0.8, 0.3, 0.7])
    m = full_metrics(y_true, y_score)
    assert {
        "pr_auc",
        "roc_auc",
        "lift_top_decile",
        "brier_score",
        "log_loss",
        "n",
        "pos_rate",
    }.issubset(m)
    assert m["n"] == 4


def test_calibration_metrics_shape() -> None:
    y_true = pd.Series([0, 1, 0, 1, 1, 0])
    y_score = pd.Series([0.1, 0.8, 0.2, 0.7, 0.9, 0.3])
    metrics = calibration_metrics(y_true, y_score, n_bins=5)
    assert {"brier_score", "log_loss", "calibration_slope", "reliability_curve"}.issubset(metrics)


def test_summarize_cv() -> None:
    metrics = [
        {"fold": 0, "pr_auc": 0.5, "roc_auc": 0.6},
        {"fold": 1, "pr_auc": 0.7, "roc_auc": 0.8},
    ]
    summary = summarize_cv(metrics)
    assert summary["pr_auc"]["mean"] == 0.6
    assert summary["roc_auc"]["mean"] == 0.7
