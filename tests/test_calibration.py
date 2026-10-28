"""Probability calibration diagnostics tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.models.evaluate import calibration_metrics


def test_calibration_metrics_keys_and_bounds() -> None:
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, 1000))
    score = pd.Series(np.clip(0.2 + 0.6 * y + rng.normal(0, 0.1, 1000), 0.001, 0.999))
    metrics = calibration_metrics(y, score)
    assert 0 <= metrics["brier_score"] <= 1
    assert metrics["log_loss"] >= 0
    assert metrics["calibration_slope"] > 0
    assert "reliability_curve" in metrics
    assert len(metrics["reliability_curve"]["prob_true"]) > 0


def test_perfect_probabilities_have_low_brier() -> None:
    y = pd.Series([0, 0, 1, 1])
    score = pd.Series([0.01, 0.01, 0.99, 0.99])
    metrics = calibration_metrics(y, score)
    assert metrics["brier_score"] < 0.01
    assert metrics["expected_calibration_error"] < 0.1


def test_probabilities_are_clipped_before_log_loss() -> None:
    y = pd.Series([0, 1, 0, 1])
    score = pd.Series([0.0, 1.0, 0.0, 1.0])
    metrics = calibration_metrics(y, score)
    assert np.isfinite(metrics["log_loss"])
