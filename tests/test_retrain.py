"""Retraining trigger + validation gate tests."""

import sys

import numpy as np
import pandas as pd
from src.monitoring.retrain import check_and_retrain, validate_model


def _frames(drift: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 5000)})
    cur = pd.DataFrame({"x": rng.normal(3 if drift else 0, 1, 5000)})
    return ref, cur


def test_no_drift_no_retrain() -> None:
    ref, cur = _frames(drift=False)
    payload = check_and_retrain(ref, cur, threshold=0.2, dry_run=False)
    assert payload["drift_detected"] is False
    assert payload["retrained"] is False


def test_drift_dry_run_skips_retrain() -> None:
    ref, cur = _frames(drift=True)
    payload = check_and_retrain(ref, cur, threshold=0.2, dry_run=True)
    assert payload["drift_detected"] is True
    assert payload["retrained"] is False
    assert payload.get("retrain_skipped") is True


def test_drift_triggers_retrain_with_mock_command() -> None:
    ref, cur = _frames(drift=True)
    cmd = [sys.executable, "-c", "print('mock train ok')"]
    payload = check_and_retrain(ref, cur, threshold=0.2, dry_run=False, train_command=cmd)
    assert payload["drift_detected"] is True
    assert payload["retrained"] is True
    assert "mock train ok" in payload.get("retrain_tail", "")


def test_validation_gate_accepts_calibrated_ensemble() -> None:
    metrics = {
        "ensemble": True,
        "test_full": {"pr_auc": 0.42, "brier_score": 0.12},
        "calibration": {"calibration_slope": 1.0},
    }
    gate = validate_model(metrics)
    assert gate["passed"] is True


def test_validation_gate_blocks_bad_model() -> None:
    metrics = {
        "ensemble": False,
        "test_full": {"pr_auc": 0.05, "brier_score": 0.80},
        "calibration": {"calibration_slope": 4.0},
    }
    gate = validate_model(metrics)
    assert gate["passed"] is False
    assert gate["checks"]["ensemble"] is False
    assert gate["checks"]["pr_auc"] is False
    assert gate["checks"]["brier"] is False
    assert gate["checks"]["calibration_slope"] is False
