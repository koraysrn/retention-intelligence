"""Model quality regression thresholds on the fixture dataset."""

from __future__ import annotations

import json

from src.config import settings

METRICS = settings.artifacts_dir / "model_ecommerce_ensemble" / "metrics.json"


def _payload() -> dict:
    return json.loads(METRICS.read_text(encoding="utf-8"))


def test_pr_auc_above_floor() -> None:
    payload = _payload()
    assert payload["test_full"]["pr_auc"] >= 0.20


def test_probabilities_are_calibrated_enough() -> None:
    payload = _payload()
    assert payload["test_full"]["brier_score"] <= 0.35
    assert payload["calibration"]["log_loss"] <= 1.0


def test_ci_and_ablation_present() -> None:
    payload = _payload()
    assert payload["ci"]["pr_auc"]["low"] is not None
    assert payload["ci"]["roc_auc"]["high"] is not None
    assert payload["leakage_ablation"]["primary_metric"] == "without_recency_pr_auc"


def test_ensemble_artifact_is_calibrated() -> None:
    payload = _payload()
    assert payload["ensemble"] is True
    assert payload["calibrated"] is True
