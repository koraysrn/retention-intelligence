"""MLflow fast-fallback tests (no live server required)."""

from __future__ import annotations

from src.utils.mlflow_logger import log_to_mlflow_safely, mlflow_reachable


def test_unreachable_server_is_detected_fast() -> None:
    assert mlflow_reachable("http://127.0.0.1:1", timeout=0.05) is False


def test_logging_returns_false_without_server() -> None:
    ok = log_to_mlflow_safely(
        run_name="test",
        experiment="test",
        params={"a": 1},
        metrics={"pr_auc": 0.5},
        tracking_uri="http://127.0.0.1:1",
    )
    assert ok is False
