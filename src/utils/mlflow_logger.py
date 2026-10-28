"""MLflow logging with a fast, non-blocking fallback.

The prototype must never stall or fail because an optional MLflow server is
absent. This module probes the tracking URI with a short TCP timeout and
returns ``False`` instead of raising when logging is unavailable. Local
versioned artifacts remain the source of truth.
"""

from __future__ import annotations

import logging
import socket
from pathlib import Path
from urllib.parse import urlparse

from src.config import settings

logger = logging.getLogger(__name__)


def mlflow_reachable(uri: str, timeout: float = 1.0) -> bool:
    """Return True only when a TCP connection to the tracking server succeeds."""
    parsed = urlparse(uri)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def log_to_mlflow_safely(
    run_name: str,
    experiment: str,
    params: dict,
    metrics: dict,
    artifacts: list[Path] | None = None,
    model=None,
    input_example=None,
    tags: dict | None = None,
    tracking_uri: str | None = None,
) -> bool:
    """Log params/metrics/artifacts/model to MLflow if reachable.

    Returns:
        True when logging completed, False when MLflow is unavailable or fails.
    """
    uri = tracking_uri or settings.mlflow_tracking_uri
    if not mlflow_reachable(uri):
        logger.warning("MLflow server unreachable; local artifacts remain authoritative")
        return False

    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.models import infer_signature

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: v for k, v in params.items() if v is not None})
            mlflow.log_metrics(
                {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            )
            for artifact in artifacts or []:
                if Path(artifact).exists():
                    mlflow.log_artifact(str(artifact))
            if model is not None and input_example is not None:
                predictions = model.predict_proba(input_example)[:, 1]
                signature = infer_signature(input_example, predictions)
                mlflow.sklearn.log_model(model, "model", signature=signature)
            for key, value in (tags or {}).items():
                mlflow.set_tag(key, value)
        logger.info("MLflow logging completed: %s", run_name)
        return True
    except Exception as exc:  # noqa: BLE001 - optional external service
        logger.warning("MLflow logging skipped: %s", exc)
        return False
