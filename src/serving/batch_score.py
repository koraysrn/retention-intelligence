"""Daily batch risk scoring pipeline.

Flow:
1. Load the persisted calibrated ensemble (joblib).
2. Read ``serving_config.json`` (thresholds, model version, feature schema).
3. Compute churn probability + risk tier.
4. Write scores.parquet including ``model_version`` for traceability.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import cast

import pandas as pd

from src.config import settings

logger = logging.getLogger(__name__)

ARTIFACT_DIR = settings.artifacts_dir / "model_ecommerce_ensemble"
DEFAULT_MODEL_PATH = ARTIFACT_DIR / "pipeline.joblib"
DEFAULT_SERVING_CONFIG = ARTIFACT_DIR / "serving_config.json"

DEFAULT_MEDIUM_RISK = 0.40
DEFAULT_HIGH_RISK = 0.70


def load_serving_config(path: Path | str | None = None) -> dict:
    """Load the serving contract; fall back to config defaults when absent."""
    path = Path(path or DEFAULT_SERVING_CONFIG)
    if path.exists():
        return cast(dict, json.loads(path.read_text(encoding="utf-8")))
    return {
        "model_version": "unknown",
        "threshold": 0.50,
        "medium_risk": DEFAULT_MEDIUM_RISK,
        "high_risk": DEFAULT_HIGH_RISK,
        "feature_schema": [],
    }


def assign_risk_tier(
    prob: pd.Series,
    medium_risk: float | None = None,
    high_risk: float | None = None,
) -> pd.Series:
    """Map calibrated probabilities to risk tiers.

    Boundaries are left-closed: ``0.40 -> medium`` and ``0.70 -> high``.
    """
    medium = DEFAULT_MEDIUM_RISK if medium_risk is None else float(medium_risk)
    high = DEFAULT_HIGH_RISK if high_risk is None else float(high_risk)
    if not 0 <= medium < high <= 1:
        raise ValueError("Risk thresholds must satisfy 0 <= medium < high <= 1")
    return pd.cut(
        prob,
        bins=[-float("inf"), medium, high, float("inf")],
        labels=["low", "medium", "high"],
        right=False,
    )


def load_model(model_path: Path | str | None = None):
    """Load the persisted sklearn pipeline."""
    from joblib import load

    path = Path(model_path or DEFAULT_MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return load(path)


def score_customers(
    features: pd.DataFrame,
    model=None,
    serving_config: dict | None = None,
) -> pd.DataFrame:
    """Produce churn probability, risk tier and model version from features."""
    pipeline = model or load_model()
    config = serving_config or load_serving_config()

    ids = features["customer_id"].astype(str).reset_index(drop=True)
    schema = config.get("feature_schema") or []
    if schema:
        missing = [c for c in schema if c not in features.columns]
        if missing:
            raise ValueError(f"Feature schema mismatch; missing columns: {missing}")
        x = features[schema].copy()
    else:
        x = features.drop(columns=["customer_id"], errors="ignore")

    proba = pipeline.predict_proba(x)[:, 1]
    out = pd.DataFrame({"customer_id": ids, "churn_probability": proba})
    out["risk_tier"] = assign_risk_tier(
        pd.Series(proba), config.get("medium_risk"), config.get("high_risk")
    )
    out["model_version"] = str(config.get("model_version", "unknown"))
    out["threshold"] = float(config.get("threshold", 0.5))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch risk scoring")
    parser.add_argument(
        "--features",
        type=Path,
        default=settings.data_processed / "customer_features.parquet",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--serving-config", type=Path, default=DEFAULT_SERVING_CONFIG)
    parser.add_argument("--out", type=Path, default=settings.data_processed / "scores.parquet")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.features.exists():
        raise FileNotFoundError(f"Feature set not found: {args.features}")

    features = pd.read_parquet(args.features)
    config = load_serving_config(args.serving_config)
    scores = score_customers(features, load_model(args.model), config)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(args.out, index=False)
    logger.info("Scores written: %s (%d customers)", args.out, len(scores))
    logger.info("Model version: %s", config.get("model_version"))
    logger.info("Risk distribution:\n%s", scores["risk_tier"].value_counts().to_dict())
    logger.info("Mean churn probability: %.4f", scores["churn_probability"].mean())


if __name__ == "__main__":
    main()
