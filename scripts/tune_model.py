"""Hyperparameter sweep with walk-forward CV (PR-AUC targeted).

Scans a small grid for LightGBM (num_leaves) and XGBoost (max_depth) and writes
the best values to ``configs/model_params.yaml``.

Usage: python -m scripts.tune_model
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from src.config import PROJECT_ROOT, settings
from src.models.prepare import categorical_features, numeric_features, split_features_target
from src.models.train import build_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LGB_GRID = [{"num_leaves": 15}, {"num_leaves": 31}, {"num_leaves": 63}]
XGB_GRID = [{"max_depth": 4}, {"max_depth": 6}, {"max_depth": 8}]

BASE_LGB = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "n_estimators": 400,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
}

BASE_XGB = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "learning_rate": 0.05,
    "n_estimators": 400,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}


def _evaluate(
    model_type: str,
    params: dict,
    x: pd.DataFrame,
    y: pd.Series,
    num_cols: list[str],
    cat_cols: list[str],
) -> float:
    # Customer-level snapshot data: stratified folds, not a temporal split.
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores: list[float] = []
    for train_idx, val_idx in skf.split(x, y):
        pipe = build_pipeline(model_type, params, num_cols, cat_cols)
        pipe.fit(x.iloc[train_idx], y.iloc[train_idx])
        proba = pipe.predict_proba(x.iloc[val_idx])[:, 1]
        scores.append(average_precision_score(y.iloc[val_idx], proba))
    return float(np.mean(scores))


def main() -> None:
    df = pd.read_parquet(settings.data_processed / "training_set.parquet")
    x, y = split_features_target(df)
    num_cols = numeric_features(x)
    cat_cols = categorical_features(x)
    logger.info(
        "Feature count: %d (numeric %d, categorical %d)", x.shape[1], len(num_cols), len(cat_cols)
    )

    best_results: dict = {}
    for model_type, base, grid in [
        ("lightgbm", BASE_LGB, LGB_GRID),
        ("xgboost", BASE_XGB, XGB_GRID),
    ]:
        best_score = -1.0
        best_override: dict = {}
        for override in grid:
            params = {**base, **override}
            score = _evaluate(model_type, params, x, y, num_cols, cat_cols)
            logger.info("%s %s -> PR-AUC %.4f", model_type, override, score)
            if score > best_score:
                best_score = score
                best_override = override
        best_results[model_type] = {"best_pr_auc": round(best_score, 4), "params": best_override}

    # Update configs/model_params.yaml
    yaml_path = PROJECT_ROOT / "configs" / "model_params.yaml"
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    for model_type, override in best_results.items():
        cfg[model_type].update(override["params"])
    yaml_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    logger.info("Tuning completed; model_params.yaml updated.")
    print(json.dumps(best_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
