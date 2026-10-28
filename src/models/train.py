"""Model factory, preprocessing helpers and the single training entry point.

The project has **one** training protocol: ``scripts.train_ensemble``. This
module exposes reusable model/preprocessing utilities and delegates ``main()``
to the ensemble trainer so there is no second, divergent training path.

Model availability:
- LightGBM and XGBoost are used when installed.
- CatBoost is used when installed; otherwise it falls back to scikit-learn's
  ``HistGradientBoostingClassifier`` (same 3-model ensemble shape, no crash).
- Logistic Regression is the interpretable baseline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import settings
from src.models.prepare import build_preprocessor

logger = logging.getLogger(__name__)

MODEL_NAME = "churn_classifier"

DEFAULT_PARAMS: dict[str, dict] = {
    "logistic": {"max_iter": 5000},
    "lightgbm": {
        "objective": "binary",
        "metric": "average_precision",
        "learning_rate": 0.05,
        "n_estimators": 300,
        "num_leaves": 31,
        "random_state": 42,
    },
    "xgboost": {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.05,
        "n_estimators": 300,
        "max_depth": 6,
        "random_state": 42,
    },
    "catboost": {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "learning_rate": 0.05,
        "iterations": 300,
        "depth": 6,
        "random_seed": 42,
        "verbose": 0,
        "allow_writing_files": False,
    },
    "hist": {
        "learning_rate": 0.06,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
        "random_state": 42,
    },
}


def _has_module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def available_model_types() -> list[str]:
    """Return the 3-model ensemble members available in this environment."""
    models = ["lightgbm", "xgboost"]
    models.append("catboost" if _has_module("catboost") else "hist")
    return models


def make_model(model_type: str, params: dict | None = None):
    """Instantiate a classifier for the given model type."""
    p = {**DEFAULT_PARAMS.get(model_type, {}), **(params or {})}

    if model_type == "logistic":
        return LogisticRegression(max_iter=p.pop("max_iter", 2000))
    if model_type == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(**p)
    if model_type == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(**p)
    if model_type == "catboost":
        if _has_module("catboost"):
            import catboost as cat

            return cat.CatBoostClassifier(**p)
        logger.warning("catboost is not installed on this Python; using HistGradientBoosting")
        return HistGradientBoostingClassifier(**DEFAULT_PARAMS["hist"])
    if model_type == "hist":
        return HistGradientBoostingClassifier(**p)
    raise ValueError(f"Unknown model type: {model_type}")


def build_pipeline(
    model_type: str,
    params: dict | None = None,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> Pipeline:
    """Build an sklearn Pipeline composed of preprocessing + classifier."""
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model = make_model(model_type, params)
    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])


def load_training_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load the Phase 1 training set."""
    path = Path(path or settings.data_processed / "training_set.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"Training set not found: {path}. Run `python -m scripts.build_features` first."
        )
    df = pd.read_parquet(path)
    logger.info("Training data loaded: %d rows, %d columns", len(df), df.shape[1])
    return df


def walk_forward_cv(
    x: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    n_splits: int = 5,
    params: dict | None = None,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> list[dict]:
    """Stratified K-Fold cross-validation (retained name for compatibility).

    This data is a **customer-level snapshot**, so a temporal walk-forward split
    is not meaningful and would create label shift. The single evaluation
    protocol uses stratified folds; the function name is kept to avoid breaking
    external imports.
    """
    from sklearn.model_selection import StratifiedKFold

    from src.models.evaluate import evaluate_binary, full_metrics

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(x, y)):
        pipe = build_pipeline(model_type, params, numeric_cols, categorical_cols)
        pipe.fit(x.iloc[train_idx], y.iloc[train_idx])
        proba = pipe.predict_proba(x.iloc[val_idx])[:, 1]
        y_val = y.iloc[val_idx]
        y_score = pd.Series(proba, index=y_val.index)

        fold_result = {"fold": fold}
        fold_result.update(full_metrics(y_val, y_score))
        fold_result.update(evaluate_binary(y_val, y_score, threshold=0.5))
        results.append(fold_result)

    return results


def _shap_explainer(
    model: object, background: np.ndarray, feature_names: list[str], model_type: str
):
    """Build the appropriate SHAP explainer for the given model type."""
    import shap

    if model_type == "logistic":
        return shap.LinearExplainer(model, background, feature_names=feature_names)

    if model_type == "lightgbm" and hasattr(model, "booster_"):
        model = model.booster_
    elif model_type == "xgboost" and hasattr(model, "get_booster"):
        model = model.get_booster()

    return shap.TreeExplainer(
        model,
        background,
        feature_names=feature_names,
        feature_perturbation="tree_path_dependent",
    )


def compute_shap_importance(
    pipeline: Pipeline, x: pd.DataFrame, model_type: str = "lightgbm"
) -> pd.DataFrame:
    """Compute feature importances from mean |SHAP| values."""
    pre = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]

    x_t = pre.transform(x)
    feature_names = list(pre.get_feature_names_out())
    background = x_t[: min(100, len(x_t))]
    explainer = _shap_explainer(clf, background, feature_names, model_type)
    shap_values = explainer.shap_values(x_t)

    vals = np.asarray(shap_values[-1]) if isinstance(shap_values, list) else np.asarray(shap_values)
    if vals.ndim == 3:
        vals = vals[..., 1]

    importance = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": np.abs(vals).mean(axis=0)}
    )
    return importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def main() -> None:
    """Single training entry point: delegate to the calibrated ensemble trainer."""
    from scripts.train_ensemble import main as ensemble_main

    ensemble_main()


if __name__ == "__main__":
    main()
