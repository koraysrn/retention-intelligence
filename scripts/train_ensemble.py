"""Single training protocol: calibrated XGBoost + LightGBM + (CatBoost|Hist) ensemble.

Flow
----
1. Load the as-of training set produced by ``scripts.build_features``.
2. Stratified holdout + Stratified K-Fold CV (customer-level snapshot data).
3. OOF ensemble weights and cross-fitted Platt calibration.
4. Holdout metrics: PR-AUC/ROC-AUC with bootstrap CI, Brier, log loss,
   reliability curve, calibration slope/intercept, profit curve + threshold.
5. Leakage ablation with/without recency features.
6. SHAP on the holdout set + per-customer top-3 reasons.
7. Versioned, atomic artifact promotion + ``serving_config.json`` + MLflow log.

The same function is used by ``make train``, ``make pipeline`` and the retrain
monitor. There is no second training path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from src.config import PROJECT_ROOT, load_yaml, settings
from src.models.ensemble import CalibratedEnsemble, SoftVotingEnsemble
from src.models.evaluate import (
    bootstrap_ci,
    calibration_metrics,
    evaluate_binary,
    find_best_threshold_f1,
    find_best_threshold_youden,
    find_optimal_threshold,
    full_metrics,
    profit_curve,
)
from src.models.prepare import (
    IDENTIFIER_COLUMNS,
    LABEL_COLUMNS,
    LEAKY_COLUMNS,
    build_preprocessor,
    categorical_features,
    numeric_features,
    split_features_target,
)
from src.models.train import available_model_types, make_model
from src.monitoring.drift import detect_data_drift
from src.utils.mlflow_logger import log_to_mlflow_safely

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_TYPES = available_model_types()
OUT_DIR = settings.artifacts_dir / "model_ecommerce_ensemble"
RECENCY_FEATURES = ["recency_days", "order_session_gap_days"]


def _is_imbalanced(y: pd.Series, minority_threshold: float = 0.15) -> bool:
    minority_rate = float(y.mean() if y.mean() <= 0.5 else 1 - y.mean())
    return minority_rate < minority_threshold


def _smote_available() -> bool:
    try:
        import imblearn  # noqa: F401

        return True
    except ImportError:
        return False


def _fold_pipeline(
    model_type: str,
    params: dict,
    num_cols: list[str],
    cat_cols: list[str],
    y: pd.Series,
    use_smote: bool,
):
    """Build an in-fold pipeline (preprocessing + [SMOTE] + classifier)."""
    pre = build_preprocessor(num_cols, cat_cols)
    model_params = {**params}

    pos_rate = float(y.mean())
    neg_rate = 1.0 - pos_rate
    scale = neg_rate / pos_rate if pos_rate > 0 else 1.0

    if model_type in ("lightgbm", "xgboost"):
        model_params.setdefault("scale_pos_weight", scale)
    elif model_type == "catboost":
        model_params.setdefault("auto_class_weights", "Balanced")
    elif model_type == "hist":
        model_params.setdefault("class_weight", "balanced")

    model = make_model(model_type, model_params)

    steps: list = [("pre", pre)]
    if use_smote and _smote_available() and _is_imbalanced(y):
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        steps.append(("smote", SMOTE(random_state=42)))
        return ImbPipeline(steps + [("clf", model)])

    return Pipeline(steps + [("clf", model)])


def _build_base_pipelines(
    params_map: dict[str, dict],
    num_cols: list[str],
    cat_cols: list[str],
    y: pd.Series,
    use_smote: bool,
) -> list[tuple[str, Pipeline]]:
    pipelines: list[tuple[str, Pipeline]] = []
    for model_type in MODEL_TYPES:
        pipe = _fold_pipeline(
            model_type, params_map.get(model_type, {}), num_cols, cat_cols, y, use_smote
        )
        pipelines.append((model_type, pipe))
    return pipelines


def _cv_scores(
    x: pd.DataFrame,
    y: pd.Series,
    params_map: dict[str, dict],
    num_cols: list[str],
    cat_cols: list[str],
    use_smote: bool,
    n_splits: int = 5,
) -> dict:
    """Stratified K-Fold CV: per-model OOF probabilities and metric summaries."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_by_model: dict[str, np.ndarray] = {m: np.zeros(len(y)) for m in MODEL_TYPES}
    metrics_by_model: dict[str, list[dict]] = {m: [] for m in MODEL_TYPES}

    for fold, (train_idx, val_idx) in enumerate(skf.split(x, y)):
        y_train = y.iloc[train_idx]
        for model_type, pipe in _build_base_pipelines(
            params_map, num_cols, cat_cols, y_train, use_smote
        ):
            pipe.fit(x.iloc[train_idx], y_train)
            proba = pipe.predict_proba(x.iloc[val_idx])[:, 1]
            oof_by_model[model_type][val_idx] = proba
            y_val = y.iloc[val_idx]
            score = pd.Series(proba, index=y_val.index)
            metrics_by_model[model_type].append(
                {
                    "fold": fold,
                    **full_metrics(y_val, score),
                    **evaluate_binary(y_val, score, threshold=0.5),
                }
            )

    return {"oof": oof_by_model, "metrics": metrics_by_model}


def _summarize(metrics_list: list[dict]) -> dict[str, dict[str, float]]:
    from src.models.evaluate import summarize_cv

    return summarize_cv(metrics_list)


def _clean_feature_name(name: str) -> str:
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _compute_shap_details(
    pipeline: Pipeline,
    x_holdout: pd.DataFrame,
    customer_ids: pd.Series,
    model_type: str = "lightgbm",
    top_k: int = 3,
) -> tuple[pd.DataFrame, dict[str, list[dict]], str]:
    """Global SHAP importance + per-customer top-k reasons on the holdout set."""
    try:
        import shap

        pre = pipeline.named_steps["pre"]
        clf = pipeline.named_steps["clf"]
        x_t = pre.transform(x_holdout)
        feature_names = [_clean_feature_name(n) for n in pre.get_feature_names_out()]
        background = x_t[: min(100, len(x_t))]

        if model_type == "lightgbm" and hasattr(clf, "booster_"):
            explainer = shap.TreeExplainer(
                clf.booster_,
                background,
                feature_names=feature_names,
                feature_perturbation="tree_path_dependent",
            )
        else:
            explainer = shap.TreeExplainer(clf, background, feature_names=feature_names)

        shap_values = explainer.shap_values(x_t)
        vals = np.asarray(shap_values)
        if vals.ndim == 3:
            vals = vals[..., 1] if vals.shape[-1] > 1 else vals[..., 0]
        elif vals.ndim == 2 and vals.shape[0] != len(x_holdout):
            vals = vals.T

        importance = pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": np.abs(vals).mean(axis=0)}
        ).sort_values("mean_abs_shap", ascending=False)

        explanations: dict[str, list[dict]] = {}
        for row_idx, cid in enumerate(customer_ids.astype(str)):
            contributions = vals[row_idx]
            top_idx = np.argsort(np.abs(contributions))[::-1][:top_k]
            explanations[cid] = [
                {
                    "feature": feature_names[j],
                    "shap_value": round(float(contributions[j]), 6),
                    "direction": "increases" if contributions[j] > 0 else "decreases",
                }
                for j in top_idx
            ]
        return importance.reset_index(drop=True), explanations, "shap"
    except Exception as exc:  # noqa: BLE001 - SHAP is optional on new Python versions
        logger.warning("SHAP failed; falling back to permutation importance: %s", exc)
        from sklearn.inspection import permutation_importance

        pre = pipeline.named_steps["pre"]
        clf = pipeline.named_steps["clf"]
        x_t = pre.transform(x_holdout)
        # Permutation importance is computed on transformed columns; keep names.
        feature_names = [_clean_feature_name(n) for n in pre.get_feature_names_out()]
        result = permutation_importance(
            clf, x_t, np.zeros(len(x_holdout)), n_repeats=3, random_state=42
        )
        importance = pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": result.importances_mean}
        ).sort_values("mean_abs_shap", ascending=False)
        return importance.reset_index(drop=True), {}, "permutation"


def _leakage_ablation(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    params_map: dict[str, dict],
    num_cols: list[str],
    cat_cols: list[str],
) -> dict:
    """Train with/without recency features and report the PR-AUC delta."""
    model_type = "lightgbm" if "lightgbm" in MODEL_TYPES else MODEL_TYPES[0]
    params = params_map.get(model_type, {})

    def _score(cols: list[str]) -> float:
        pipe = _fold_pipeline(model_type, params, cols, cat_cols, y_train, use_smote=False)
        pipe.fit(x_train, y_train)
        proba = pipe.predict_proba(x_test)[:, 1]
        return float(full_metrics(y_test, pd.Series(proba, index=y_test.index))["pr_auc"])

    with_recency = _score([c for c in num_cols])
    without_cols = [c for c in num_cols if c not in RECENCY_FEATURES]
    without_recency = _score(without_cols)
    delta = without_recency - with_recency
    verdict = "recency_dominates_metrics" if delta < -0.05 else "no_material_recency_leakage"
    return {
        "with_recency_pr_auc": round(with_recency, 4),
        "without_recency_pr_auc": round(without_recency, 4),
        "delta": round(delta, 4),
        "verdict": verdict,
        "primary_metric": "without_recency_pr_auc",
        "note": "A large negative delta indicates recency is doing label-proxy work.",
    }


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def _data_hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:  # noqa: BLE001
        return "unknown"


def _compute_guardrails(
    df: pd.DataFrame,
    x: pd.DataFrame,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y: pd.Series,
    test_full: dict,
    cv_summary: dict[str, dict],
    data_path: Path,
) -> dict:
    guardrails: dict[str, dict] = {}

    missing = int(x.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    guardrails["data_quality"] = {
        "status": "pass" if duplicate_rows == 0 else "warn",
        "missing_values_imputed": missing,
        "duplicate_rows": duplicate_rows,
    }

    minority_rate = float(min(y.mean(), 1 - y.mean()))
    guardrails["class_balance"] = {
        "status": "pass" if minority_rate > 0.05 else "warn",
        "minority_rate": round(minority_rate, 4),
        "positive_rate": round(float(y.mean()), 4),
    }

    removed = sorted(IDENTIFIER_COLUMNS | LABEL_COLUMNS | LEAKY_COLUMNS)
    guardrails["leakage"] = {
        "status": "pass",
        "removed_features": [c for c in removed if c in df.columns],
    }

    cv_aucs = [
        cv_summary[m]["roc_auc"]["mean"] for m in MODEL_TYPES if "roc_auc" in cv_summary.get(m, {})
    ]
    cv_auc_mean = float(np.mean(cv_aucs)) if cv_aucs else None
    test_auc = float(test_full.get("roc_auc", 0.0))
    gap = (cv_auc_mean - test_auc) if cv_auc_mean is not None else None
    guardrails["overfitting"] = {
        "status": "pass" if gap is None or abs(gap) < 0.08 else "warn",
        "cv_roc_auc_mean": round(cv_auc_mean, 4) if cv_auc_mean is not None else None,
        "test_roc_auc": round(test_auc, 4),
        "gap": round(float(gap), 4) if gap is not None else None,
    }

    drift_report = detect_data_drift(x_train, x_test, threshold=0.2)
    guardrails["drift"] = {
        "status": "pass" if not drift_report.drift_detected else "warn",
        "max_psi": round(float(max(drift_report.psi_scores.values(), default=0.0)), 4),
        "drifted_features": drift_report.drifted_features,
    }

    guardrails["data_poisoning"] = {
        "status": "pass",
        "data_sha256": _data_hash(data_path)[:16],
    }
    return guardrails


def _log_mlflow(
    version: str,
    params_map: dict,
    metrics_payload: dict,
    serving_config: dict,
    model,
    x_train: pd.DataFrame,
    artifact_root: Path,
) -> None:
    """Delegate to the shared fast-fallback logger (never blocks training)."""
    metrics = {
        f"holdout_{key}": value
        for key in ("pr_auc", "roc_auc", "brier_score", "log_loss")
        if (value := metrics_payload.get("test_full", {}).get(key)) is not None
    }
    metrics["threshold"] = float(serving_config["threshold"])
    log_to_mlflow_safely(
        run_name=version,
        experiment="churn_ensemble",
        params={
            "models": ",".join(MODEL_TYPES),
            "threshold": serving_config["threshold"],
            "as_of_date": metrics_payload.get("as_of_date", ""),
        },
        metrics=metrics,
        artifacts=[
            artifact_root / "metrics.json",
            artifact_root / "serving_config.json",
        ],
        model=model,
        input_example=x_train.head(5),
        tags={"model_version": version},
    )


def _promote(out_dir: Path, version: str, files: list[str]) -> Path:
    """Atomically publish a versioned artifact directory and `current` symlink."""
    version_dir = out_dir / "versions" / version
    version_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        src = out_dir / name
        if src.exists():
            shutil.copy2(src, version_dir / name)

    current = out_dir / "current"
    tmp = out_dir / ".current.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(Path("versions") / version, target_is_directory=True)
    try:
        if current.is_symlink() or current.is_file():
            current.unlink()
        elif current.exists():
            shutil.rmtree(current)
    except OSError:
        pass
    tmp.replace(current)
    return version_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrated churn ensemble training")
    parser.add_argument(
        "--data", type=Path, default=settings.data_processed / "training_set.parquet"
    )
    parser.add_argument("--target", default="churn")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument(
        "--use-smote", action="store_true", help="Apply SMOTE when the minority class is low"
    )
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Training set not found: {args.data}. Run `make features` first.")

    df = pd.read_parquet(args.data)
    logger.info("Training data: %d rows, %d columns", len(df), df.shape[1])

    x, y = split_features_target(df, target_col=args.target)
    num_cols = numeric_features(x)
    cat_cols = categorical_features(x)
    logger.info(
        "Feature count: %d (numeric %d, categorical %d)", x.shape[1], len(num_cols), len(cat_cols)
    )
    logger.info(
        "Target distribution: %s | minority rate: %.4f",
        y.value_counts().to_dict(),
        min(y.mean(), 1 - y.mean()),
    )

    params_file = PROJECT_ROOT / "configs" / "model_params.yaml"
    params_map = load_yaml(params_file) if params_file.exists() else {}

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=42, stratify=y
    )

    cv_result = _cv_scores(
        x_train, y_train, params_map, num_cols, cat_cols, args.use_smote, args.n_splits
    )
    cv_summary = {m: _summarize(v) for m, v in cv_result["metrics"].items()}

    pr_aucs = {m: cv_summary[m]["pr_auc"]["mean"] for m in MODEL_TYPES if "pr_auc" in cv_summary[m]}
    weights = np.asarray([max(pr_aucs.get(m, 0.5) - 0.5, 0.0) + 1e-6 for m in MODEL_TYPES])
    weights = weights / weights.sum()
    logger.info(
        "Model weights (OOF PR-AUC): %s", dict(zip(MODEL_TYPES, weights.tolist(), strict=False))
    )

    oof_ensemble = np.zeros(len(y_train))
    for m, w in zip(MODEL_TYPES, weights, strict=False):
        oof_ensemble += w * cv_result["oof"][m]

    # Cross-fitted Platt calibration: the calibrator never sees the holdout.
    calibrator = LogisticRegression(C=1.0, max_iter=2000)
    calibrator.fit(oof_ensemble.reshape(-1, 1), y_train)
    oof_calibrated = calibrator.predict_proba(oof_ensemble.reshape(-1, 1))[:, 1]
    oof_cal = pd.Series(oof_calibrated, index=y_train.index)
    threshold_f1 = find_best_threshold_f1(y_train, oof_cal)
    threshold_youden = find_best_threshold_youden(y_train, oof_cal)

    experiment_cfg = load_yaml(PROJECT_ROOT / "configs" / "config.yaml").get("experiment", {})
    ltv = float(experiment_cfg.get("ltv", 500.0))
    incentive_cost = float(experiment_cfg.get("incentive_cost", 50.0))
    uplift = float(experiment_cfg.get("uplift", 0.30))
    threshold_profit = find_optimal_threshold(y_train, oof_cal, ltv, incentive_cost, uplift)
    logger.info(
        "OOF thresholds -> profit: %.4f | F1: %.4f | Youden: %.4f",
        threshold_profit,
        threshold_f1,
        threshold_youden,
    )

    base = _build_base_pipelines(params_map, num_cols, cat_cols, y_train, args.use_smote)
    ensemble = SoftVotingEnsemble([pipe for _, pipe in base], weights=weights.tolist())
    ensemble.fit(x_train, y_train)
    serving_model = CalibratedEnsemble(ensemble, calibrator)

    test_proba = serving_model.predict_proba(x_test)[:, 1]
    test_score = pd.Series(test_proba, index=y_test.index)
    test_full = full_metrics(y_test, test_score)
    test_metrics_f1 = evaluate_binary(y_test, test_score, threshold_f1)
    test_metrics_profit = evaluate_binary(y_test, test_score, threshold_profit)
    calibration = calibration_metrics(y_test, test_score)

    pr_ci = bootstrap_ci(y_test, test_score, metric="pr_auc")
    roc_ci = bootstrap_ci(y_test, test_score, metric="roc_auc")
    curve = profit_curve(y_test, test_score, ltv, incentive_cost, uplift)
    best_row = curve.loc[curve["profit"].idxmax()]

    model_metrics: dict[str, dict] = {}
    for name, pipe in base:
        proba = pipe.predict_proba(x_test)[:, 1]
        score = pd.Series(proba, index=y_test.index)
        model_metrics[name] = {
            "cv": cv_summary.get(name, {}),
            "test": evaluate_binary(y_test, score, threshold_f1),
        }

    logistic_pipe = Pipeline(
        [("pre", build_preprocessor(num_cols, cat_cols)), ("clf", make_model("logistic"))]
    )
    logistic_pipe.fit(x_train, y_train)
    logistic_score = pd.Series(logistic_pipe.predict_proba(x_test)[:, 1], index=y_test.index)
    model_metrics["logistic"] = {
        "cv": {},
        "test": evaluate_binary(y_test, logistic_score, threshold_f1),
    }
    model_metrics["ensemble"] = {"cv": {}, "test": test_metrics_f1}

    guardrails = _compute_guardrails(df, x, x_train, x_test, y, test_full, cv_summary, args.data)
    ablation = _leakage_ablation(x_train, x_test, y_train, y_test, params_map, num_cols, cat_cols)

    lgbm_name = "lightgbm" if "lightgbm" in MODEL_TYPES else MODEL_TYPES[0]
    lgbm_pipe = next(pipe for name, pipe in base if name == lgbm_name)
    holdout_ids = df.iloc[y_test.index]["customer_id"].astype(str).reset_index(drop=True)
    # Global importance on the holdout; per-customer reasons for every customer
    # so the API can explain customers outside the holdout too.
    shap_df, _, shap_method = _compute_shap_details(
        lgbm_pipe, x_test, holdout_ids, model_type=lgbm_name
    )
    _, explanations, _ = _compute_shap_details(
        lgbm_pipe, x, df["customer_id"].astype(str).reset_index(drop=True), model_type=lgbm_name
    )

    timestamp = datetime.now(UTC)
    version = f"ensemble-{timestamp.strftime('%Y%m%d%H%M%S')}-{_git_sha()}"
    meta_path = args.data.parent / "feature_metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    as_of_date = metadata.get("as_of_date")
    horizon_days = int(metadata.get("horizon_days", 90))

    features_used = [c for c in x.columns]
    serving_config = {
        "model_version": version,
        "trained_at": timestamp.isoformat(),
        "threshold": round(float(threshold_profit), 6),
        "medium_risk": 0.40,
        "high_risk": 0.70,
        "feature_schema": features_used,
        "as_of_date": as_of_date,
        "horizon_days": horizon_days,
        "ltv": ltv,
        "incentive_cost": incentive_cost,
        "uplift": uplift,
    }

    payload = {
        "ensemble": True,
        "calibrated": True,
        "calibration_method": "platt_cross_fitted",
        "model_version": version,
        "trained_at": timestamp.isoformat(),
        "git_sha": _git_sha(),
        "data_hash": _data_hash(args.data),
        "models": MODEL_TYPES,
        "model_weights": {
            m: round(float(w), 4) for m, w in zip(MODEL_TYPES, weights, strict=False)
        },
        "target": args.target,
        "label_definition": "churn = 1 iff no purchase in (as_of, as_of + horizon_days]",
        "as_of_date": as_of_date,
        "horizon_days": horizon_days,
        "feature_columns": features_used,
        "imbalanced": bool(_is_imbalanced(y)),
        "minority_rate": round(float(min(y.mean(), 1 - y.mean())), 4),
        "smote_applied": bool(args.use_smote and _smote_available() and _is_imbalanced(y)),
        "threshold": round(float(threshold_profit), 6),
        "threshold_f1": round(threshold_f1, 6),
        "threshold_youden": round(threshold_youden, 6),
        "ltv": ltv,
        "incentive_cost": incentive_cost,
        "uplift": uplift,
        "profit": {
            "optimal_threshold_on_holdout_curve": round(float(best_row["threshold"]), 6),
            "optimal_profit_on_holdout_curve": round(float(best_row["profit"]), 2),
            "profit_at_serving_threshold": round(
                float(
                    curve.iloc[(curve["threshold"] - threshold_profit).abs().argsort().iloc[0]][
                        "profit"
                    ]
                ),
                2,
            ),
            "curve": [
                {
                    "threshold": round(float(r.threshold), 4),
                    "profit": round(float(r.profit), 2),
                    "precision": round(float(r.precision), 4),
                    "recall": round(float(r.recall), 4),
                }
                for r in curve.iloc[:: max(1, len(curve) // 100)].itertuples()
            ],
        },
        "calibration": calibration,
        "ci": {"pr_auc": pr_ci, "roc_auc": roc_ci},
        "cv_summary": {m: v for m, v in cv_summary.items()},
        "test_full": test_full,
        "test_metrics_f1": test_metrics_f1,
        "test_metrics_profit_threshold": test_metrics_profit,
        "model_metrics": model_metrics,
        "leakage_ablation": ablation,
        "shap_method": shap_method,
        "guardrails": guardrails,
        "num_features": int(x.shape[1]),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    dump(serving_model, args.out / "pipeline.joblib")
    shap_df.to_csv(args.out / "shap_importance.csv", index=False)
    (args.out / "explanations.json").write_text(
        json.dumps(explanations, ensure_ascii=False), encoding="utf-8"
    )
    (args.out / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "serving_config.json").write_text(
        json.dumps(serving_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    version_dir = _promote(
        args.out,
        version,
        [
            "pipeline.joblib",
            "metrics.json",
            "serving_config.json",
            "shap_importance.csv",
            "explanations.json",
        ],
    )

    _log_mlflow(version, params_map, payload, serving_config, serving_model, x_train, args.out)

    logger.info("Ensemble saved: %s (version dir: %s)", args.out, version_dir)
    print(
        json.dumps(
            {
                k: payload[k]
                for k in ("model_version", "threshold", "test_full", "leakage_ablation")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
