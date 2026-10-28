"""Model evaluation metrics.

Metric selection rationale: docs/metrics.md
Priority: PR-AUC > Recall/Precision > ROC-AUC. Threshold optimisation uses the
incremental profit curve with configurable LTV, incentive cost and uplift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary(y_true: pd.Series, y_score: pd.Series, threshold: float = 0.5) -> dict:
    """Compute binary classification metrics."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def profit_curve(
    y_true: pd.Series,
    y_score: pd.Series,
    ltv: float,
    incentive_cost: float,
    uplift: float = 1.0,
) -> pd.DataFrame:
    """Incremental profit per threshold.

    ``sklearn.precision_recall_curve`` returns ``n+1`` precision/recall values
    and ``n`` thresholds. The first precision/recall point (threshold=+inf) is
    dropped so every row has a real threshold, and a final row at threshold=1.0
    represents "contact nobody".

    Profit(τ) = TP(τ) × uplift × LTV − FP(τ) × incentive_cost − FN(τ) × LTV
    """
    if ltv < 0 or incentive_cost < 0 or not 0 <= uplift <= 1:
        raise ValueError("ltv/incentive_cost must be non-negative and uplift in [0, 1]")

    yt = np.asarray(y_true, dtype=int)
    ys = np.asarray(y_score, dtype=float)
    n_pos = int(yt.sum())

    precision, recall, thresholds = precision_recall_curve(yt, ys)
    # sklearn appends one final (precision=1, recall=0) point with no matching
    # threshold; drop that point so arrays align one-to-one with thresholds.
    precision = precision[:-1]
    recall = recall[:-1]

    tp = recall * n_pos
    fp = np.divide(tp, precision, out=np.zeros_like(tp), where=precision > 0) - tp
    fn = n_pos - tp
    profit = tp * uplift * ltv - fp * incentive_cost - fn * ltv

    rows = [
        {
            "threshold": float(t),
            "profit": float(p),
            "baseline_profit": 0.0,
            "profit_delta": float(p),
            "precision": float(prec),
            "recall": float(rec),
            "tp": float(tpv),
            "fp": float(fpv),
            "fn": float(fnv),
        }
        for t, p, prec, rec, tpv, fpv, fnv in zip(
            thresholds, profit, precision, recall, tp, fp, fn, strict=True
        )
    ]
    # Explicit "contact nobody" endpoint.
    rows.append(
        {
            "threshold": 1.0,
            "profit": -float(n_pos) * ltv,
            "baseline_profit": 0.0,
            "profit_delta": -float(n_pos) * ltv,
            "precision": 0.0,
            "recall": 0.0,
            "tp": 0.0,
            "fp": 0.0,
            "fn": float(n_pos),
        }
    )
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def find_optimal_threshold(
    y_true: pd.Series,
    y_score: pd.Series,
    ltv: float,
    incentive_cost: float,
    uplift: float = 1.0,
) -> float:
    """Find the threshold that maximises incremental profit."""
    curve = profit_curve(y_true, y_score, ltv, incentive_cost, uplift)
    return float(curve.loc[curve["profit"].idxmax(), "threshold"])


def find_best_threshold_f1(y_true: pd.Series, y_score: pd.Series) -> float:
    """Find the decision threshold that maximises the F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    thresholds = np.append(thresholds, 1.0)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def find_best_threshold_youden(y_true: pd.Series, y_score: pd.Series) -> float:
    """Find the threshold maximizing Youden's J (TPR - FPR) on the ROC curve."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def lift_at_top_decile(y_true: pd.Series, y_score: pd.Series) -> float:
    """Compute the lift within the top decile."""
    df = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(y_score)})
    k = max(1, int(len(df) * 0.1))
    top = df.nlargest(k, "score")
    baseline = df["y"].mean()
    if baseline == 0:
        return float("inf")
    return float(top["y"].mean() / baseline)


def calibration_metrics(y_true: pd.Series, y_score: pd.Series, n_bins: int = 10) -> dict:
    """Brier score, log loss, reliability curve and calibration slope/intercept."""
    yt = np.asarray(y_true, dtype=int)
    ys = np.clip(np.asarray(y_score, dtype=float), 1e-6, 1 - 1e-6)

    prob_true, prob_pred = calibration_curve(yt, ys, n_bins=n_bins, strategy="quantile")
    eps = 1e-6
    logit = np.log(ys / (1 - ys)).reshape(-1, 1)
    # Calibration slope/intercept: fit observed outcome on the predicted logit.
    slope_model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    slope_model.fit(logit, yt)
    slope = float(slope_model.coef_[0, 0])
    intercept = float(slope_model.intercept_[0])

    # Expected calibration error over the reliability bins.
    bin_ids = np.digitize(ys, np.quantile(ys, np.linspace(0, 1, n_bins + 1))[1:-1])
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            ece += mask.mean() * abs(float(yt[mask].mean()) - float(ys[mask].mean()))

    return {
        "brier_score": float(brier_score_loss(yt, ys)),
        "log_loss": float(log_loss(yt, ys, labels=[0, 1])),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "expected_calibration_error": float(ece),
        "reliability_curve": {
            "prob_true": [float(x) for x in prob_true],
            "prob_pred": [float(x) for x in prob_pred],
        },
        "eps": eps,
    }


def bootstrap_ci(
    y_true: pd.Series,
    y_score: pd.Series,
    metric: str = "pr_auc",
    n_boot: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap CI for PR-AUC / ROC-AUC."""
    yt = np.asarray(y_true, dtype=int)
    ys = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = len(yt)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(yt[idx])) < 2:
            continue
        if metric == "pr_auc":
            values.append(float(average_precision_score(yt[idx], ys[idx])))
        elif metric == "roc_auc":
            values.append(float(roc_auc_score(yt[idx], ys[idx])))
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    if not values:
        return {"mean": None, "low": None, "high": None, "n_boot": 0}
    return {
        "mean": float(np.mean(values)),
        "low": float(np.quantile(values, alpha / 2)),
        "high": float(np.quantile(values, 1 - alpha / 2)),
        "n_boot": len(values),
    }


def full_metrics(y_true: pd.Series, y_score: pd.Series) -> dict:
    """Threshold-independent summary metrics + calibrated probability quality."""
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "lift_top_decile": lift_at_top_decile(y_true, y_score),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "log_loss": float(log_loss(y_true, np.clip(y_score, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "n": int(len(y_true)),
        "pos_rate": float(np.mean(np.asarray(y_true))),
    }


def summarize_cv(metrics_list: list[dict]) -> dict:
    """Summarize fold metrics as mean/standard deviation."""
    keys = {k for m in metrics_list for k in m if k != "fold"}
    summary: dict[str, dict[str, float]] = {}
    for key in sorted(keys):
        vals = [m[key] for m in metrics_list if key in m and m[key] is not None]
        if vals:
            summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return summary
