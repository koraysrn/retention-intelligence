"""CUPED — variance reduction using pre-experiment metrics.

Y_adjusted = Y − θ × (X_pre − X̄_pre)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_theta(
    outcome: pd.Series,
    pre_experiment_metric: pd.Series,
    group_assignment: pd.Series,
) -> float:
    """Estimate the CUPED coefficient θ, control-weighted.

    The control arm is unperturbed by treatment, so θ is estimated there when
    enough observations exist; otherwise the pooled estimate is used.
    """
    data = pd.DataFrame(
        {"y": outcome, "x": pre_experiment_metric, "group": group_assignment.astype(str)}
    ).dropna()
    if len(data) < 2:
        return 0.0

    control = data[data["group"] == "control"]
    sample = control if len(control) >= 10 else data
    var_x = float(sample["x"].var(ddof=1))
    if not np.isfinite(var_x) or var_x == 0:
        return 0.0

    cov_xy = float(sample["x"].cov(sample["y"]))
    return cov_xy / var_x


def apply_cuped(
    outcome: pd.Series,
    pre_experiment_metric: pd.Series,
    group_assignment: pd.Series,
) -> pd.Series:
    """Apply the CUPED adjustment to the outcome metric.

    Missing pre-period values are filled with the control mean **before**
    adjustment, so adjusted/unadjusted rows are never mixed. Only rows with a
    missing outcome remain NaN.
    """
    frame = pd.DataFrame(
        {
            "y": pd.to_numeric(outcome, errors="coerce"),
            "x": pd.to_numeric(pre_experiment_metric, errors="coerce"),
            "group": group_assignment.astype(str),
        }
    )
    theta = estimate_theta(frame["y"], frame["x"], frame["group"])
    control_x = frame.loc[frame["group"] == "control", "x"]
    x_mean = float(control_x.mean()) if control_x.notna().any() else float(frame["x"].mean())
    x_filled = frame["x"].fillna(x_mean)
    adjusted = frame["y"] - theta * (x_filled - x_mean)
    return adjusted


def variance_reduction(
    outcome: pd.Series,
    pre_experiment_metric: pd.Series,
    group_assignment: pd.Series,
) -> float:
    """Return the relative variance reduction (0-1) provided by CUPED."""
    data = pd.DataFrame(
        {"y": outcome, "x": pre_experiment_metric, "group": group_assignment}
    ).dropna()
    if len(data) < 2:
        return 0.0

    adjusted = apply_cuped(outcome, pre_experiment_metric, group_assignment).dropna()
    var_raw = float(data["y"].var(ddof=1))
    var_adj = float(adjusted.var(ddof=1))
    if var_raw <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - var_adj / var_raw))
