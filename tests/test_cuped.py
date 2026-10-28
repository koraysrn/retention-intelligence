"""CUPED variance reduction tests."""

import numpy as np
import pandas as pd
from src.experiments.cuped import apply_cuped, estimate_theta, variance_reduction


def _data() -> tuple[pd.Series, pd.Series, pd.Series]:
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(100, 30, n)  # pre-experiment metric
    noise = rng.normal(0, 5, n)
    y = x * 0.5 + noise  # outcome, strongly correlated with the pre-metric
    group = pd.Series(["control", "treatment"] * (n // 2))
    return pd.Series(y), pd.Series(x), group


def test_apply_cuped_reduces_variance() -> None:
    y, x, group = _data()
    adjusted = apply_cuped(y, x, group)
    assert y.var() > adjusted.var()
    assert not adjusted.isna().all()


def test_estimate_theta_positive() -> None:
    y, x, group = _data()
    theta = estimate_theta(y, x, group)
    assert abs(theta - 0.5) < 0.05


def test_variance_reduction_between_0_and_1() -> None:
    y, x, group = _data()
    vr = variance_reduction(y, x, group)
    assert 0.0 <= vr <= 1.0
    assert vr > 0.2  # noticeable reduction under strong correlation


def test_constant_pre_metric_theta_zero() -> None:
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    x = pd.Series([5.0, 5.0, 5.0, 5.0])
    group = pd.Series(["control", "control", "treatment", "treatment"])
    assert estimate_theta(y, x, group) == 0.0


def test_missing_pre_values_are_imputed_not_mixed() -> None:
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    x = pd.Series([10.0, np.nan, 30.0, 40.0])
    group = pd.Series(["control", "control", "treatment", "treatment"])
    adjusted = apply_cuped(y, x, group)
    # Missing pre-period values are filled with the control mean before the
    # adjustment; adjusted and unadjusted rows are never mixed.
    assert adjusted.isna().sum() == 0
    assert len(adjusted) == 4
