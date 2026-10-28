"""A/B statistical analysis tests."""

import math

import numpy as np
import pandas as pd
import pytest
from src.experiments.analysis import (
    analyze_experiment,
    cuped_t_test,
    lan_demets_alpha_spent,
    multiple_metric_correction,
    obrien_fleming_boundary,
    proportions_z_test,
    simulate_outcomes,
)


def test_proportions_z_test_detects_difference() -> None:
    rng = np.random.default_rng(0)
    control = pd.Series(rng.random(5000) < 0.50).astype(int)
    treatment = pd.Series(rng.random(5000) < 0.40).astype(int)
    result = proportions_z_test(control, treatment, alpha=0.05)
    assert result.significant
    assert result.p1 > result.p2
    assert result.lift < 0
    assert result.ci_high < 0  # negative difference; the CI upper bound is also negative


def test_proportions_z_test_no_difference() -> None:
    rng = np.random.default_rng(1)
    control = pd.Series(rng.random(5000) < 0.50).astype(int)
    treatment = pd.Series(rng.random(5000) < 0.50).astype(int)
    result = proportions_z_test(control, treatment, alpha=0.05)
    assert not result.significant


def test_proportions_z_test_empty_group_raises() -> None:
    with pytest.raises(ValueError):
        proportions_z_test(pd.Series(dtype=int), pd.Series([0, 1]))


def test_cuped_t_test_detects_difference() -> None:
    rng = np.random.default_rng(2)
    control = pd.Series(rng.normal(100, 10, 1000))
    treatment = pd.Series(rng.normal(110, 10, 1000))
    result = cuped_t_test(control, treatment, alpha=0.05)
    assert result["significant"]
    assert result["mean_diff"] > 0


def test_cuped_t_test_small_group_raises() -> None:
    with pytest.raises(ValueError):
        cuped_t_test(pd.Series([1.0]), pd.Series([2.0]))


def test_obrien_fleming_boundary_decreases() -> None:
    b_early = obrien_fleming_boundary(0.25)
    b_late = obrien_fleming_boundary(1.0)
    assert b_early > b_late
    assert math.isclose(b_late, 1.96, abs_tol=0.01)


def test_obrien_fleming_invalid_fraction() -> None:
    with pytest.raises(ValueError):
        obrien_fleming_boundary(1.5)


def test_simulate_outcomes_produces_expected_columns() -> None:
    assignment = pd.Series(["control", "treatment"] * 500)
    data = simulate_outcomes(assignment, seed=7, baseline_churn=0.5, treatment_effect=0.2)
    assert {"assignment", "churn", "revenue", "pre_revenue", "discount_cost"}.issubset(data.columns)
    assert set(data["churn"].unique()).issubset({0, 1})
    c = data.loc[data["assignment"] == "control", "churn"].mean()
    t = data.loc[data["assignment"] == "treatment", "churn"].mean()
    assert t < c


def test_simulate_outcomes_invalid_effect_raises() -> None:
    with pytest.raises(ValueError):
        simulate_outcomes(pd.Series(["control", "treatment"]), treatment_effect=1.5)


def test_analyze_experiment_reports_decision() -> None:
    assignment = pd.Series(["control", "treatment"] * 5000)
    data = simulate_outcomes(assignment, seed=7, baseline_churn=0.5, treatment_effect=0.2)
    report = analyze_experiment(data, alpha=0.05)
    assert report["n_control"] == 5000
    assert report["n_treatment"] == 5000
    assert report["decision"] in {"LAUNCH", "OPTIMIZE", "HOLD"}
    assert "churn_test" in report
    assert report["churn_test"]["significant"]


def test_analyze_experiment_launch_when_net_positive() -> None:
    rng = np.random.default_rng(3)
    n = 5000
    control_churn = (rng.random(n) < 0.5).astype(int)
    treatment_churn = (rng.random(n) < 0.3).astype(int)
    churn = pd.Series(np.concatenate([control_churn, treatment_churn]))
    assignment = pd.Series(["control"] * n + ["treatment"] * n)
    revenue = pd.Series(np.where(churn == 0, 300.0, 0.0))

    data = pd.DataFrame({"assignment": assignment, "churn": churn, "revenue": revenue})
    report = analyze_experiment(data, alpha=0.05)
    assert report["churn_test"]["significant"]
    assert report["decision"] == "LAUNCH"


def test_analyze_experiment_missing_columns_raises() -> None:
    with pytest.raises(ValueError, match="Missing columns"):
        analyze_experiment(pd.DataFrame({"assignment": ["control", "treatment"]}))


def test_z_test_ci_and_z_have_same_sign() -> None:
    rng = np.random.default_rng(0)
    control = pd.Series(rng.random(5000) < 0.50).astype(int)
    treatment = pd.Series(rng.random(5000) < 0.40).astype(int)
    result = proportions_z_test(control, treatment)
    assert result.z_stat < 0
    assert result.ci_high < 0
    assert result.lift_ci_high < 0


def test_cuped_rows_are_not_mixed_with_unadjusted() -> None:
    assignment = pd.Series(["control", "treatment"] * 500)
    data = simulate_outcomes(assignment, seed=7, baseline_churn=0.5, treatment_effect=0.2)
    data.loc[data.index[:20], "pre_revenue"] = np.nan
    report = analyze_experiment(data)
    assert report["net_revenue_test"]["cuped_used"] is True
    # Missing pre-period values are imputed, so every row is adjusted once.
    assert report["net_revenue_test"]["cuped_n"] == len(data)


def test_sequential_boundary_is_reported() -> None:
    assignment = pd.Series(["control", "treatment"] * 5000)
    data = simulate_outcomes(assignment, seed=7, baseline_churn=0.5, treatment_effect=0.2)
    report = analyze_experiment(data, information_fraction=0.5)
    assert report["sequential"]["obrien_fleming_z"] > 1.96
    assert 0 < report["sequential"]["alpha_spent"] < 0.05


def test_decision_requires_practical_significance() -> None:
    rng = np.random.default_rng(3)
    n = 5000
    control_churn = (rng.random(n) < 0.5).astype(int)
    treatment_churn = (rng.random(n) < 0.3).astype(int)
    churn = pd.Series(np.concatenate([control_churn, treatment_churn]))
    assignment = pd.Series(["control"] * n + ["treatment"] * n)
    revenue = pd.Series(np.where(churn == 0, 300.0, 0.0))
    data = pd.DataFrame({"assignment": assignment, "churn": churn, "revenue": revenue})
    report = analyze_experiment(data, mde_absolute=0.25)
    assert report["decision_inputs"]["practical_significance"] is False
    assert report["decision"] == "HOLD"


def test_lan_demets_alpha_increases_with_information() -> None:
    assert lan_demets_alpha_spent(0.25) < lan_demets_alpha_spent(1.0)
    assert math.isclose(lan_demets_alpha_spent(1.0), 0.05, abs_tol=1e-9)


def test_multiple_metric_correction() -> None:
    adjusted = multiple_metric_correction({"a": 0.01, "b": 0.04})
    assert adjusted["a"] == pytest.approx(0.02)
    assert adjusted["b"] == pytest.approx(0.08)
