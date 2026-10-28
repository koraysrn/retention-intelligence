"""Stratified A/B assignment tests."""

import pandas as pd
import pytest
from src.experiments.assignment import (
    assign_groups,
    covariate_balance,
    srm_chi_square,
    strata_balance_summary,
)


def _customers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(200)],
            "country": ["UK"] * 100 + ["DE"] * 100,
            "spend_decile": list(range(10)) * 20,
            "tenure_bucket": ["0-30", "31-90"] * 100,
        }
    )


def test_assignment_is_binary_and_deterministic() -> None:
    customers = _customers()
    a1 = assign_groups(customers, treatment_ratio=0.5, seed=42)
    a2 = assign_groups(customers, treatment_ratio=0.5, seed=42)
    assert set(a1.unique()).issubset({"control", "treatment"})
    assert (a1 == a2).all()


def test_different_seed_changes_assignment() -> None:
    customers = _customers()
    a1 = assign_groups(customers, treatment_ratio=0.5, seed=1)
    a2 = assign_groups(customers, treatment_ratio=0.5, seed=2)
    assert not (a1 == a2).all()


def test_treatment_ratio_approximate() -> None:
    customers = _customers()
    a = assign_groups(customers, treatment_ratio=0.3, seed=42)
    ratio = (a == "treatment").mean()
    assert 0.2 < ratio < 0.4


def test_invalid_treatment_ratio_raises() -> None:
    with pytest.raises(ValueError):
        assign_groups(_customers(), treatment_ratio=1.5)


def test_missing_id_column_raises() -> None:
    with pytest.raises(ValueError, match="customer_id"):
        assign_groups(pd.DataFrame({"x": [1, 2]}), id_column="customer_id")


def test_strata_balance_summary_has_expected_columns() -> None:
    customers = _customers()
    a = assign_groups(customers, 0.5, seed=42)
    summary = strata_balance_summary(customers, a)
    assert {"country", "spend_decile", "tenure_bucket", "n", "treatment_ratio"}.issubset(
        summary.columns
    )
    assert summary["treatment_ratio"].between(0, 1).all()


def test_assignment_is_balanced_within_each_stratum() -> None:
    customers = _customers()
    a = assign_groups(customers, 0.5, seed=42)
    df = customers.copy()
    df["group"] = a
    for _, sub in df.groupby(["country", "spend_decile", "tenure_bucket"]):
        assert abs((sub["group"] == "treatment").mean() - 0.5) <= 0.5 / len(sub)


def test_srm_not_detected_for_balanced_assignment() -> None:
    customers = _customers()
    a = assign_groups(customers, 0.5, seed=42)
    srm = srm_chi_square(a, 0.5)
    assert srm["srm_detected"] is False
    assert srm["p_value"] > 0.001


def test_covariate_balance_reports_smd() -> None:
    customers = _customers()
    customers["age"] = range(200)
    a = assign_groups(customers, 0.5, seed=42)
    balance = covariate_balance(customers, a, covariates=["age"])
    assert set(balance.columns) == {"covariate", "smd", "balanced"}
    assert abs(float(balance.iloc[0]["smd"])) < 0.2


def test_strata_balance_summary_empty_when_no_strata() -> None:
    customers = pd.DataFrame({"customer_id": ["A", "B", "C"]})
    a = assign_groups(customers, 0.5, seed=1)
    summary = strata_balance_summary(customers, a)
    assert summary.empty
