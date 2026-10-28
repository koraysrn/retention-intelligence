"""Stratified randomization (A/B assignment) with SRM/balance diagnostics.

Assignment is deterministic and balanced **within each stratum** (country,
spend/risk decile, tenure bucket). Within a stratum, customers are sorted by a
seeded hash and the first ``round(n * treatment_ratio)`` receive treatment. This
is stratified complete randomization: it guarantees the target ratio per
stratum instead of relying on global hash luck.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy import stats

STRATA_COLUMNS = ["country", "spend_decile", "tenure_bucket"]


def _hash_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{value}::{seed}".encode()).hexdigest()


def assign_groups(
    customers: pd.DataFrame,
    treatment_ratio: float = 0.5,
    seed: int = 42,
    id_column: str = "customer_id",
) -> pd.Series:
    """Deterministically assign customers with stratified block randomization."""
    if not 0 < treatment_ratio < 1:
        raise ValueError("treatment_ratio must be between 0 and 1")
    if id_column not in customers.columns:
        raise ValueError(f"Column '{id_column}' not found in DataFrame")

    df = customers.copy()
    strata_cols = [c for c in STRATA_COLUMNS if c in df.columns]
    if strata_cols:
        stratum_key = df[strata_cols].astype(str).agg("|".join, axis=1)
    else:
        stratum_key = pd.Series(["__all__"] * len(df), index=df.index)

    order = df[id_column].astype(str).map(lambda s: _hash_key(s, seed))
    df = df.assign(_order=order, _stratum=stratum_key)
    assignment = pd.Series("control", index=df.index, name="group")

    for _, group in df.groupby("_stratum", dropna=False, sort=True):
        ordered = group.sort_values("_order")
        n = len(ordered)
        n_treatment = int(round(n * treatment_ratio))
        if n >= 2:
            n_treatment = min(max(1, n_treatment), n - 1)
        assignment.loc[ordered.index[:n_treatment]] = "treatment"

    return assignment


def strata_balance_summary(
    customers: pd.DataFrame,
    assignment: pd.Series,
) -> pd.DataFrame:
    """Report group size and treatment ratio per stratum."""
    strata_cols = [c for c in STRATA_COLUMNS if c in customers.columns]
    if not strata_cols:
        return pd.DataFrame()

    df = customers[strata_cols].copy()
    df["group"] = assignment.astype(str)

    rows: list[dict] = []
    for keys, sub in df.groupby(strata_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(sub)
        n_treatment = int((sub["group"] == "treatment").sum())
        row = dict(zip(strata_cols, keys, strict=True))
        row["n"] = n
        row["n_treatment"] = n_treatment
        row["n_control"] = n - n_treatment
        row["treatment_ratio"] = n_treatment / n if n else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def srm_chi_square(
    assignment: pd.Series,
    expected_treatment_ratio: float = 0.5,
) -> dict:
    """Sample Ratio Mismatch check (chi-square goodness of fit)."""
    counts = assignment.astype(str).value_counts()
    n_control = int(counts.get("control", 0))
    n_treatment = int(counts.get("treatment", 0))
    n = n_control + n_treatment
    if n == 0:
        return {"chi2": 0.0, "p_value": 1.0, "srm_detected": False, "n": 0}
    expected = np.array([n * (1 - expected_treatment_ratio), n * expected_treatment_ratio])
    observed = np.array([n_control, n_treatment])
    chi2, p_value = stats.chisquare(observed, f_exp=expected)
    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "srm_detected": bool(p_value < 0.001),
        "n_control": n_control,
        "n_treatment": n_treatment,
        "observed_ratio": float(n_treatment / n),
        "expected_ratio": float(expected_treatment_ratio),
    }


def standardized_mean_difference(
    control: pd.Series,
    treatment: pd.Series,
) -> float:
    """Standardized mean difference (SMD) for a numeric covariate."""
    c = pd.to_numeric(control, errors="coerce").dropna()
    t = pd.to_numeric(treatment, errors="coerce").dropna()
    if len(c) < 2 or len(t) < 2:
        return 0.0
    pooled = np.sqrt((float(c.var(ddof=1)) + float(t.var(ddof=1))) / 2.0)
    if pooled == 0:
        return 0.0
    return float((float(t.mean()) - float(c.mean())) / pooled)


def covariate_balance(
    customers: pd.DataFrame,
    assignment: pd.Series,
    covariates: list[str] | None = None,
    smd_threshold: float = 0.10,
) -> pd.DataFrame:
    """Per-covariate SMD between control and treatment (balance check)."""
    covariates = covariates or [
        c
        for c in ("age", "tenure_days", "total_sessions", "total_spend_usd")
        if c in customers.columns
    ]
    rows: list[dict] = []
    control_mask = assignment.astype(str) == "control"
    treatment_mask = assignment.astype(str) == "treatment"
    for col in covariates:
        values = pd.to_numeric(customers[col], errors="coerce")
        smd = standardized_mean_difference(values[control_mask], values[treatment_mask])
        rows.append(
            {
                "covariate": col,
                "smd": smd,
                "balanced": abs(smd) < smd_threshold,
            }
        )
    return pd.DataFrame(rows)
