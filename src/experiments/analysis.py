"""Statistical analysis for A/B tests.

- Two-proportion z-test with a sign-consistent CI and lift CI.
- CUPED-adjusted Welch t-test for continuous metrics.
- O'Brien-Fleming / Lan-DeMets sequential testing.
- Guardrails, multiplicity correction and a practical-significance decision rule.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.experiments.assignment import assign_groups
from src.experiments.cuped import apply_cuped

logger = logging.getLogger(__name__)


@dataclass
class ProportionsTestResult:
    p1: float
    p2: float
    lift: float
    z_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    significant: bool
    lift_ci_low: float = 0.0
    lift_ci_high: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def proportions_z_test(
    control: pd.Series,
    treatment: pd.Series,
    alpha: float = 0.05,
) -> ProportionsTestResult:
    """Two-independent-proportion z-test (churn-rate comparison).

    ``z`` and the CI use the same orientation (treatment − control). A negative
    lift means the treatment reduced churn.
    """
    n1, n2 = len(control), len(treatment)
    if n1 == 0 or n2 == 0:
        raise ValueError("Both groups must have at least one observation")

    x1, x2 = int(control.sum()), int(treatment.sum())
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)

    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    diff = p2 - p1  # treatment - control, same sign as the CI
    z = diff / se if se > 0 else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

    se_diff = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    margin = stats.norm.ppf(1 - alpha / 2) * se_diff
    lift = diff / p1 if p1 > 0 else float("inf")
    lift_low = (diff - margin) / p1 if p1 > 0 else float("-inf")
    lift_high = (diff + margin) / p1 if p1 > 0 else float("inf")

    return ProportionsTestResult(
        p1=float(p1),
        p2=float(p2),
        lift=float(lift),
        z_stat=float(z),
        p_value=p_value,
        ci_low=float(diff - margin),
        ci_high=float(diff + margin),
        significant=bool(p_value < alpha),
        lift_ci_low=float(lift_low),
        lift_ci_high=float(lift_high),
    )


def cuped_t_test(
    control: pd.Series,
    treatment: pd.Series,
    alpha: float = 0.05,
) -> dict:
    """Two-sample Welch t-test (for CUPED-adjusted continuous metrics)."""
    n1, n2 = len(control), len(treatment)
    if n1 < 2 or n2 < 2:
        raise ValueError("At least 2 observations are required per group for the t-test")

    c = control.astype(float)
    t = treatment.astype(float)
    mean_diff = float(t.mean() - c.mean())

    var1 = float(c.var(ddof=1))
    var2 = float(t.var(ddof=1))
    se = math.sqrt(var1 / n1 + var2 / n2)

    if se > 0:
        t_stat = mean_diff / se
        df = (var1 / n1 + var2 / n2) ** 2 / (
            (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
        )
        p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df)))
        margin = stats.t.ppf(1 - alpha / 2, df) * se
    else:
        t_stat, df, p_value, margin = 0.0, float(n1 + n2 - 2), 1.0, 0.0

    return {
        "mean_diff": float(mean_diff),
        "t_stat": float(t_stat),
        "df": float(df),
        "p_value": float(p_value),
        "ci_low": float(mean_diff - margin),
        "ci_high": float(mean_diff + margin),
        "significant": bool(p_value < alpha),
    }


def obrien_fleming_boundary(information_fraction: float, alpha: float = 0.05) -> float:
    """O'Brien-Fleming interim-analysis boundary (z-statistic threshold)."""
    if not 0 < information_fraction <= 1:
        raise ValueError("information_fraction must be in the 0-1 range")
    return float(stats.norm.ppf(1 - alpha / 2) / math.sqrt(information_fraction))


def lan_demets_alpha_spent(information_fraction: float, alpha: float = 0.05) -> float:
    """Approximate Lan-DeMets alpha spending corresponding to OBF boundaries."""
    if not 0 < information_fraction <= 1:
        raise ValueError("information_fraction must be in the 0-1 range")
    boundary = obrien_fleming_boundary(information_fraction, alpha)
    return float(2 * (1 - stats.norm.cdf(boundary)))


def multiple_metric_correction(p_values: dict[str, float], method: str = "bonferroni") -> dict:
    """Adjust p-values for multiple primary/secondary metrics."""
    if method == "bonferroni":
        adjusted = {k: min(1.0, v * len(p_values)) for k, v in p_values.items()}
    elif method == "holm":
        ordered = sorted(p_values.items(), key=lambda item: item[1])
        adjusted = {}
        running = 0.0
        for rank, (key, value) in enumerate(ordered):
            running = max(running, min(1.0, value * (len(ordered) - rank)))
            adjusted[key] = running
    else:
        raise ValueError(f"Unknown correction method: {method}")
    return adjusted


def simulate_outcomes(
    assignment: pd.Series,
    seed: int = 7,
    baseline_churn: float = 0.50,
    treatment_effect: float = 0.20,
    avg_revenue: float = 300.0,
    incentive_cost: float = 50.0,
) -> pd.DataFrame:
    """Generate synthetic experiment outcomes with a deterministic seed."""
    if not 0 <= treatment_effect <= 1:
        raise ValueError("treatment_effect must be in the 0-1 range")

    rng = np.random.default_rng(seed)
    n = len(assignment)
    is_treatment = (assignment.astype(str) == "treatment").to_numpy()

    p_churn = np.where(is_treatment, baseline_churn * (1 - treatment_effect), baseline_churn)
    p_churn = np.clip(p_churn, 0.0, 1.0)
    churn = rng.random(n) < p_churn

    pre_revenue = np.maximum(rng.normal(avg_revenue, avg_revenue * 0.5, size=n), 0.0)
    post_revenue = np.where(
        churn,
        0.0,
        np.maximum(rng.normal(avg_revenue, avg_revenue * 0.4, size=n), 0.0),
    )
    discount_cost = np.where(is_treatment, float(incentive_cost), 0.0)

    return pd.DataFrame(
        {
            "assignment": assignment.astype(str),
            "churn": churn.astype(int),
            "revenue": post_revenue,
            "pre_revenue": pre_revenue,
            "discount_cost": discount_cost,
            "unsubscribe": (rng.random(n) < np.where(is_treatment, 0.006, 0.005)).astype(int),
            "complaint": (rng.random(n) < np.where(is_treatment, 0.004, 0.003)).astype(int),
        }
    )


def analyze_experiment(
    data: pd.DataFrame,
    alpha: float = 0.05,
    mde_absolute: float = 0.0,
    revenue_non_inferiority_margin: float = 0.0,
    information_fraction: float = 1.0,
    apply_multiplicity: bool = True,
) -> dict:
    """Analyze experiment data end-to-end and produce a decision report."""
    required = {"assignment", "churn"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    assignment = data["assignment"].astype(str)
    control_mask = assignment == "control"
    treatment_mask = assignment == "treatment"

    churn_test = proportions_z_test(
        data.loc[control_mask, "churn"],
        data.loc[treatment_mask, "churn"],
        alpha,
    )

    report: dict = {
        "n_control": int(control_mask.sum()),
        "n_treatment": int(treatment_mask.sum()),
        "churn_test": churn_test.to_dict(),
        "information_fraction": float(information_fraction),
        "alpha": float(alpha),
    }

    p_values = {"churn": churn_test.p_value}

    net_revenue_test = None
    if "revenue" in data.columns:
        discount = (
            data["discount_cost"]
            if "discount_cost" in data.columns
            else pd.Series(0.0, index=data.index)
        )
        net_revenue = data["revenue"] - discount

        if "pre_revenue" in data.columns:
            adjusted = apply_cuped(net_revenue, data["pre_revenue"], assignment)
            # No mixing of adjusted/unadjusted rows: drop any unusable rows.
            valid = adjusted.notna()
            adjusted = adjusted[valid]
            analysis_assignment = assignment[valid]
            cuped_used = True
        else:
            adjusted = net_revenue
            analysis_assignment = assignment
            cuped_used = False

        adjusted_control = adjusted[analysis_assignment == "control"]
        adjusted_treatment = adjusted[analysis_assignment == "treatment"]
        net_revenue_test = cuped_t_test(adjusted_control, adjusted_treatment, alpha)
        mean_control = float(net_revenue[control_mask].mean())
        mean_treatment = float(net_revenue[treatment_mask].mean())
        net_revenue_test["lift"] = (
            (mean_treatment - mean_control) / mean_control if mean_control > 0 else float("inf")
        )
        net_revenue_test["cuped_used"] = cuped_used
        net_revenue_test["cuped_n"] = int(len(adjusted))
        net_revenue_test["non_inferiority_margin"] = float(revenue_non_inferiority_margin)
        net_revenue_test["non_inferior"] = bool(
            net_revenue_test["ci_low"] > -float(revenue_non_inferiority_margin)
        )
        report["net_revenue_test"] = net_revenue_test
        p_values["net_revenue"] = float(net_revenue_test["p_value"])

    # Guardrails (only when the columns exist).
    guardrails: dict[str, dict] = {}
    for col in ("unsubscribe", "complaint"):
        if col in data.columns:
            c = data.loc[control_mask, col]
            t = data.loc[treatment_mask, col]
            test = proportions_z_test(c, t, alpha)
            guardrails[col] = {
                "control_rate": test.p1,
                "treatment_rate": test.p2,
                "p_value": test.p_value,
                "passed": not (test.significant and test.p2 > test.p1),
            }
    guardrails_passed = all(g["passed"] for g in guardrails.values())
    report["guardrails"] = guardrails
    report["guardrails_passed"] = guardrails_passed

    if apply_multiplicity and len(p_values) > 1:
        report["adjusted_p_values"] = multiple_metric_correction(p_values, method="holm")
        report["multiplicity_method"] = "holm"

    # Sequential test: only declare significance if the observed z exceeds the
    # O'Brien-Fleming boundary for the current information fraction.
    boundary = obrien_fleming_boundary(information_fraction, alpha)
    report["sequential"] = {
        "information_fraction": float(information_fraction),
        "obrien_fleming_z": boundary,
        "observed_abs_z": abs(churn_test.z_stat),
        "alpha_spent": lan_demets_alpha_spent(information_fraction, alpha),
        "can_stop": bool(abs(churn_test.z_stat) >= boundary),
    }

    # Decision rule with practical significance + non-inferiority.
    churn_sig = churn_test.significant
    churn_direction = churn_test.p2 < churn_test.p1
    practical = (churn_test.p1 - churn_test.p2) >= float(mde_absolute)
    churn_wins = churn_sig and churn_direction and practical
    net_non_inferior = True if net_revenue_test is None else bool(net_revenue_test["non_inferior"])
    net_erodes = not net_non_inferior

    if churn_wins and net_non_inferior and guardrails_passed:
        decision = "LAUNCH"
    elif churn_sig and churn_direction and net_erodes:
        decision = "OPTIMIZE"
    else:
        decision = "HOLD"
    report["decision"] = decision
    report["decision_inputs"] = {
        "churn_significant": churn_sig,
        "churn_direction_improved": churn_direction,
        "practical_significance": practical,
        "net_non_inferior": net_non_inferior,
        "guardrails_passed": guardrails_passed,
        "mde_absolute": float(mde_absolute),
    }
    report["simulated"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B experiment analysis (synthetic demo)")
    parser.add_argument("--n", type=int, default=10000, help="Total number of customers")
    parser.add_argument("--baseline", type=float, default=0.50, help="Baseline churn rate")
    parser.add_argument("--effect", type=float, default=0.20, help="Relative treatment effect")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("artifacts/ab_analysis_report.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    customers = pd.DataFrame({"customer_id": [f"C{i}" for i in range(args.n)]})
    assignment = assign_groups(customers, treatment_ratio=0.5, seed=42)
    outcomes = simulate_outcomes(
        assignment,
        seed=args.seed,
        baseline_churn=args.baseline,
        treatment_effect=args.effect,
    )
    report = analyze_experiment(outcomes, args.alpha)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Analysis report saved: %s", args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
