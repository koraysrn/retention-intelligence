# A/B Test Design

Measures whether a re-engagement campaign changes the **90-day churn rate**
defined in [`docs/churn_definition.md`](churn_definition.md), and whether net
revenue is non-inferior after incentive cost.

## 1. Hypotheses

| # | Hypothesis |
|---|---|
| H0 (primary) | The campaign does not change the 90-day churn rate |
| H1 (primary) | The campaign reduces the 90-day churn rate |
| H0 (secondary) | Net revenue is inferior beyond the non-inferiority margin |
| H1 (secondary) | Net revenue is non-inferior (or better) |

## 2. Parameters

| Parameter | Value |
|---|---|
| Baseline churn | `training_set.parquet → churn.mean()` (as-of label) |
| MDE | Relative (default −20%); converted to absolute for the practical-significance check |
| α / power | 0.05 two-sided / 0.80 |
| Ratio | 50/50 |
| Unit | Customer |
| Strata | `country`, `spend_decile`, `tenure_bucket` |
| Assignment | Stratified **block randomization** (seeded hash order within each stratum) |

`risk_decile` is no longer used as a stratum name. When scores exist, the model
risk decile is reported alongside; the assignment stratum is `spend_decile`,
which is known pre-treatment.

## 3. Sample size

Two-proportion formula, plus a buffer:

```text
n per group = (Zα/2 + Zβ)² × [p1(1−p1) + p2(1−p2)] / (p1 − p2)²
```

`python -m scripts.run_ab_experiment` computes this from the actual baseline.

## 4. Assignment quality

`src/experiments/assignment.py` reports:

- **SRM:** chi-square goodness of fit against the intended ratio. `p < 0.001`
  is flagged.
- **SMD balance:** standardized mean difference per covariate; `|SMD| < 0.10`
  is considered balanced.
- **Per-stratum treatment ratio** (`strata_balance_summary`).

The report is written to `artifacts/ab_experiment_report.json → assignment`.

## 5. CUPED

```text
Y_adjusted = Y − θ × (X_pre − X̄_pre)
```

- θ is estimated on the **control arm** when it has at least 10 observations
  (otherwise pooled).
- Missing pre-period values are filled with the control mean *before*
  adjustment; adjusted and unadjusted rows are never mixed.
- `variance_reduction()` reports the variance reduction.

## 6. Sequential testing

The analysis reports the O'Brien-Fleming boundary for the current information
fraction and the corresponding Lan-DeMets alpha spending. A result is only
“stoppable” when `|z| >= boundary`; the final analysis uses
`information_fraction = 1.0`.

## 7. Decision rule

```text
churn_wins = significant AND treatment churn < control churn
             AND (control − treatment) >= MDE_absolute

net_non_inferior = CI_low(net_revenue_diff) > −non_inferiority_margin

guardrails_passed = unsubscribe/complaint not significantly worse
                    (when those columns exist)

LAUNCH    if churn_wins AND net_non_inferior AND guardrails_passed
OPTIMIZE  if churn improved significantly but net revenue erodes
HOLD      otherwise
```

`net_erodes` is **not** `mean_diff < 0`; it is a non-inferiority test against a
configurable margin. Holm correction is applied to the reported p-values when
multiple metrics are tested. The report explicitly includes
`decision_inputs` so the rule is auditable.

## 8. Tracked metrics

| Type | Metric |
|---|---|
| Primary | 90-day churn |
| Secondary | Net revenue (CUPED-adjusted), AOV, purchase frequency |
| Guardrail | Unsubscribe, complaint, margin |
| Health | SRM, SMD balance, per-stratum ratio |

## 9. Simulation disclaimer

`scripts/run_ab_experiment.py` simulates outcomes because no real campaign is
executed. The report always contains `"simulated": true`. Do not present the
simulated lift as a real business result.
