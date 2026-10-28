# Model Metrics and Evaluation Protocol

This document matches the code in [`src/models/evaluate.py`](../src/models/evaluate.py)
and [`scripts/train_ensemble.py`](../scripts/train_ensemble.py). There is one
protocol: a stratified holdout plus StratifiedKFold CV.

## 1. Metrics

| Metric | Role | Priority |
|---|---|---|
| **PR-AUC** | Ranking quality for the minority churn class | Highest |
| **Recall / Precision** | Business cost trade-off at the operating threshold | High |
| **ROC-AUC** | Secondary ranking comparison | Medium |
| **Brier score / log loss** | Probability quality (calibration) | High |
| **Reliability curve / ECE / calibration slope** | Are predicted probabilities trustworthy? | High |
| **Profit curve** | Threshold choice from LTV, incentive cost and uplift | Business |
| **Lift (top decile)** | Campaign targeting efficiency | Supporting |
| **Bootstrap CI** | Uncertainty on PR-AUC / ROC-AUC | Required |

Accuracy is not used: the churn class is not balanced and accuracy hides the
cost of missed churners.

## 2. Profit curve

```text
TP(τ) = recall(τ) × n_positive
FP(τ) = TP(τ) / precision(τ) − TP(τ)
FN(τ) = n_positive − TP(τ)

Profit(τ) = TP(τ) × uplift × LTV − FP(τ) × incentive_cost − FN(τ) × LTV
```

- `LTV`, `incentive_cost` and `uplift` come from `configs/config.yaml`
  (`experiment.ltv`, `experiment.incentive_cost`, `experiment.uplift`), not
  hardcoded constants.
- `sklearn.precision_recall_curve` returns `n+1` precision/recall points and
  `n` thresholds. We drop the final point (which has no threshold) and append an
  explicit `threshold = 1.0` (“contact nobody”) row.
- `find_optimal_threshold()` uses this corrected curve. A hand-checkable unit
  test lives in [`tests/test_evaluate.py`](../tests/test_evaluate.py).

## 3. Calibration

- Platt (sigmoid) calibration is fitted on **out-of-fold** ensemble scores
  (cross-fitting); the holdout is untouched.
- Reported in `metrics.json`: Brier score, log loss, ECE, reliability curve,
  calibration slope/intercept.
- The serving threshold is stored in `artifacts/model_ecommerce_ensemble/serving_config.json`
  and read by batch scoring and the API.

## 4. Evaluation protocol

```text
|---------------- TRAIN (80%) ----------------|---- HOLDOUT (20%) ----|
        |-- StratifiedKFold(5) OOF --|
```

1. Stratified 80/20 holdout.
2. StratifiedKFold CV on the training portion for OOF probabilities.
3. Cross-fitted Platt calibration on OOF scores.
4. Holdout: PR-AUC/ROC-AUC with bootstrap 95% CI, Brier/log loss, calibration,
   profit curve and SHAP.
5. Leakage ablation: PR-AUC with vs without `recency_days` and
   `order_session_gap_days`; the without-recency value is the primary metric.

## 5. Risk tiers

| Tier | Calibrated probability |
|---|---|
| `low` | `p < 0.40` |
| `medium` | `0.40 <= p < 0.70` |
| `high` | `p >= 0.70` |

Boundaries are left-closed; `0.40` is **medium** and `0.70` is **high**. They
are stored in `serving_config.json` and enforced by tests.
