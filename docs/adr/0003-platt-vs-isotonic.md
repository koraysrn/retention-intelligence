# ADR 0003 — Platt scaling (cross-fitted) over isotonic calibration

## Context

The API exposes `churn_probability` and the profit curve uses calibrated
probabilities. With ~1,200 fixture rows, an isotonic calibrator has few
observations per bin and can overfit; it also produces stepwise probabilities.

## Decision

Use Platt (sigmoid) scaling fitted on **out-of-fold** ensemble scores. The
holdout set is never used to fit the calibrator. Report Brier, log loss,
reliability curve, ECE and calibration slope/intercept.

## Consequences

- Smooth, monotonic probabilities suitable for thresholding.
- Requires a reasonably sized OOF sample; on very small data the slope estimate
  is noisy, which is why the validation gate checks the slope range.
- If the dataset grows, revisit isotonic calibration and compare Brier/log loss
  on a held-out calibration set.
