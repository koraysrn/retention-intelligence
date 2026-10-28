# ADR 0002 — Three-model soft-voting ensemble

## Context

A single LightGBM pipeline is simple and fast, but the project claims an
ensemble. On Python 3.15, CatBoost currently has no wheel, and a hard import
would break the clean-clone pipeline.

## Decision

Use a three-member soft-voting ensemble:

1. LightGBM
2. XGBoost
3. CatBoost when installed, otherwise scikit-learn
   `HistGradientBoostingClassifier` (`hist`)

Weights are derived from OOF PR-AUC. The ensemble is wrapped in a cross-fitted
Platt calibrator. All members share the same preprocessing contract and the
same as-of features.

## Consequences

- `make pipeline` works on Python 3.15 without CatBoost.
- `metrics.json.models` records which third member was actually used.
- The ensemble remains more expensive than a single model; for a small
  portfolio prototype this is acceptable, but a single well-calibrated LightGBM
  is a legitimate future simplification.
