# ADR 0001 — Stratified holdout/CV instead of temporal walk-forward

## Context

The training data is a **customer-level as-of snapshot**, not a transaction
stream. The earlier code sorted by `last_order_date` and used `TimeSeriesSplit`
plus a 70/15/15 chronological split. That ordering is correlated with the label
(e.g. customers with a recent post-cutoff order have `churn=0`), so a temporal
split creates label shift and an unstable validation estimate.

## Decision

Use a stratified 80/20 holdout and `StratifiedKFold` CV on the customer
snapshot. The as-of cutoff already removes future information; the split is for
estimating generalisation, not for simulating time.

## Consequences

- Stable class balance in every fold.
- CV and holdout use the same protocol.
- If a true multi-cohort temporal dataset is added later, replace this with a
  cohort-based as-of split; do not reintroduce `TimeSeriesSplit` on the
  snapshot.
