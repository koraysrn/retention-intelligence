# Churn Label, As-Of Date and Leakage Prevention

This is the single canonical definition used by the Python feature builder, the
dbt mart models, the model card and the A/B experiment. Earlier versions used
`churn = 1 - is_repeat_customer`; that definition has been removed from the
model target because it describes *whether a customer ever placed a second
order*, not whether they will churn in a future window.

## 1. Time windows

```text
                 as_of_date                         as_of_date + horizon_days
                     |                                          |
 observation window  |            outcome window                |
 (features)          |            (label only)                  |
 t <= as_of          |   as_of < t <= as_of + horizon_days      |
```

| Term | Definition |
|---|---|
| `as_of_date` | Observation cutoff. Must be supplied or inferred as `max(date) - horizon_days`. |
| Observation window | All events with `t <= as_of_date`. Only these may enter features. |
| Outcome window | `as_of_date < t <= as_of_date + horizon_days`. Only the label may use it. |
| `horizon_days` | Default **90** days; configurable via `--horizon-days` and `configs/config.yaml`. |

## 2. Label

```text
churn = 1  iff there is no purchase in (as_of_date, as_of_date + horizon_days]
churn = 0  iff there is at least one purchase in that window
```

The outcome window is **left-open, right-closed**: an order exactly at
`as_of_date` belongs to the observation window; an order exactly at
`as_of_date + horizon_days` belongs to the outcome window.

### Segment definitions

| Pre-as-of segment | Definition | Typical churn interpretation |
|---|---|---|
| Zero-order | No purchase at or before `as_of_date` | `churn = 1` unless they convert in the outcome window |
| One-time | Exactly one pre-as-of order | `churn = 1` if no second purchase in the window |
| Repeat | Two or more pre-as-of orders | `churn = 1` if no purchase in the window |

`pre_asof_segment` is an audit column, never a model feature. Segment-level
performance is reported separately in the model card.

## 3. Features (as-of snapshot)

All date-derived features use only pre-cutoff data:

| Feature | As-of rule |
|---|---|
| `has_purchase` | At least one order with `order_date <= as_of` |
| `days_to_first_order` | `first_pre_asof_order - signup_date` |
| `recency_days` | `as_of - last_pre_asof_order` (days) |
| `order_session_gap_days` | `last_pre_asof_order - last_pre_asof_session` |
| `session_recency_days` | `as_of - last_session_at_or_before_as_of` |
| `session_span_days` | last pre-as-of session − first pre-as-of session |
| `tenure_days` | `as_of - signup_date` |
| `days_to_first_session` | first session − signup date |

When a transaction-level order history is available (the committed fixture
includes `data/fixtures/orders_sample.csv`), the pre-as-of order dates are
computed exactly. If only the aggregate 27-column CSV is available, the builder
falls back to `last_order_date` as an outcome proxy and documents this as a
limitation; the `last_order_date` value after the cutoff is never used as a
feature.

## 4. Leakage prevention

Excluded from the model input:

- `total_orders` — determines the legacy repeat label almost exactly.
- `total_spend_usd`, `avg_order_value`, `avg_discount_pct` — post-hoc order
  aggregates.
- `order_span_days`, `clv_tier`, order-profile categories.
- `is_repeat_customer`, `churn`, `churn_legacy_repeat`,
  `ordered_in_outcome_window`, `pre_asof_segment`.
- PII (`name`, `email`) and raw dates.

The leakage ablation in `metrics.json` reports PR-AUC with and without
`recency_days` / `order_session_gap_days`. If recency is doing label-proxy work,
the report says so instead of hiding it; the primary reported metric is the
**without-recency** PR-AUC.

## 5. Validation protocol

- Customer-level snapshot → **stratified holdout** (80/20) and **StratifiedKFold
  CV**, not a temporal walk-forward split (which would create label shift).
- Platt calibration is cross-fitted on OOF scores; the holdout is never used to
  fit the calibrator.
- The holdout is used once for final metrics, calibration, profit curve and
  SHAP.
