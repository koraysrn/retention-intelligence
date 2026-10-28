# Model Card — Churn Ensemble (Prototype)

## Intended use

- Rank existing e-commerce customers by the probability of **no purchase in the
  next 90 days**.
- Prioritise retention campaigns and human review for high-risk, consenting
  customers.
- Provide calibrated probabilities for threshold-based budgeting.

## Out of scope

- Real-time individual credit/eligibility decisions.
- Customers without a pre-cutoff order/session history (zero-order customers are
  scored, but the model has little signal for them; report the segment
  separately).
- Causal uplift claims. The model is a **propensity/risk** model; the A/B layer
  measures campaign impact, but the score itself is not an uplift model.
- Production-grade RAG or autonomous coupon sending. The agent layer is a
  prototype and requires confirmation.

## Label definition

```text
as_of_date: observation cutoff
observation window: t <= as_of_date
outcome window: as_of_date < t <= as_of_date + 90 days
churn = 1 iff no purchase in the outcome window
```

Full definition: [`docs/churn_definition.md`](churn_definition.md).

## Features

18 model features: demographics (`age`, `age_group_code`, `country`),
acquisition (`marketing_opt_in`, `tenure_days`), session/engagement
(`total_sessions`, `days_to_first_session`, `session_span_days`,
`session_recency_days`, device/source), cart behaviour (`has_abandoned_cart`),
pre-cutoff purchase presence/timing (`has_purchase`, `days_to_first_order`,
`recency_days`, `order_session_gap_days`) and ratings (`has_rating`,
`avg_rating_given`).

Dropped: PII, raw dates, `total_orders`, `total_spend_usd`, `avg_order_value`,
`avg_discount_pct`, `order_span_days`, `clv_tier`, order-profile categories,
all label/audit columns.

## Leakage ablation

`metrics.json → leakage_ablation` reports PR-AUC with and without
`recency_days` / `order_session_gap_days`. The **primary reported metric is the
without-recency PR-AUC**; if the delta is material, recency is doing label-proxy
work and should not be presented as genuine predictive power.

## Metrics (fixture, synthetic)

The exact numbers are written to
`artifacts/model_ecommerce_ensemble/metrics.json` on every run. On the committed
1,200-row fixture (240-row holdout) the current run reports PR-AUC **0.726**
(95% CI 0.656–0.792) against a naive positive-rate baseline of 0.588, ROC-AUC
0.672, Brier 0.223, log loss 0.638 and calibration slope 0.742. The
without-recency ablation is 0.716 versus 0.718 with recency, so recency is not
doing meaningful label-proxy work on this synthetic fixture. These are
prototype numbers, not a real-customer claim. Always read the metric, CI and
calibration together; AUC alone is not reported.

| Metric | Where |
|---|---|
| PR-AUC / ROC-AUC + bootstrap 95% CI | `metrics.json → test_full`, `ci` |
| Brier / log loss / ECE / slope | `metrics.json → calibration` |
| Profit curve + serving threshold | `metrics.json → profit`, `serving_config.json` |
| Segment performance | build per `pre_asof_segment` from `training_set.parquet` |
| Leakage ablation | `metrics.json → leakage_ablation` |

## Segment performance

`pre_asof_segment` distinguishes `zero_order`, `one_time` and `repeat`
customers. Zero-order customers have no purchase history; their scores should
not be interpreted as strongly as repeat customers'. Segment metrics are not
hardcoded in this card; compute them from the holdout predictions and report
them with the same CI/calibration discipline.

## Calibration

Platt (sigmoid) calibration is cross-fitted on OOF scores. The holdout is used
only for evaluation. `serving_config.json` stores the selected threshold; the
API and batch scorer read it.

## Ethical / legal notes

- **KVKK/GDPR:** the source file contains name/email. They never enter the
  model or the LLM prompt; API read keys receive masked PII.
- **Consent:** coupon actions check `marketing_opt_in`, require explicit
  `confirm=true`, enforce a budget/recipient cap and an idempotency key, and
  write an audit log.
- **Synthetic data:** the committed fixture is CC0-1.0 and contains no real
  people. Results are illustrative.
- **No protected-attribute targeting:** demographics are used only as model
  features; the campaign layer does not segment on protected attributes.

## Retraining and monitoring

- `src/monitoring/retrain.py` calls `scripts.train_ensemble` (not the old
  single-model trainer), validates PR-AUC/Brier/calibration slope, and only then
  promotes a versioned artifact.
- Versions live under `artifacts/model_ecommerce_ensemble/versions/` with a
  `current` symlink; rollback means pointing `current` at the previous version.
- PSI drift checks are in `src/monitoring/drift.py`; they are a trigger, not a
  replacement for outcome monitoring.

## Known limitations

See [`docs/data_dictionary.md`](data_dictionary.md) §4. In short: synthetic
fixture, lifetime session/rating aggregates, no production pgvector/RAG, and
no real Kafka/Feast/GrowthBook integration.
