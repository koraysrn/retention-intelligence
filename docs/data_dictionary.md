# Data Dictionary, Provenance and Limitations

## 1. Source data

| Item | Value |
|---|---|
| Production schema | `ecommerce_data.csv` (27 columns, customer level) |
| Committed fixture | [`data/fixtures/ecommerce_sample.csv`](../data/fixtures/ecommerce_sample.csv) |
| Order history fixture | [`data/fixtures/orders_sample.csv`](../data/fixtures/orders_sample.csv) |
| Rows | 1,200 customers / 3,251 orders |
| Seed | 42 (orders seed 43) |
| License | CC0-1.0, synthetic — no real customer records |
| Checksums | `ecommerce_sample.sha256`, `orders_sample.sha256` |
| As-of recommendation | `2025-10-02` (dataset end 2025-12-31 minus 90 days) |
| Materialize | `make data` (copies fixture to the project root) |
| Use real data | `DATA_URL=... python -m scripts.download_data` |

## 2. Customer columns

| Column | Type | Description |
|---|---|---|
| `customer_id` | int | Customer key (unique) |
| `name` | string | PII — dropped before modelling |
| `email` | string | PII — dropped before modelling |
| `country` | string | Country |
| `age` | int | Age |
| `age_group` | string | 18-24, 25-34, 35-44, 45-54, 55+ |
| `signup_date` | datetime | Registration date |
| `marketing_opt_in` | bool | Marketing consent (used by action safety) |
| `total_orders` | int | Lifetime order count (leaky; dropped) |
| `total_spend_usd` | float | Lifetime spend (post-hoc; dropped) |
| `avg_order_value` | float | Lifetime AOV (post-hoc; dropped) |
| `avg_discount_pct` | float | Average discount (post-hoc; dropped) |
| `first_order_date` / `last_order_date` | datetime | Order extremes |
| `preferred_payment` | string | Order preference (dropped) |
| `preferred_device_ord` / `preferred_source` | string | Order preference (dropped) |
| `top_category_bought` | string | Order preference (dropped) |
| `avg_rating_given` | float | Average rating |
| `total_sessions` | int | Lifetime session count |
| `preferred_device_sess` / `preferred_source_sess` | string | Session preference |
| `first_session_date` / `last_session_date` | datetime | Session extremes |
| `has_abandoned_cart` | int | Cart abandonment flag |
| `clv_tier` | string | Lifetime CLV tier (dropped) |
| `is_repeat_customer` | int | Legacy label (comparison only) |

## 3. Order columns (`orders_sample.csv`)

| Column | Description |
|---|---|
| `order_id` | Order key |
| `customer_id` | Customer key |
| `order_date` | Order timestamp (normalized to day for as-of features) |
| `amount_usd` | Order amount |
| `discount_pct` | Discount applied |
| `channel`, `device`, `category` | Order context |

## 4. Known limitations

1. The fixture is synthetic. Its effect sizes are illustrative, not a business
   forecast.
2. `total_sessions` and `avg_rating_given` are lifetime aggregates in the
   27-column export; the fixture provides order-level history but not
   session-level history. They are weaker leakage risks than order aggregates,
   but a production build should ship a session-event table for a fully as-of
   snapshot.
3. The aggregate fallback cannot know the last pre-cutoff order when
   `last_order_date` is after the cutoff. The exact order-history path is used
   whenever `orders_sample.csv` / `orders.parquet` is available.
4. No real pgvector/RAG corpus is distributed; the agent layer is a prototype.
