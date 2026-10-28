-- As-of customer feature snapshot (mart layer).
-- This is the auditable SQL counterpart of src/features/ecommerce.py and is
-- tested for parity against the Python builder (tests/test_feature_parity.py).
--
-- The cutoff and horizon are dbt vars so the model is reproducible:
--   dbt run --vars '{"as_of_date": "2025-10-02 11:00:00", "horizon_days": 90}'

{% set as_of = var('as_of_date', '2025-10-02 11:00:00') %}
{% set horizon_days = var('horizon_days', 90) %}

with customers as (
    select * from {{ ref('stg_customers') }}
),

orders_pre as (
    select
        customer_id,
        min(cast(order_date as date)) as first_order_asof,
        max(cast(order_date as date)) as last_order_asof,
        count(*)                      as order_count_asof
    from {{ source('raw', 'orders') }}
    where cast(order_date as date) <= cast('{{ as_of }}' as date)
    group by customer_id
),

enriched as (
    select
        c.customer_id,
        c.age,
        case c.age_group
            when '18-24' then 0
            when '25-34' then 1
            when '35-44' then 2
            when '45-54' then 3
            when '55+'   then 4
            else -1
        end as age_group_code,
        c.country,
        cast(c.marketing_opt_in as integer) as marketing_opt_in,
        date_diff('day', cast(c.signup_date as date), cast('{{ as_of }}' as date)) as tenure_days,
        c.total_sessions,
        date_diff(
            'day',
            cast(c.signup_date as date),
            least(coalesce(cast(c.first_session_date as date), cast('{{ as_of }}' as date)), cast('{{ as_of }}' as date))
        ) as days_to_first_session,
        date_diff(
            'day',
            least(coalesce(cast(c.first_session_date as date), cast('{{ as_of }}' as date)), cast('{{ as_of }}' as date)),
            least(coalesce(cast(c.last_session_date as date), cast('{{ as_of }}' as date)), cast('{{ as_of }}' as date))
        ) as session_span_days,
        date_diff(
            'day',
            least(coalesce(cast(c.last_session_date as date), cast('{{ as_of }}' as date)), cast('{{ as_of }}' as date)),
            cast('{{ as_of }}' as date)
        ) as session_recency_days,
        coalesce(c.preferred_device_sess, 'MISSING') as preferred_device_sess,
        coalesce(c.preferred_source_sess, 'MISSING') as preferred_source_sess,
        cast(c.has_abandoned_cart as integer) as has_abandoned_cart,
        case when o.first_order_asof is not null then 1 else 0 end as has_purchase,
        date_diff('day', cast(c.signup_date as date), o.first_order_asof) as days_to_first_order,
        date_diff('day', o.last_order_asof, cast('{{ as_of }}' as date)) as recency_days,
        date_diff(
            'day',
            least(coalesce(cast(c.last_session_date as date), cast('{{ as_of }}' as date)), cast('{{ as_of }}' as date)),
            o.last_order_asof
        ) as order_session_gap_days,
        case when c.avg_rating_given is not null then 1 else 0 end as has_rating,
        c.avg_rating_given,
        case
            when coalesce(o.order_count_asof, 0) = 0 then 'zero_order'
            when o.order_count_asof = 1 then 'one_time'
            else 'repeat'
        end as pre_asof_segment
    from customers c
    left join orders_pre o using (customer_id)
)

select * from enriched
