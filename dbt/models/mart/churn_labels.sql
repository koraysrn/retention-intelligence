-- Outcome-window churn label (mart layer).
-- churn = 1 iff there is no order in (as_of, as_of + horizon_days].
{% set as_of = var('as_of_date', '2025-10-02 11:00:00') %}
{% set horizon_days = var('horizon_days', 90) %}

with customers as (
    select customer_id, is_repeat_customer from {{ ref('stg_customers') }}
),

outcome as (
    select distinct customer_id
    from {{ source('raw', 'orders') }}
    where cast(order_date as date) > cast('{{ as_of }}' as date)
      and cast(order_date as date) <= cast('{{ as_of }}' as date) + interval '{{ horizon_days }} days'
)

select
    c.customer_id,
    case when o.customer_id is null then 1 else 0 end as churn,
    case when o.customer_id is null then 0 else 1 end as ordered_in_outcome_window,
    1 - c.is_repeat_customer as churn_legacy_repeat
from customers c
left join outcome o using (customer_id)
