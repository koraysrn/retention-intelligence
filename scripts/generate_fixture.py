"""Generate deterministic, schema-compatible synthetic e-commerce fixtures.

The real ``ecommerce_data.csv`` is not distributed with the repository (PII +
licensing). This script creates:

- ``ecommerce_sample.csv``: 1,200 rows, exact 27-column customer schema.
- ``orders_sample.csv``: transaction-level companion used to compute genuinely
  leak-free as-of features and outcome-window labels.

Both are deterministic (seed 42) and synthetic (CC0-1.0).

Usage:
    python -m scripts.generate_fixture --rows 1200
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "customer_id",
    "name",
    "email",
    "country",
    "age",
    "age_group",
    "signup_date",
    "marketing_opt_in",
    "total_orders",
    "total_spend_usd",
    "avg_order_value",
    "avg_discount_pct",
    "first_order_date",
    "last_order_date",
    "preferred_payment",
    "preferred_device_ord",
    "preferred_source",
    "top_category_bought",
    "avg_rating_given",
    "total_sessions",
    "preferred_device_sess",
    "preferred_source_sess",
    "first_session_date",
    "last_session_date",
    "has_abandoned_cart",
    "clv_tier",
    "is_repeat_customer",
]

COUNTRIES = ["TR", "US", "DE", "UK", "FR", "NL", "JP", "IN"]
DEVICES = ["mobile", "desktop", "tablet"]
SOURCES = ["organic", "email", "paid_search", "social", "direct"]
PAYMENTS = ["card", "paypal", "apple_pay", "bank_transfer"]
CATEGORIES = ["Beauty", "Sports", "Electronics", "Home", "Fashion"]
FIRST_NAMES = ["Alex", "Sam", "Jordan", "Taylor", "Casey", "Morgan", "Riley", "Avery"]
LAST_NAMES = ["Smith", "Lee", "Brown", "Garcia", "Kim", "Patel", "Nguyen", "Schmidt"]


def _age_group(age: int) -> str:
    if age < 25:
        return "18-24"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    return "55+"


def _clv_tier(total_orders: int, spend: float) -> str:
    if total_orders == 0:
        return "no_purchase"
    if spend >= 2000:
        return "vip"
    if spend >= 1000:
        return "high"
    if spend >= 300:
        return "medium"
    return "low"


def generate(rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic customer-level DataFrame with the production schema."""
    rng = np.random.default_rng(seed)
    # The fixture spans 2023-01-01 .. 2025-12-31. The recommended as-of date is
    # 90 days before the end, leaving a full outcome window.
    end = pd.Timestamp("2025-12-31 12:00:00")
    as_of = end - pd.Timedelta(days=90)
    start = pd.Timestamp("2023-01-01")

    records: list[dict] = []
    for i in range(rows):
        cid = 1000 + i
        first_name = FIRST_NAMES[i % len(FIRST_NAMES)]
        last_name = LAST_NAMES[(i // len(FIRST_NAMES)) % len(LAST_NAMES)]
        country = COUNTRIES[int(rng.integers(0, len(COUNTRIES)))]
        age = int(rng.integers(18, 70))
        signup = start + pd.Timedelta(days=int(rng.integers(0, 700)))
        marketing = bool(rng.random() < 0.55)

        # Engagement drives both the pre-cutoff features and the outcome.
        engagement = float(rng.beta(2.2, 2.0))  # 0..1
        total_sessions = int(1 + rng.poisson(6 + 18 * engagement))
        first_session = signup + pd.Timedelta(days=int(rng.integers(0, 3)))
        # Higher engagement -> more recent pre-cutoff session.
        days_before_cutoff = int(rng.integers(0, 120) * (1.0 - 0.75 * engagement))
        last_session = as_of - pd.Timedelta(days=days_before_cutoff)
        if last_session < first_session:
            last_session = first_session
        abandoned = int(rng.random() < (0.55 - 0.35 * engagement))

        # Purchase history. ~12% are zero-order customers before the cutoff.
        is_zero_order_pre = rng.random() < 0.12
        total_orders = 0
        first_order = pd.NaT
        last_order = pd.NaT
        avg_rating = np.nan
        avg_discount = np.nan
        avg_order_value = np.nan
        total_spend = 0.0
        payment = None
        device_ord = None
        source_ord = None
        category = None

        if not is_zero_order_pre:
            # 45% one-time buyers, 55% repeat buyers before the cutoff.
            repeat_pre = rng.random() < 0.55
            total_orders = int(rng.integers(2, 7)) if repeat_pre else 1
            first_order = signup + pd.Timedelta(days=int(rng.integers(1, 180)))
            if first_order > as_of - pd.Timedelta(days=10):
                first_order = as_of - pd.Timedelta(days=int(rng.integers(10, 60)))
            last_order = first_order
            if repeat_pre:
                span = int((as_of - first_order).days)
                last_order = first_order + pd.Timedelta(days=int(rng.integers(1, max(2, span + 1))))
                last_order = min(last_order, as_of - pd.Timedelta(days=int(rng.integers(1, 20))))
            avg_order_value = float(rng.lognormal(4.3, 0.45))
            total_spend = round(float(avg_order_value * total_orders), 2)
            avg_discount = round(float(rng.uniform(0, 15)), 2)
            avg_rating = float(np.clip(rng.normal(3.8 + 0.6 * engagement, 0.7), 1, 5))
            payment = str(rng.choice(PAYMENTS))
            device_ord = str(rng.choice(DEVICES))
            source_ord = str(rng.choice(SOURCES))
            category = str(rng.choice(CATEGORIES))

        # Outcome window: a purchase strictly after as_of and on/before end.
        # Probability rises with engagement and falls with abandoned cart.
        p_outcome = float(np.clip(0.08 + 0.72 * engagement - 0.12 * abandoned, 0.02, 0.95))
        orders_in_outcome = bool(rng.random() < p_outcome)
        if orders_in_outcome:
            outcome_date = as_of + pd.Timedelta(
                days=int(rng.integers(1, 90)),
                hours=int(rng.integers(0, 24)),
            )
            outcome_date = min(outcome_date, end)
            if pd.isna(first_order):
                # Zero-order before cutoff, converted in the outcome window.
                first_order = outcome_date
                last_order = outcome_date
                total_orders = 1
                avg_order_value = float(rng.lognormal(4.3, 0.45))
                total_spend = round(float(avg_order_value), 2)
                avg_discount = round(float(rng.uniform(0, 15)), 2)
                avg_rating = float(np.clip(rng.normal(4.2, 0.5), 1, 5))
                payment = str(rng.choice(PAYMENTS))
                device_ord = str(rng.choice(DEVICES))
                source_ord = str(rng.choice(SOURCES))
                category = str(rng.choice(CATEGORIES))
            else:
                total_orders += 1
                last_order = outcome_date
                total_spend = round(float(total_spend + rng.lognormal(4.3, 0.45)), 2)

        is_repeat = int(total_orders >= 2)

        records.append(
            {
                "customer_id": cid,
                "name": f"{first_name} {last_name}",
                "email": f"customer{cid}@example.test",
                "country": country,
                "age": age,
                "age_group": _age_group(age),
                "signup_date": signup.strftime("%Y-%m-%d"),
                "marketing_opt_in": marketing,
                "total_orders": total_orders,
                "total_spend_usd": round(float(total_spend), 2),
                "avg_order_value": round(float(avg_order_value), 2)
                if not np.isnan(avg_order_value)
                else None,
                "avg_discount_pct": avg_discount if not pd.isna(avg_discount) else None,
                "first_order_date": first_order.isoformat() if not pd.isna(first_order) else None,
                "last_order_date": last_order.isoformat() if not pd.isna(last_order) else None,
                "preferred_payment": payment,
                "preferred_device_ord": device_ord,
                "preferred_source": source_ord,
                "top_category_bought": category,
                "avg_rating_given": round(float(avg_rating), 2)
                if not pd.isna(avg_rating)
                else None,
                "total_sessions": total_sessions,
                "preferred_device_sess": str(rng.choice(DEVICES)),
                "preferred_source_sess": str(rng.choice(SOURCES)),
                "first_session_date": first_session.isoformat(),
                "last_session_date": last_session.isoformat(),
                "has_abandoned_cart": abandoned,
                "clv_tier": _clv_tier(total_orders, total_spend),
                "is_repeat_customer": is_repeat,
            }
        )

    return pd.DataFrame(records, columns=COLUMNS)


ORDER_COLUMNS = [
    "order_id",
    "customer_id",
    "order_date",
    "amount_usd",
    "discount_pct",
    "channel",
    "device",
    "category",
]


def generate_orders(customers: pd.DataFrame, as_of: pd.Timestamp, seed: int = 43) -> pd.DataFrame:
    """Reconstruct an order-level history from the customer fixture.

    The order history is what makes the as-of feature builder and the
    outcome-window label unambiguous. Every order is placed before or after the
    cutoff; the label only depends on orders strictly after it.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    order_idx = 0

    for row in customers.itertuples(index=False):
        if pd.isna(row.first_order_date):
            continue
        first = pd.Timestamp(row.first_order_date)
        last = pd.Timestamp(row.last_order_date)
        n_orders = int(row.total_orders)
        if n_orders <= 0:
            continue

        dates: list[pd.Timestamp] = []
        if n_orders == 1:
            dates = [last]
        else:
            pre_end = min(last, as_of)
            if first < pre_end:
                pre_n = n_orders if last <= as_of else max(1, n_orders - 1)
                span_days = max(1, (pre_end - first).days)
                offsets = sorted(
                    int(x) for x in rng.integers(0, span_days + 1, size=max(0, pre_n - 1))
                )
                dates = [first] + [first + pd.Timedelta(days=o) for o in offsets]
                dates = sorted(set(dates))[:pre_n]
                if last > as_of:
                    dates.append(last)
                else:
                    dates[-1] = last
            else:
                dates = [last]

        while len(dates) < n_orders:
            dates.append(dates[-1] + pd.Timedelta(days=int(rng.integers(1, 40))))
        dates = sorted(dates)[:n_orders]

        per_order = float(row.total_spend_usd) / max(1, n_orders)
        for order_date in dates:
            order_idx += 1
            rows.append(
                {
                    "order_id": f"O{order_idx:07d}",
                    "customer_id": int(row.customer_id),
                    "order_date": order_date.isoformat(),
                    "amount_usd": round(per_order * float(rng.uniform(0.8, 1.2)), 2),
                    "discount_pct": round(float(rng.uniform(0, 15)), 2),
                    "channel": str(rng.choice(SOURCES)),
                    "device": str(rng.choice(DEVICES)),
                    "category": str(rng.choice(CATEGORIES)),
                }
            )

    return pd.DataFrame(rows, columns=ORDER_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic e-commerce fixture")
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/fixtures/ecommerce_sample.csv"))
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    as_of = pd.Timestamp("2025-12-31 12:00:00") - pd.Timedelta(days=90)
    orders = generate_orders(df, as_of=as_of, seed=args.seed + 1)
    orders_path = args.out.with_name("orders_sample.csv")
    orders.to_csv(orders_path, index=False)

    for path in (args.out, orders_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
        print(f"Wrote {len(pd.read_csv(path))} rows -> {path} (sha256={digest[:16]}...)")

    print("as_of recommendation: 2025-10-02 (end - 90d), churn horizon: 90d")


if __name__ == "__main__":
    main()
