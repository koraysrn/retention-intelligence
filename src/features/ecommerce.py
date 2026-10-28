"""Customer-level, leakage-free feature engineering and churn labelling.

Label (see ``docs/churn_definition.md``)
----------------------------------------
- ``as_of_date`` is the observation cutoff.
- The observation window is ``t <= as_of_date``.
- The outcome window is ``as_of_date < t <= as_of_date + horizon_days``.
- ``churn = 1`` when the customer has **no purchase in the outcome window**.
- ``churn = 0`` when the customer purchased in the outcome window.

When a transaction-level ``orders`` frame is supplied, the label and every
order-derived feature are computed from actual order dates. When only the
aggregate 27-column customer export is available, we conservatively use
``last_order_date`` as an outcome-window proxy and never use order data after
the cutoff for features. This limitation is documented in the model card.

Features
--------
Only information available at ``as_of_date`` is used. Raw dates, PII, the label
and columns that determine the label (``total_orders`` and order-value
derivatives) never enter the model input.
"""

from __future__ import annotations

import pandas as pd

ID_COLUMN = "customer_id"
LABEL_COLUMN = "is_repeat_customer"
TARGET_COLUMN = "churn"
RECENCY_CHURN_COLUMN = "recency_churn"
DEFAULT_HORIZON_DAYS = 90

# PII / identifier columns - never enter the model input
PII_COLUMNS = ["name", "email"]

# Columns that determine the legacy repeat label or are post-hoc aggregates.
LEAKY_COLUMNS = [
    "total_orders",
    "total_spend_usd",
    "avg_order_value",
    "avg_discount_pct",
    "order_span_days",
    "clv_tier",
    "preferred_payment",
    "preferred_device_ord",
    "preferred_source",
    "top_category_bought",
]

RAW_DATE_COLUMNS = [
    "signup_date",
    "first_order_date",
    "last_order_date",
    "first_session_date",
    "last_session_date",
]

AGE_GROUP_ORDER = ["18-24", "25-34", "35-44", "45-54", "55+"]

# Categorical columns encoded with OrdinalEncoder inside the preprocessor
STRING_CATEGORICALS = [
    "country",
    "preferred_device_sess",
    "preferred_source_sess",
]

# Final feature columns (deterministic order - train/serve share the schema).
FEATURE_COLUMNS = [
    # Demographics
    "age",
    "age_group_code",
    "country",
    # Acquisition / marketing
    "marketing_opt_in",
    "tenure_days",
    # Session / engagement (snapshot at as_of)
    "total_sessions",
    "days_to_first_session",
    "session_span_days",
    "session_recency_days",
    "preferred_device_sess",
    "preferred_source_sess",
    # Cart behaviour
    "has_abandoned_cart",
    # Purchase PRESENCE (pre-cutoff) + first order timing + order recency
    "has_purchase",
    "days_to_first_order",
    "recency_days",
    "order_session_gap_days",
    # Rating
    "has_rating",
    "avg_rating_given",
]

# Columns used only for audit / segmentation, never as model input.
# NOTE: outcome-window flags stay in the label frame only; putting them in the
# feature table would be direct target leakage.
AUDIT_COLUMNS = ["pre_asof_segment"]


def _ensure_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast date columns to day-normalized datetime (invalid values become NaT)."""
    out = df.copy()
    for col in RAW_DATE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out


def _ensure_order_dates(orders: pd.DataFrame | None) -> pd.DataFrame | None:
    if orders is None:
        return None
    out = orders.copy()
    if "order_date" not in out.columns:
        raise ValueError("orders must contain an 'order_date' column")
    out["order_date"] = pd.to_datetime(out["order_date"], errors="coerce").dt.normalize()
    if "customer_id" in out.columns:
        out["customer_id"] = out["customer_id"].astype(str)
    return out


def _max_observed_date(df: pd.DataFrame) -> pd.Timestamp:
    maxima: list[pd.Timestamp] = []
    for col in RAW_DATE_COLUMNS:
        if col in df.columns and df[col].notna().any():
            maxima.append(pd.Timestamp(df[col].max()))
    if not maxima:
        raise ValueError("No date column found; cannot infer as_of_date")
    return max(maxima)


def resolve_as_of(
    df: pd.DataFrame,
    as_of_date: pd.Timestamp | str | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> pd.Timestamp:
    """Resolve and validate the observation cutoff.

    When ``as_of_date`` is omitted we use ``max(observed_date) - horizon_days``
    so that the outcome window is fully contained in the data.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    if as_of_date is not None:
        return pd.Timestamp(as_of_date).normalize()
    return (_max_observed_date(_ensure_datetimes(df)) - pd.Timedelta(days=horizon_days)).normalize()


def _legacy_repeat_churn(df: pd.DataFrame) -> pd.Series | None:
    """The old ``1 - is_repeat_customer`` definition, kept only for comparison."""
    if LABEL_COLUMN not in df.columns:
        return None
    return (1 - df[LABEL_COLUMN].astype(int)).rename("churn_legacy_repeat")


def _order_window_flags(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    horizon_days: int,
    orders: pd.DataFrame | None,
) -> pd.Series:
    """Return 1 when an order exists in ``(as_of, as_of + horizon]``."""
    outcome_end = as_of + pd.Timedelta(days=horizon_days)
    ids = df[ID_COLUMN].astype(str)

    if orders is not None and not orders.empty:
        in_window = orders.loc[
            (orders["order_date"] > as_of) & (orders["order_date"] <= outcome_end),
            "customer_id",
        ].astype(str)
        return ids.isin(set(in_window)).astype(int)

    # Aggregate fallback: last_order_date is the only observable outcome proxy.
    last_order = pd.to_datetime(df["last_order_date"], errors="coerce")
    flags = last_order.notna() & (last_order > as_of) & (last_order <= outcome_end)
    return flags.astype(int)


def build_churn_labels(
    df: pd.DataFrame,
    as_of_date: pd.Timestamp | str | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    orders: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate the outcome-window ``churn`` label.

    ``churn = 1`` iff there is no purchase in ``(as_of, as_of + horizon_days]``.
    The legacy repeat-buyer label is returned as ``churn_legacy_repeat`` for
    comparison only and never used by the model.
    """
    if ID_COLUMN not in df.columns:
        raise ValueError(f"Identifier column not found: {ID_COLUMN}")

    out = _ensure_datetimes(df)
    orders = _ensure_order_dates(orders)
    as_of = resolve_as_of(out, as_of_date, horizon_days)
    ordered = _order_window_flags(out, as_of, horizon_days, orders)

    labels = pd.DataFrame(
        {
            ID_COLUMN: out[ID_COLUMN].astype(str),
            TARGET_COLUMN: (1 - ordered).astype(int),
            "ordered_in_outcome_window": ordered.astype(int),
        }
    )
    legacy = _legacy_repeat_churn(out)
    if legacy is not None:
        labels["churn_legacy_repeat"] = legacy.to_numpy()

    labels.attrs["as_of_date"] = as_of.isoformat()
    labels.attrs["horizon_days"] = int(horizon_days)
    labels.attrs["definition"] = "no purchase in (as_of, as_of + horizon_days]"
    return labels


def build_customer_features(
    df: pd.DataFrame,
    as_of_date: pd.Timestamp | str | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    orders: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the as-of customer feature snapshot (leakage-free).

    Args:
        df: Raw customer-level DataFrame (27-column schema).
        as_of_date: Observation cutoff. Defaults to ``max(date) - horizon``.
        horizon_days: Outcome window length used only to resolve the default.
        orders: Optional transaction-level frame. When supplied, recency and
            first-order timing are computed from real pre-cutoff orders.

    Returns:
        DataFrame containing ``customer_id`` + ``FEATURE_COLUMNS`` + audit
        columns (audit columns are dropped by the model preparation layer).
    """
    out = _ensure_datetimes(df)
    orders = _ensure_order_dates(orders)
    as_of = resolve_as_of(out, as_of_date, horizon_days)
    out["_cid"] = out[ID_COLUMN].astype(str)

    # --- As-of order history ------------------------------------------------
    if orders is not None and not orders.empty:
        pre = orders[orders["order_date"] <= as_of]
        grouped = pre.groupby("customer_id")["order_date"]
        first_order = grouped.min().rename("_first_order_asof")
        last_order = grouped.max().rename("_last_order_asof")
        order_count = grouped.size().rename("_order_count_asof")
        order_frame = pd.concat([first_order, last_order, order_count], axis=1).reset_index()
        order_frame = order_frame.rename(columns={"customer_id": "_cid"})
        out = out.merge(order_frame, on="_cid", how="left")
        out["_first_order_asof"] = pd.to_datetime(out["_first_order_asof"])
        out["_last_order_asof"] = pd.to_datetime(out["_last_order_asof"])
        out["_order_count_asof"] = out["_order_count_asof"].fillna(0).astype(int)
    else:
        # Aggregate fallback: a post-cutoff last order is unknown at the cutoff.
        out["_first_order_asof"] = out["first_order_date"].where(out["first_order_date"] <= as_of)
        out["_last_order_asof"] = out["last_order_date"].where(out["last_order_date"] <= as_of)
        out["_order_count_asof"] = (out["_first_order_asof"].notna()).astype(int)

    # --- Session history, truncated at the cutoff ---------------------------
    out["_first_session_asof"] = out["first_session_date"].where(out["first_session_date"] <= as_of)
    # Clamping (not using the future value) keeps the feature stable when
    # post-cutoff session rows are edited.
    out["_last_session_asof"] = out["last_session_date"].clip(upper=as_of)

    # --- Binary indicators --------------------------------------------------
    out["has_purchase"] = out["_first_order_asof"].notna().astype(int)
    out["has_rating"] = out["avg_rating_given"].notna().astype(int)
    out["marketing_opt_in"] = out["marketing_opt_in"].astype(int)
    out["has_abandoned_cart"] = out["has_abandoned_cart"].astype(int)

    # --- Ordinal encoding ---------------------------------------------------
    age_map = {g: i for i, g in enumerate(AGE_GROUP_ORDER)}
    out["age_group_code"] = out["age_group"].map(age_map).fillna(-1).astype(int)

    # --- Time-derived features (all relative to as_of) ----------------------
    signup = out["signup_date"]
    out["tenure_days"] = (as_of - signup).dt.days.clip(lower=0)
    out["days_to_first_order"] = (out["_first_order_asof"] - signup).dt.days
    out["days_to_first_session"] = (out["_first_session_asof"] - signup).dt.days
    out["session_span_days"] = (out["_last_session_asof"] - out["_first_session_asof"]).dt.days
    out["session_recency_days"] = (as_of - out["_last_session_asof"]).dt.days
    out["recency_days"] = (as_of - out["_last_order_asof"]).dt.days
    out["order_session_gap_days"] = (out["_last_order_asof"] - out["_last_session_asof"]).dt.days

    # --- Segment (audit only; based on pre-cutoff order presence) -----------
    out["pre_asof_segment"] = "zero_order"
    out.loc[out["_order_count_asof"] == 1, "pre_asof_segment"] = "one_time"
    out.loc[out["_order_count_asof"] >= 2, "pre_asof_segment"] = "repeat"

    # --- Missing categorical values are flagged as "MISSING" -----------------
    for col in STRING_CATEGORICALS:
        out[col] = out[col].fillna("MISSING").astype(str)

    features = out[[ID_COLUMN] + FEATURE_COLUMNS + AUDIT_COLUMNS].copy()
    features[ID_COLUMN] = features[ID_COLUMN].astype(str)
    features.attrs["as_of_date"] = as_of.isoformat()
    features.attrs["horizon_days"] = int(horizon_days)
    return features


def build_dataset(
    df: pd.DataFrame,
    as_of_date: pd.Timestamp | str | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    orders: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build features, labels and the joined training set for one as-of date."""
    as_of = resolve_as_of(df, as_of_date, horizon_days)
    features = build_customer_features(df, as_of, horizon_days, orders)
    labels = build_churn_labels(df, as_of, horizon_days, orders)
    recency_labels = build_recency_churn_labels(df, horizon_days, as_of)
    training = (
        features.merge(labels, on=ID_COLUMN, how="inner")
        .merge(recency_labels, on=ID_COLUMN, how="inner")
        .copy()
    )
    training.attrs["as_of_date"] = as_of.isoformat()
    training.attrs["horizon_days"] = int(horizon_days)
    return {
        "features": features,
        "labels": labels,
        "training_set": training,
        "as_of_date": as_of.isoformat(),
        "horizon_days": int(horizon_days),
    }


def build_recency_churn_labels(
    df: pd.DataFrame,
    horizon_days: int = 365,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Legacy recency label kept for diagnostics only.

    ``recency_churn = 1`` when the last observed order is older than
    ``horizon_days`` or missing. It is **not** the model target.
    """
    out = _ensure_datetimes(df)
    if reference_date is None:
        reference_date = _max_observed_date(out)

    last_order = out["last_order_date"]
    churned = last_order.isna() | ((reference_date - last_order).dt.days > horizon_days)
    return pd.DataFrame(
        {
            ID_COLUMN: out[ID_COLUMN].astype(str),
            RECENCY_CHURN_COLUMN: churned.astype(int),
        }
    )
