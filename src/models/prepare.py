"""Feature preparation for model input.

Training and serving must share the same transformation, so feature preparation
is packaged as a ``sklearn`` ``ColumnTransformer`` and persisted alongside the
model (prevents train/serve skew).
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Identifier/PII columns are removed from the model input
IDENTIFIER_COLUMNS = {"customer_id", "name", "email"}

# Label / audit columns are removed from the model input
LABEL_COLUMNS = {
    "churn",
    "recency_churn",
    "is_repeat_customer",
    "churn_legacy_repeat",
    "ordered_in_outcome_window",
    "pre_asof_segment",
}

# Post-hoc aggregates and legacy label determinants (leaky)
LEAKY_COLUMNS = {
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
}


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "churn",
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into features (X) and target (y).

    Identifier, label, leaky, datetime and constant (nunique <= 1) columns are
    dropped from the features.

    Args:
        df: Training/analysis DataFrame.
        target_col: Name of the target column.

    Returns:
        A (X, y) pair.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    y = df[target_col].reset_index(drop=True)

    drop = [c for c in IDENTIFIER_COLUMNS | LABEL_COLUMNS | LEAKY_COLUMNS if c in df.columns]
    # Merge suffixes can recreate a label-derived audit column (e.g.
    # ``ordered_in_outcome_window_x``); drop any such column defensively.
    drop += [
        c
        for c in df.columns
        if any(c.startswith(f"{label}_") for label in LABEL_COLUMNS)
        or c.startswith("ordered_in_outcome_window")
    ]
    x = df.drop(columns=sorted(set(drop))).reset_index(drop=True)

    to_drop: list[str] = []
    for col in x.columns:
        if pd.api.types.is_datetime64_any_dtype(x[col]) or x[col].nunique(dropna=True) <= 1:
            to_drop.append(col)

    if to_drop:
        x = x.drop(columns=to_drop)

    return x, y


def numeric_features(x: pd.DataFrame) -> list[str]:
    return list(x.select_dtypes(include="number").columns)


def categorical_features(x: pd.DataFrame) -> list[str]:
    return list(x.select_dtypes(include=["object", "string", "category", "bool"]).columns)


def build_preprocessor(
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> ColumnTransformer:
    """Build a transformation pipeline for numeric/categorical columns.

    - Numeric: median imputation + StandardScaler (for logistic regression
      convergence; tree models are scale-invariant).
    - Categorical: ``MISSING`` imputation + OrdinalEncoder (unknown -> -1).
    """
    numeric_cols = numeric_cols or []
    categorical_cols = categorical_cols or []

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
                        (
                            "encoder",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")
