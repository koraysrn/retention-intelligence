"""Phase 3 — End-to-end A/B experiment design and analysis pipeline.

Flow:
1. Read the actual baseline churn rate from the training set.
2. Compute the sample size (sample_size).
3. Build strata from customer features and perform stratified assignment (assignment).
4. Generate synthetic experiment outcomes with a deterministic seed (outcomes
   are simulated because no real campaign is executed).
5. Run statistical analysis and write a JSON report.

Usage: python -m scripts.run_ab_experiment
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from src.config import PROJECT_ROOT, load_yaml, settings
from src.data.loader import load_ecommerce_data
from src.experiments.analysis import analyze_experiment, simulate_outcomes
from src.experiments.assignment import (
    assign_groups,
    covariate_balance,
    srm_chi_square,
    strata_balance_summary,
)
from src.experiments.sample_size import sample_size_proportions

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 A/B experiment pipeline")
    parser.add_argument(
        "--features",
        type=Path,
        default=settings.data_processed / "customer_features.parquet",
    )
    parser.add_argument(
        "--training",
        type=Path,
        default=settings.data_processed / "training_set.parquet",
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "config.yaml")
    parser.add_argument(
        "--out", type=Path, default=settings.artifacts_dir / "ab_experiment_report.json"
    )
    parser.add_argument(
        "--effect", type=float, default=None, help="Relative treatment effect (default: MDE)"
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if not args.features.exists():
        raise FileNotFoundError(
            f"Feature set not found: {args.features}. Run build_features first."
        )
    if not args.training.exists():
        raise FileNotFoundError(
            f"Training set not found: {args.training}. Run build_features first."
        )

    cfg = load_yaml(args.config).get("experiment", {})
    treatment_ratio = float(cfg.get("treatment_ratio", 0.5))
    alpha = float(cfg.get("alpha", 0.05))
    power = float(cfg.get("power", 0.80))
    mde = float(cfg.get("mde_relative", 0.20))
    buffer = float(cfg.get("buffer", 0.20))
    incentive_cost = float(cfg.get("incentive_cost", 50.0))
    treatment_effect = mde if args.effect is None else args.effect

    # 1) Baseline churn rate
    training = pd.read_parquet(args.training)
    baseline_churn = float(training["churn"].mean())
    logger.info("Baseline churn rate: %.4f", baseline_churn)

    # 2) Sample size
    sample = sample_size_proportions(baseline_churn, mde, alpha, power, buffer)
    logger.info(
        "Required sample: %d per group, %d total",
        sample.n_per_group,
        sample.total_required,
    )

    # 3) Strata and assignment
    features = pd.read_parquet(args.features).copy()
    # ``total_spend_usd`` is deliberately excluded from the model feature set
    # (leakage prevention), so re-join it from the raw source for A/B
    # stratification and the revenue simulation below.
    raw = load_ecommerce_data(settings.ecommerce_data)[["customer_id", "total_spend_usd"]]
    features = features.merge(raw, on="customer_id", how="left")
    features["total_spend_usd"] = features["total_spend_usd"].fillna(0.0)
    features["spend_decile"] = pd.qcut(
        features["total_spend_usd"], 10, labels=False, duplicates="drop"
    )
    # If batch scores exist, also expose the model risk decile for reporting.
    scores_path = settings.data_processed / "scores.parquet"
    if scores_path.exists():
        scores = pd.read_parquet(scores_path)[["customer_id", "churn_probability"]]
        scores["customer_id"] = scores["customer_id"].astype(str)
        features["customer_id"] = features["customer_id"].astype(str)
        features = features.merge(scores, on="customer_id", how="left")
        features["risk_decile"] = pd.qcut(
            features["churn_probability"], 10, labels=False, duplicates="drop"
        )
    features["tenure_bucket"] = pd.cut(
        features["tenure_days"],
        bins=[-1, 30, 90, 180, 365, 10**9],
        labels=["0-30", "31-90", "91-180", "181-365", "365+"],
    )
    assignment = assign_groups(features, treatment_ratio, seed=42, id_column="customer_id")
    logger.info("Assignment: %s", assignment.value_counts().to_dict())
    srm = srm_chi_square(assignment, treatment_ratio)
    balance = covariate_balance(
        features,
        assignment,
        covariates=[
            c
            for c in ("age", "tenure_days", "total_sessions", "total_spend_usd")
            if c in features.columns
        ],
    )

    # 4) Synthetic experiment outcomes (no real campaign is executed)
    avg_revenue = float(features["total_spend_usd"].mean())
    outcomes = simulate_outcomes(
        assignment,
        seed=args.seed,
        baseline_churn=baseline_churn,
        treatment_effect=treatment_effect,
        avg_revenue=avg_revenue,
        incentive_cost=incentive_cost,
    )

    # 5) Analysis
    avg_revenue = float(features["total_spend_usd"].mean())
    report = analyze_experiment(
        outcomes,
        alpha,
        mde_absolute=baseline_churn * mde,
        revenue_non_inferiority_margin=max(1.0, avg_revenue * 0.01),
    )
    report["sample_size"] = sample.to_dict()
    report["baseline_churn_rate"] = baseline_churn
    report["mde_relative"] = mde
    report["power"] = power
    report["alpha"] = alpha
    report["treatment_ratio"] = treatment_ratio
    report["assignment"] = {
        "method": "stratified_block_randomization",
        "strata_columns": [
            c for c in ("country", "spend_decile", "tenure_bucket") if c in features.columns
        ],
        "srm": srm,
        "balance": balance.to_dict(orient="records"),
        "strata": strata_balance_summary(features, assignment).to_dict(orient="records"),
    }
    report["simulated"] = True

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("A/B report saved: %s", args.out)

    print("\n=== Phase 3 A/B Experiment Report ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
