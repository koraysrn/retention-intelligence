"""End-to-end fixture pipeline: build features -> train -> score."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from src.serving.batch_score import load_serving_config, score_customers

pytestmark = pytest.mark.integration


def test_fixture_pipeline_end_to_end(tmp_path: Path) -> None:
    features_dir = tmp_path / "processed"
    model_dir = tmp_path / "model"
    scores_path = tmp_path / "scores.parquet"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.build_features",
            "--out",
            str(features_dir),
            "--horizon-days",
            "90",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.train_ensemble",
            "--data",
            str(features_dir / "training_set.parquet"),
            "--out",
            str(model_dir),
            "--n-splits",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["ensemble"] is True
    assert (model_dir / "pipeline.joblib").exists()
    assert (model_dir / "serving_config.json").exists()

    features = pd.read_parquet(features_dir / "customer_features.parquet")
    from joblib import load

    model = load(model_dir / "pipeline.joblib")
    config = load_serving_config(model_dir / "serving_config.json")
    scores = score_customers(features, model=model, serving_config=config)
    scores.to_parquet(scores_path, index=False)

    assert len(scores) == len(features)
    assert scores["churn_probability"].between(0, 1).all()
    assert set(scores["risk_tier"].unique()).issubset({"low", "medium", "high"})
    assert scores["model_version"].iloc[0] == metrics["model_version"]
