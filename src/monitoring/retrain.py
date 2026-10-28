"""Automatic retraining trigger + validation gate + model promotion.

The retrain path calls the **same** calibrated ensemble trainer as
``make train``; it never overwrites the ensemble with a single LightGBM model.

Promotion rules (configurable):
- PR-AUC >= ``min_pr_auc``
- Brier score <= ``max_brier``
- calibration slope in ``[slope_min, slope_max]``

When the gate fails, the candidate is kept under ``artifacts/retrain_candidate``
and the currently promoted ``current`` symlink is left untouched.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from src.config import settings
from src.monitoring.drift import detect_data_drift

logger = logging.getLogger(__name__)

DEFAULT_MIN_PR_AUC = 0.20
DEFAULT_MAX_BRIER = 0.30
DEFAULT_SLOPE_RANGE = (0.5, 2.0)


def validate_model(
    metrics: dict,
    min_pr_auc: float = DEFAULT_MIN_PR_AUC,
    max_brier: float = DEFAULT_MAX_BRIER,
    slope_range: tuple[float, float] = DEFAULT_SLOPE_RANGE,
) -> dict:
    """Return a validation-gate report for a metrics payload."""
    pr_auc = float(metrics.get("test_full", {}).get("pr_auc", 0.0))
    brier = float(metrics.get("test_full", {}).get("brier_score", 1.0))
    slope = float(metrics.get("calibration", {}).get("calibration_slope", 1.0))
    checks = {
        "pr_auc": pr_auc >= min_pr_auc,
        "brier": brier <= max_brier,
        "calibration_slope": slope_range[0] <= slope <= slope_range[1],
        "ensemble": bool(metrics.get("ensemble", False)),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "calibration_slope": slope,
        "thresholds": {
            "min_pr_auc": min_pr_auc,
            "max_brier": max_brier,
            "slope_range": list(slope_range),
        },
    }


def _promote_candidate(candidate_dir: Path, target_dir: Path) -> str | None:
    candidate_metrics = candidate_dir / "metrics.json"
    if not candidate_metrics.exists():
        return None
    metrics = json.loads(candidate_metrics.read_text(encoding="utf-8"))
    version = str(metrics.get("model_version", "unknown"))

    target_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "pipeline.joblib",
        "metrics.json",
        "serving_config.json",
        "shap_importance.csv",
        "explanations.json",
    ):
        src = candidate_dir / name
        if src.exists():
            shutil.copy2(src, target_dir / name)

    version_dir = target_dir / "versions" / version
    version_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "pipeline.joblib",
        "metrics.json",
        "serving_config.json",
        "shap_importance.csv",
        "explanations.json",
    ):
        src = candidate_dir / name
        if src.exists():
            shutil.copy2(src, version_dir / name)

    current = target_dir / "current"
    tmp = target_dir / ".current.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(Path("versions") / version, target_is_directory=True)
    if current.is_symlink() or current.is_file():
        current.unlink()
    elif current.exists():
        shutil.rmtree(current)
    tmp.replace(current)
    return version


def check_and_retrain(
    reference,
    current,
    threshold: float = 0.2,
    train_command: list[str] | None = None,
    dry_run: bool = True,
    min_pr_auc: float = DEFAULT_MIN_PR_AUC,
    max_brier: float = DEFAULT_MAX_BRIER,
) -> dict:
    """Run a drift check; retrain + gate + promote when drift is detected."""
    report = detect_data_drift(reference, current, threshold)
    payload = report.to_dict()
    payload["retrained"] = False
    payload["promoted"] = False

    if not report.drift_detected:
        return payload

    if dry_run:
        payload["retrain_skipped"] = True
        return payload

    candidate_dir = settings.artifacts_dir / "retrain_candidate"
    cmd = train_command or [
        sys.executable,
        "-m",
        "scripts.train_ensemble",
        "--out",
        str(candidate_dir),
    ]
    logger.info("Drift detected, starting retraining: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    payload["retrained"] = proc.returncode == 0
    payload["retrain_exit_code"] = proc.returncode
    payload["retrain_tail"] = (proc.stdout or proc.stderr or "")[-500:]

    metrics_path = candidate_dir / "metrics.json"
    if proc.returncode == 0 and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        gate = validate_model(metrics, min_pr_auc, max_brier)
        payload["validation_gate"] = gate
        if gate["passed"]:
            version = _promote_candidate(
                candidate_dir, settings.artifacts_dir / "model_ecommerce_ensemble"
            )
            payload["promoted"] = version is not None
            payload["model_version"] = version
        else:
            payload["promotion_blocked"] = True
            logger.warning("Retrain candidate blocked by validation gate: %s", gate["checks"])
    elif proc.returncode == 0:
        # Custom/mock training commands may not emit metrics; treat as successful smoke test.
        payload["validation_gate"] = {"passed": None, "reason": "no metrics artifact"}

    return payload
