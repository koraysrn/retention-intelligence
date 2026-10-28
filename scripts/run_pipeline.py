"""End-to-end pipeline orchestration.

The canonical order (also used by ``make pipeline``):

1. scripts.faz0_data_quality
2. scripts.ingest
3. dbt build (run + tests)
4. scripts.build_features --as-of ... --horizon-days ...
5. scripts.train_ensemble
6. src.serving.batch_score
7. scripts.run_ab_experiment
8. src.agents.orchestrator (optional prototype; opt in with --with-agents)

Usage:
    python -m scripts.run_pipeline
    python -m scripts.run_pipeline --as-of 2025-10-02 --horizon-days 90
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DBT = Path(sys.executable).parent / "dbt"
DBT_CMD = str(DBT) if DBT.exists() else "dbt"


def run_step(name: str, cmd: list[str]) -> bool:
    logger.info("Step started: %s -> %s", name, " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("Step failed: %s (exit=%d)", name, result.returncode)
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end pipeline")
    parser.add_argument("--as-of", dest="as_of", default=None)
    parser.add_argument("--horizon-days", type=int, default=90)
    parser.add_argument("--with-agents", action="store_true")
    parser.add_argument("--only", default=None, help="Run a single named step")
    args = parser.parse_args()

    feature_args = [
        sys.executable,
        "-m",
        "scripts.build_features",
        "--horizon-days",
        str(args.horizon_days),
    ]
    if args.as_of:
        feature_args += ["--as-of", args.as_of]

    steps: list[tuple[str, list[str]]] = [
        ("faz0", [sys.executable, "-m", "scripts.faz0_data_quality"]),
        ("ingest", [sys.executable, "-m", "scripts.ingest"]),
        ("dbt", [DBT_CMD, "build", "--project-dir", "dbt", "--profiles-dir", "dbt"]),
        ("features", feature_args),
        ("train", [sys.executable, "-m", "scripts.train_ensemble"]),
        ("score", [sys.executable, "-m", "src.serving.batch_score"]),
        ("ab", [sys.executable, "-m", "scripts.run_ab_experiment"]),
    ]
    if args.with_agents:
        steps.append(("agents", [sys.executable, "-m", "src.agents.orchestrator", "1000"]))

    for name, cmd in steps:
        if args.only and name != args.only:
            continue
        if not run_step(name, cmd):
            sys.exit(1)
    logger.info("Pipeline completed.")


if __name__ == "__main__":
    main()
