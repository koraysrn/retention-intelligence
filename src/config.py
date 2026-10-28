"""Centralized configuration management.

Environment variables (`.env`) and YAML configuration files are resolved from a
single access point so that every module consumes the same settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = "dev"
    log_level: str = "INFO"

    # Paths
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    ecommerce_data: Path = PROJECT_ROOT / "ecommerce_data.csv"

    # Model / MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_artifact_root: str = "s3://mlflow/artifacts"

    # LLM
    llm_provider: str = "openai"  # openai | anthropic | azure | bedrock | ollama | deepseek | mock
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet"
    ollama_base_url: str = "http://localhost:11434"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"

    # Vector database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "rag"
    postgres_user: str = "rag"
    postgres_password: str = "rag_password"

    # Churn / risk thresholds
    churn_horizon_days: int = 90
    high_risk_threshold: float = 0.70
    medium_risk_threshold: float = 0.40

    # Serving / auth (prototype defaults; override in .env / secret manager)
    serving_host: str = "0.0.0.0"
    serving_port: int = 8010
    api_key: str = "dev-admin-key"
    api_read_key: str = "dev-read-key"

    # Action safety limits
    action_budget_limit: float = 10_000.0
    action_max_recipients: int = 100
    action_cost_per_coupon: float = 50.0

    # Monitoring
    psi_threshold: float = 0.2


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return cast(dict[str, Any], yaml.safe_load(f))


def get_settings() -> Settings:
    """Return the application settings in a singleton-like manner."""
    return Settings()


settings = get_settings()
