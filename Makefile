# =============================================================================
# Retention Intelligence — frequently used commands
# Environment: environment.yml (Python 3.15), created by `make setup`.
# =============================================================================
SHELL := /bin/bash
ENV_NAME ?= retention-intelligence
MAMBA_ROOT ?= $(CURDIR)/.mamba
PYTHON ?= $(MAMBA_ROOT)/envs/$(ENV_NAME)/bin/python
DBT ?= $(MAMBA_ROOT)/envs/$(ENV_NAME)/bin/dbt
AS_OF ?=
HORIZON ?= 90

.PHONY: help setup data ingest dbt-run dbt-test dbt-build features train score \
        ab-analyze ab-experiment pipeline test lint format typecheck run-api demo \
        infra-up infra-down agents clean-artifacts

help: ## List commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## Create the Python 3.15 env, install deps, copy profiles/.env
	bash scripts/bootstrap.sh

data: ## Materialize ecommerce_data.csv from the committed synthetic fixture
	$(PYTHON) -m scripts.download_data

ingest: ## CSV + order history -> DuckDB
	$(PYTHON) -m scripts.ingest

dbt-run: ## Run dbt models
	$(DBT) run --project-dir dbt --profiles-dir dbt

dbt-test: ## Run dbt tests
	$(DBT) test --project-dir dbt --profiles-dir dbt

dbt-build: ## Run dbt models + tests
	$(DBT) build --project-dir dbt --profiles-dir dbt

features: ## Build as-of features (optional AS_OF=YYYY-MM-DD HORIZON=90)
	$(PYTHON) -m scripts.build_features \
		$(if $(AS_OF),--as-of $(AS_OF),) --horizon-days $(HORIZON)

train: ## Train the calibrated ensemble (single training path)
	$(PYTHON) -m scripts.train_ensemble

score: ## Batch risk scoring (reads serving_config.json)
	$(PYTHON) -m src.serving.batch_score

ab-analyze: ## Synthetic A/B analysis demo
	$(PYTHON) -m src.experiments.analysis

ab-experiment: ## A/B assignment + CUPED/sequential analysis
	$(PYTHON) -m scripts.run_ab_experiment

pipeline: ## End-to-end: data quality -> ingest -> dbt -> features -> train -> score -> A/B
	$(PYTHON) -m scripts.faz0_data_quality
	$(PYTHON) -m scripts.ingest
	$(DBT) build --project-dir dbt --profiles-dir dbt
	$(PYTHON) -m scripts.build_features $(if $(AS_OF),--as-of $(AS_OF),) --horizon-days $(HORIZON)
	$(PYTHON) -m scripts.train_ensemble
	$(PYTHON) -m src.serving.batch_score
	$(PYTHON) -m scripts.run_ab_experiment

test: ## Run the test suite with coverage
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing

lint: ## Ruff lint
	$(PYTHON) -m ruff check src scripts tests

format: ## Ruff format
	$(PYTHON) -m ruff format src scripts tests

typecheck: ## Mypy type check
	$(PYTHON) -m mypy src

run-api: ## Serving API on http://127.0.0.1:8010
	$(PYTHON) -m uvicorn src.serving.api:app --host $${SERVING_HOST:-127.0.0.1} --port $${SERVING_PORT:-8010} --reload

agents: ## Prototype agent workflow (not production)
	$(PYTHON) -m src.agents.orchestrator 1000

demo: data pipeline ## One-command demo (then: make run-api)
	@echo "Demo artifacts ready. Start the API with: make run-api"
	@echo "Admin key: $${API_KEY:-dev-admin-key} | Read key: $${API_READ_KEY:-dev-read-key}"

infra-up: ## Start local infrastructure (optional)
	docker compose up -d

infra-down: ## Stop local infrastructure
	docker compose down

clean-artifacts: ## Remove generated artifacts (keeps fixtures)
	rm -rf artifacts data/processed/*.parquet data/processed/*.duckdb
