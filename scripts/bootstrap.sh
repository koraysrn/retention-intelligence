#!/usr/bin/env bash
# Clean-clone bootstrap for the retention-intelligence prototype.
#
# Uses environment.yml (Python 3.15 + conda-forge core) and pip for packages
# that do not yet publish Python 3.15 conda builds. Idempotent.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="retention-intelligence"
MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-$ROOT/.mamba}"
MAMBA="$ROOT/.tools/bin/micromamba"

if [ ! -x "$MAMBA" ]; then
  echo "Installing micromamba into .tools/ (no system conda required)..."
  mkdir -p .tools
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C .tools bin/micromamba
fi

export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"

if [ ! -x "$MAMBA_ROOT/envs/$ENV_NAME/bin/python" ]; then
  echo "Creating environment from environment.yml..."
  # The conda solve succeeds even if the pip section cannot build an optional
  # package on a brand-new Python; the explicit installs below are the source
  # of truth for the prototype.
  "$MAMBA" create -y -f environment.yml || true
fi

PY="$MAMBA_ROOT/envs/$ENV_NAME/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: Python environment was not created." >&2
  exit 1
fi

echo "Installing the project (editable, no dependency resolution)..."
"$PY" -m pip install --no-deps -e "$ROOT"

# mypy 2.x currently imports a pathspec module missing from the version dbt
# needs; pin the 1.x line for the prototype.
"$PY" -m pip install --only-binary=:all: "mypy>=1.11,<1.15" types-PyYAML || true

echo "Installing runtime packages that lack Python 3.15 conda builds..."
for pkg in lightgbm xgboost shap imbalanced-learn; do
  "$PY" -m pip install --only-binary=:all: "$pkg" || \
    echo "WARNING: optional package $pkg could not be installed; code falls back where possible."
done

# dbt-core 1.11 pins mashumaro<3.15, which crashes on Python 3.15. Install dbt
# first, then upgrade mashumaro with --no-deps (verified working combination).
if "$PY" -m pip install --only-binary=:all: "dbt-core==1.11.7" "dbt-duckdb==1.10.0"; then
  "$PY" -m pip install --only-binary=:all: --no-deps "mashumaro>=3.18" || true
  "$PY" -c "import dbt.cli.main" >/dev/null 2>&1 || \
    echo "WARNING: dbt is installed but not importable on this Python."
else
  echo "WARNING: dbt could not be installed; Python feature parity still works."
fi

if [ ! -f dbt/profiles.yml ]; then
  cp dbt/profiles.example.yml dbt/profiles.yml
  echo "Created dbt/profiles.yml from the example."
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example."
fi

echo "Setup complete. Activate with:"
echo "  source $MAMBA_ROOT/envs/$ENV_NAME/bin/activate"
echo "Then run: make data && make pipeline && make test"
