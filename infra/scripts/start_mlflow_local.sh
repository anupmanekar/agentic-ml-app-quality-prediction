#!/bin/bash
# Start MLflow tracking server locally (Phase 1 / Phase 2 development)
# Uses SQLite as backend and local filesystem for artifacts

set -e

MLFLOW_PORT=${MLFLOW_PORT:-5000}
BACKEND_URI="sqlite:///$(pwd)/mlruns/mlflow.db"
ARTIFACT_ROOT="$(pwd)/mlartifacts"

mkdir -p mlartifacts

echo "Starting MLflow server on port $MLFLOW_PORT"
echo "  Backend store : $BACKEND_URI"
echo "  Artifact root : $ARTIFACT_ROOT"
echo ""
echo "After server starts, run:"
echo "  uv run python infra/scripts/setup_mlflow_experiment.py"
echo ""

mlflow server \
  --backend-store-uri "$BACKEND_URI" \
  --default-artifact-root "$ARTIFACT_ROOT" \
  --host 0.0.0.0 \
  --port "$MLFLOW_PORT"
