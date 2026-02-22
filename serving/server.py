"""
Starts the MLflow model serving endpoint for the champion defect-predictor.

MLflow model serving exposes a REST API at:
  POST http://localhost:5001/invocations

The request body must be JSON in MLflow's dataframe_split format.
See serving/predict.py for a client that calls this endpoint.

Usage:
    uv run python -m serving.server
    uv run python -m serving.server --port 5001
    uv run python -m serving.server --version 6   # serve a specific version
"""
import os
import sys
import argparse
import subprocess
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()

REGISTERED_MODEL = "defect-predictor"
CHAMPION_ALIAS = "champion"
DEFAULT_PORT = 5001


def get_model_uri(version: int | None = None) -> str:
    """
    Build the MLflow model URI to serve.
    Uses the champion alias by default; falls back to explicit version.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    if version:
        uri = f"models:/{REGISTERED_MODEL}/{version}"
        print(f"Serving model version : {version}")
    else:
        try:
            mv = client.get_model_version_by_alias(REGISTERED_MODEL, CHAMPION_ALIAS)
            uri = f"models:/{REGISTERED_MODEL}@{CHAMPION_ALIAS}"
            print(f"Serving champion alias : v{mv.version} ({REGISTERED_MODEL}@{CHAMPION_ALIAS})")
        except mlflow.exceptions.MlflowException:
            print(f"ERROR: No '{CHAMPION_ALIAS}' alias found for '{REGISTERED_MODEL}'.")
            print(f"       Run first: uv run python -m training.pipelines.promote")
            sys.exit(1)

    return uri


def serve(port: int = DEFAULT_PORT, version: int | None = None):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    model_uri = get_model_uri(version)

    print(f"\nStarting MLflow model server...")
    print(f"  Model URI    : {model_uri}")
    print(f"  Tracking URI : {tracking_uri}")
    print(f"  Endpoint     : http://localhost:{port}/invocations")
    print(f"\nPress Ctrl+C to stop.\n")

    # mlflow models serve starts a local REST server backed by the model
    cmd = [
        "mlflow", "models", "serve",
        "--model-uri", model_uri,
        "--port", str(port),
        "--env-manager", "local",    # use current venv, no conda/virtualenv creation
        "--host", "0.0.0.0",
    ]

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = tracking_uri

    subprocess.run(cmd, env=env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to serve on")
    parser.add_argument("--version", type=int, default=None,
                        help="Specific model version to serve (default: champion alias)")
    args = parser.parse_args()
    serve(port=args.port, version=args.version)
