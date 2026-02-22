"""
Creates the MLflow experiment and validates the tracking server is reachable.
Run this once after starting the MLflow server.

Usage:
    uv run python infra/scripts/setup_mlflow_experiment.py
"""
import os
import sys
import mlflow
from dotenv import load_dotenv

load_dotenv()

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "software-quality-prediction")


def main():
    mlflow.set_tracking_uri(TRACKING_URI)

    # Verify server is reachable
    try:
        client = mlflow.tracking.MlflowClient()
        client.search_experiments()
    except Exception as e:
        print(f"ERROR: Cannot reach MLflow server at {TRACKING_URI}")
        print(f"       Make sure it is running: uv run bash infra/scripts/start_mlflow_local.sh")
        print(f"       Details: {e}")
        sys.exit(1)

    # Create experiment if it doesn't exist
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        print(f"Created experiment : {EXPERIMENT_NAME}")
        print(f"Experiment ID      : {experiment_id}")
    else:
        print(f"Experiment already exists : {EXPERIMENT_NAME}")
        print(f"Experiment ID             : {experiment.experiment_id}")
        print(f"Lifecycle stage           : {experiment.lifecycle_stage}")

    print(f"\nTracking URI : {TRACKING_URI}")
    print(f"UI           : {TRACKING_URI}/#/experiments/{experiment.experiment_id if experiment else experiment_id}")


if __name__ == "__main__":
    main()
