"""
Verifies the full MLflow tracking stack works end-to-end.
Logs a dummy run with params, metrics and a tag, then prints the UI link.

Usage:
    uv run python infra/scripts/smoke_test.py
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
    mlflow.set_experiment(EXPERIMENT_NAME)

    try:
        with mlflow.start_run(run_name="smoke-test") as run:
            mlflow.log_param("model_type", "xgboost")
            mlflow.log_param("n_estimators", 200)
            mlflow.log_metric("auc_roc", 0.87)
            mlflow.log_metric("f1_score", 0.81)
            mlflow.set_tag("stage", "smoke-test")

            print(f"Run ID   : {run.info.run_id}")
            print(f"Run name : {run.info.run_name}")
            print(f"\nUI link  : {TRACKING_URI}/#/experiments/1/runs/{run.info.run_id}")
            print("\nSmoke test passed — MLflow tracking is working correctly.")

    except Exception as e:
        print(f"ERROR: Smoke test failed: {e}")
        print(f"       Is the MLflow server running at {TRACKING_URI}?")
        sys.exit(1)


if __name__ == "__main__":
    main()
