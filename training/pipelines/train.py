"""
Main training pipeline for the defect predictor.

Runs:
  1. Feature engineering
  2. Cross-validation
  3. Final model training
  4. Evaluation with plots
  5. Full MLflow logging (params, metrics, artifacts, model)

Usage:
    uv run python -m training.pipelines.train
    uv run python -m training.pipelines.train --run-name my-experiment-1
    uv run python -m training.pipelines.train --config configs/best_params.yaml --run-name tuned-final
"""
import argparse
import os
import mlflow
import mlflow.xgboost
import yaml
from dotenv import load_dotenv
from pathlib import Path

from training.features.feature_engineering import run_pipeline as build_features
from training.models.xgboost_model import build_model, cross_validate
from training.evaluation.evaluator import evaluate_and_log

load_dotenv()

DEFAULT_CONFIG_PATH = Path("configs/model_config.yaml")
ARTIFACT_DIR = Path("data/processed/artifacts")


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(run_name: str | None = None, config_path: Path = DEFAULT_CONFIG_PATH):
    config = load_config(config_path)
    print(f"\n--- Config : {config_path} ---")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", config["experiment"]["name"])
    registered_model_name = config["experiment"]["registered_model_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # ------------------------------------------------------------------ #
    # 1. Feature Engineering
    # ------------------------------------------------------------------ #
    print("\n--- Feature Engineering ---")
    X_train, X_test, y_train, y_test, feature_names, scaler = build_features(
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
    )

    # ------------------------------------------------------------------ #
    # 2. Build Model
    # ------------------------------------------------------------------ #
    xgb_params = config["xgboost"]
    model = build_model(xgb_params)

    # ------------------------------------------------------------------ #
    # 3. MLflow Run
    # ------------------------------------------------------------------ #
    with mlflow.start_run(run_name=run_name or "baseline-xgboost") as run:
        print(f"\n--- MLflow Run: {run.info.run_id} ---")

        # -- Log all config params --
        mlflow.log_params({
            "model_type": "xgboost",
            "n_estimators": xgb_params["n_estimators"],
            "max_depth": xgb_params["max_depth"],
            "learning_rate": xgb_params["learning_rate"],
            "subsample": xgb_params["subsample"],
            "colsample_bytree": xgb_params["colsample_bytree"],
            "test_size": config["training"]["test_size"],
            "cv_folds": config["training"]["cv_folds"],
            "n_features": len(feature_names),
            "train_size": len(X_train),
            "test_size_rows": len(X_test),
        })

        mlflow.set_tags({
            "dataset": "NASA-JM1",
            "phase": "1-baseline",
            "feature_set": "mccabe+halstead+derived",
            "config_file": config_path.name,
        })

        # -- Cross-validation --
        print("\n--- Cross-Validation ---")
        cv_metrics = cross_validate(model, X_train, y_train, cv_folds=config["training"]["cv_folds"])
        mlflow.log_metrics(cv_metrics)
        print(f"  CV AUC-ROC : {cv_metrics['cv_auc_roc_mean']:.4f} ± {cv_metrics['cv_auc_roc_std']:.4f}")
        print(f"  CV F1      : {cv_metrics['cv_f1_mean']:.4f} ± {cv_metrics['cv_f1_std']:.4f}")

        # -- Final training --
        print("\n--- Training Final Model ---")
        model.fit(X_train, y_train)

        # -- Evaluation --
        print("\n--- Evaluation on Held-Out Test Set ---")
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = evaluate_and_log(
            mlflow_run=run,
            y_true=y_test,
            y_pred_proba=y_pred_proba,
            feature_names=feature_names,
            importances=model.feature_importances_,
            artifact_dir=ARTIFACT_DIR,
        )
        print(f"  AUC-ROC       : {metrics['auc_roc']}")
        print(f"  Avg Precision : {metrics['avg_precision']}")
        print(f"  F1            : {metrics['f1']}")
        print(f"  Precision     : {metrics['precision']}")
        print(f"  Recall        : {metrics['recall']}")

        # -- Log scaler as artifact --
        if scaler is not None:
            mlflow.log_artifact("data/processed/scaler.pkl", artifact_path="preprocessor")

        # -- Log the model to MLflow Model Registry --
        print("\n--- Logging Model to Registry ---")
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name,
            input_example=X_test.iloc[:5],
        )

        print(f"\nRun complete.")
        print(f"  Run ID  : {run.info.run_id}")
        print(f"  UI link : {tracking_uri}/#/experiments/1/runs/{run.info.run_id}")

    return run.info.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config YAML")
    args = parser.parse_args()
    train(run_name=args.run_name, config_path=args.config)
