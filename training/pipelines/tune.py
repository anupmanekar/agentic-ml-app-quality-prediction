"""
Hyperparameter tuning using Optuna + MLflow nested runs.

Each Optuna trial is logged as a child run nested under a single
parent "hyperparameter-tuning" run. This lets you compare all trials
side-by-side in the MLflow UI under one parent.

After tuning, the best params are saved to configs/best_params.yaml
and can be promoted into model_config.yaml for the final training run.

Search space is informed by manual tuning observations:
  - max_depth 3-5  (depth 6 overfit on 26 features)
  - lower learning_rate with more estimators tends to help
  - scale_pos_weight 3-5 for ~22% defect rate

Usage:
    uv run python -m training.pipelines.tune
    uv run python -m training.pipelines.tune --n-trials 50
    uv run python -m training.pipelines.tune --n-trials 20 --run-name my-tuning-run
"""
import argparse
import os
import yaml
import mlflow
import mlflow.xgboost
import optuna
from dotenv import load_dotenv
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from training.features.feature_engineering import run_pipeline as build_features

load_dotenv()

CONFIG_PATH = Path("configs/model_config.yaml")
BEST_PARAMS_PATH = Path("configs/best_params.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def objective(trial: optuna.Trial, X_train, y_train, cv_folds: int, parent_run_id: str) -> float:
    """
    Optuna objective function.
    Each call = one trial = one MLflow child run.
    Returns CV AUC-ROC (Optuna maximizes this).
    """
    # --- Define search space (informed by manual tuning) ---
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 5),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 3.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
        "gamma": trial.suggest_float("gamma", 0.0, 0.3),
    }

    model = XGBClassifier(
        **params,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
    cv_auc = float(cv_scores.mean())

    # --- Log each trial as a nested MLflow child run ---
    with mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True) as child_run:
        mlflow.log_params(params)
        mlflow.log_metric("cv_auc_roc_mean", round(cv_auc, 4))
        mlflow.log_metric("cv_auc_roc_std", round(float(cv_scores.std()), 4))
        mlflow.set_tag("trial_number", str(trial.number))

    return cv_auc


def tune(n_trials: int = 30, run_name: str = "hyperparameter-tuning"):
    config = load_config()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", config["experiment"]["name"])

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # --- Feature engineering (reuse processed data if already exists) ---
    print("\n--- Loading Features ---")
    X_train, X_test, y_train, y_test, feature_names, _ = build_features(
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
    )

    # --- Optuna study under a single parent MLflow run ---
    with mlflow.start_run(run_name=run_name) as parent_run:
        mlflow.set_tag("tuning_type", "optuna-bayesian")
        mlflow.set_tag("n_trials", str(n_trials))
        mlflow.set_tag("dataset", "NASA-JM1")
        mlflow.set_tag("phase", "1-tuning")
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("cv_folds", config["training"]["cv_folds"])

        print(f"\n--- Starting Optuna Study ({n_trials} trials) ---")
        print(f"    Parent run : {parent_run.info.run_id}")
        print(f"    Each trial will appear as a child run in MLflow UI\n")

        # Suppress Optuna's per-trial logs — MLflow UI is the source of truth
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(
                trial, X_train, y_train,
                config["training"]["cv_folds"],
                parent_run.info.run_id,
            ),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        # --- Log best results to parent run ---
        best = study.best_trial
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        mlflow.log_metric("best_cv_auc_roc", round(best.value, 4))

        print(f"\n--- Tuning Complete ---")
        print(f"  Best CV AUC-ROC : {best.value:.4f}")
        print(f"  Best params     :")
        for k, v in best.params.items():
            print(f"    {k:25s}: {v}")

        # --- Save best params to file ---
        best_params = {
            "experiment": config["experiment"],
            "training": config["training"],
            "xgboost": {
                "n_estimators": best.params["n_estimators"],
                "max_depth": best.params["max_depth"],
                "learning_rate": round(best.params["learning_rate"], 5),
                "subsample": round(best.params["subsample"], 3),
                "colsample_bytree": round(best.params["colsample_bytree"], 3),
                "scale_pos_weight": round(best.params["scale_pos_weight"], 2),
                "min_child_weight": best.params["min_child_weight"],
                "gamma": round(best.params["gamma"], 4),
                "eval_metric": "auc",
            },
            "thresholds": config["thresholds"],
        }

        with open(BEST_PARAMS_PATH, "w") as f:
            yaml.dump(best_params, f, default_flow_style=False, sort_keys=False)

        mlflow.log_artifact(str(BEST_PARAMS_PATH), artifact_path="tuning")

        print(f"\n  Best params saved to : {BEST_PARAMS_PATH}")
        print(f"  UI link : {tracking_uri}/#/experiments/1/runs/{parent_run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--run-name", type=str, default="hyperparameter-tuning", help="MLflow parent run name")
    args = parser.parse_args()
    tune(n_trials=args.n_trials, run_name=args.run_name)
