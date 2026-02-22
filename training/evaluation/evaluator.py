"""
Model evaluation utilities.
Computes classification metrics and generates plots as MLflow artifacts.
Designed to be used both during training (Phase 1) and by the
model_evaluator agent node (Phase 3).
"""
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


def compute_metrics(y_true: pd.Series, y_pred_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute standard binary classification metrics.

    Args:
        y_true       : Ground truth labels (0/1)
        y_pred_proba : Predicted probabilities for the positive class
        threshold    : Decision threshold for binary predictions

    Returns:
        Dictionary of metric name -> value
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    return {
        "auc_roc": round(roc_auc_score(y_true, y_pred_proba), 4),
        "avg_precision": round(average_precision_score(y_true, y_pred_proba), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
    }


def plot_roc_curve(y_true: pd.Series, y_pred_proba: np.ndarray) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Defect Predictor")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_precision_recall_curve(y_true: pd.Series, y_pred_proba: np.ndarray) -> plt.Figure:
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_prec = average_precision_score(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label=f"Avg Precision = {avg_prec:.3f}", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Defect Predictor")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true: pd.Series, y_pred_proba: np.ndarray, threshold: float = 0.5) -> plt.Figure:
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Clean", "Predicted Defect"])
    ax.set_yticklabels(["Actual Clean", "Actual Defect"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_feature_importance(feature_names: list[str], importances: np.ndarray, top_n: int = 20) -> plt.Figure:
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_features[::-1], top_importances[::-1])
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances")
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def evaluate_and_log(
    mlflow_run,
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    feature_names: list[str],
    importances: np.ndarray,
    artifact_dir: Path = Path("data/processed/artifacts"),
) -> dict:
    """
    Compute all metrics, generate all plots, log everything to an active MLflow run.
    Returns the metrics dict.
    """
    import mlflow

    metrics = compute_metrics(y_true, y_pred_proba)
    mlflow.log_metrics(metrics)

    plots = {
        "roc_curve.png": plot_roc_curve(y_true, y_pred_proba),
        "pr_curve.png": plot_precision_recall_curve(y_true, y_pred_proba),
        "confusion_matrix.png": plot_confusion_matrix(y_true, y_pred_proba),
        "feature_importance.png": plot_feature_importance(feature_names, importances),
    }

    for filename, fig in plots.items():
        path = save_figure(fig, artifact_dir / filename)
        mlflow.log_artifact(str(path), artifact_path="plots")

    return metrics
