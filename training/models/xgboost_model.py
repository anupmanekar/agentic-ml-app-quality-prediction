"""
XGBoost defect prediction model wrapper.
Keeps model construction and hyperparameter handling in one place.
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def build_model(params: dict) -> XGBClassifier:
    """
    Build an XGBClassifier from a params dict (loaded from model_config.yaml).
    scale_pos_weight handles the class imbalance (~80% clean, ~20% defect).
    """
    return XGBClassifier(
        n_estimators=params.get("n_estimators", 200),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        scale_pos_weight=params.get("scale_pos_weight", 3),  # ~80/20 imbalance
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )


def cross_validate(
    model: XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
) -> dict:
    """
    Run stratified k-fold CV and return mean/std for AUC-ROC and F1.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    auc_scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring="f1", n_jobs=-1)

    return {
        "cv_auc_roc_mean": round(float(np.mean(auc_scores)), 4),
        "cv_auc_roc_std": round(float(np.std(auc_scores)), 4),
        "cv_f1_mean": round(float(np.mean(f1_scores)), 4),
        "cv_f1_std": round(float(np.std(f1_scores)), 4),
    }
