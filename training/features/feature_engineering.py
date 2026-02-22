"""
Feature engineering for the NASA JM1 software defect dataset.

Raw features (McCabe + Halstead code metrics):
  loc             : Lines of code
  v(g)            : Cyclomatic complexity (McCabe)
  ev(g)           : Essential complexity
  iv(g)           : Design complexity
  n               : Total operators + operands (Halstead length)
  v               : Halstead volume
  l               : Halstead program level
  d               : Halstead difficulty
  i               : Halstead intelligence
  e               : Halstead effort
  b               : Halstead time estimator
  t               : Time to program
  lOCode          : Number of code lines
  lOComment       : Number of comment lines
  lOBlank         : Number of blank lines
  locCodeAndComment: Lines with both code and comment
  uniq_Op         : Unique operators
  uniq_Opnd       : Unique operands
  total_Op        : Total operators
  total_Opnd      : Total operands
  branchCount     : Number of branches

Derived features (created here):
  comment_ratio   : lOComment / (loc + 1)
  blank_ratio     : lOBlank / (loc + 1)
  operand_richness: uniq_Opnd / (total_Opnd + 1)
  operator_richness: uniq_Op / (total_Op + 1)
  complexity_ratio: ev(g) / (v(g) + 1)   — essential vs total complexity

Target:
  defects         : 1 (defect-prone) or 0 (clean)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

# Columns that map directly to meaningful names
RENAME_MAP = {
    "v(g)": "cyclomatic_complexity",
    "ev(g)": "essential_complexity",
    "iv(g)": "design_complexity",
    "n": "halstead_length",
    "v": "halstead_volume",
    "l": "halstead_level",
    "d": "halstead_difficulty",
    "i": "halstead_intelligence",
    "e": "halstead_effort",
    "b": "halstead_time_est",
    "t": "time_to_program",
    "lOCode": "code_lines",
    "lOComment": "comment_lines",
    "lOBlank": "blank_lines",
    "locCodeAndComment": "code_and_comment_lines",
    "uniq_Op": "unique_operators",
    "uniq_Opnd": "unique_operands",
    "total_Op": "total_operators",
    "total_Opnd": "total_operands",
    "branchCount": "branch_count",
}

TARGET_COL = "defects"


def load_raw(path: Path = RAW_DATA_DIR / "jm1.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Replace '?' placeholder missing values with NaN
    - Drop duplicate rows
    - Cast all feature columns to float
    """
    df = df.replace("?", np.nan)
    df = df.drop_duplicates()

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    df[feature_cols] = df[feature_cols].astype(float)

    # Drop rows where more than 50% of features are missing
    threshold = len(feature_cols) * 0.5
    df = df.dropna(thresh=int(threshold) + 1, subset=feature_cols)

    # Fill remaining missing values with column median
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=RENAME_MAP)


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'true'/'false' string target to binary int."""
    df[TARGET_COL] = df[TARGET_COL].map({"true": 1, "false": 0, True: 1, False: 0}).astype(int)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable ratio features that often help tree models."""
    df["comment_ratio"] = df["comment_lines"] / (df["loc"] + 1)
    df["blank_ratio"] = df["blank_lines"] / (df["loc"] + 1)
    df["operand_richness"] = df["unique_operands"] / (df["total_operands"] + 1)
    df["operator_richness"] = df["unique_operators"] / (df["total_operators"] + 1)
    df["complexity_ratio"] = df["essential_complexity"] / (df["cyclomatic_complexity"] + 1)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != TARGET_COL]


def run_pipeline(
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], StandardScaler | None]:
    """
    Full feature engineering pipeline.

    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    df = clean(df)
    df = rename_columns(df)
    df = encode_target(df)
    df = add_derived_features(df)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test), columns=feature_cols, index=X_test.index
        )
        joblib.dump(scaler, PROCESSED_DATA_DIR / "scaler.pkl")

    # Persist processed splits for downstream use
    X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

    print(f"Train size : {len(X_train)} rows")
    print(f"Test size  : {len(X_test)} rows")
    print(f"Features   : {len(feature_cols)}")
    print(f"Defect rate (train): {y_train.mean():.2%}")
    print(f"Defect rate (test) : {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test, feature_cols, scaler


if __name__ == "__main__":
    run_pipeline()
