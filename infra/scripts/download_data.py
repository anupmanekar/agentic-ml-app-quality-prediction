"""
Downloads the NASA JM1 software defect dataset from OpenML.
JM1 contains McCabe and Halstead code metrics for C functions,
labelled as defect-prone or clean.

Dataset reference:
  Shepperd, M. et al. (2013). Researcher Bias: The Use of Machine Learning
  in Software Defect Prediction. IEEE TSE.
  OpenML ID: 1053

Usage:
    uv run python infra/scripts/download_data.py
"""
import os
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_openml

RAW_DATA_DIR = Path("data/raw")
OUTPUT_FILE = RAW_DATA_DIR / "jm1.csv"


def download():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        print(f"Dataset already exists at {OUTPUT_FILE} — skipping download.")
        df = pd.read_csv(OUTPUT_FILE)
        print(f"Shape: {df.shape}")
        return

    print("Downloading NASA JM1 dataset from OpenML ...")
    dataset = fetch_openml(name="jm1", version=1, as_frame=True, parser="auto")

    df = dataset.frame
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved to      : {OUTPUT_FILE}")
    print(f"Shape         : {df.shape}")
    print(f"Columns       : {list(df.columns)}")
    print(f"Target column : {dataset.target_names[0] if dataset.target_names else 'defects'}")
    print(f"Class balance :\n{df['defects'].value_counts()}")


if __name__ == "__main__":
    download()
