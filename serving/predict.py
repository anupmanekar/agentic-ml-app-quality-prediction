"""
Prediction client for the MLflow model serving endpoint.

Sends feature vectors to the running model server and returns
defect probability scores.

The server must be running before calling this:
    uv run python -m serving.server

Usage:
    # Score a single sample interactively
    uv run python -m serving.predict

    # Score from a JSON file
    uv run python -m serving.predict --input data/processed/sample.json

    # Score from the test set (first N rows)
    uv run python -m serving.predict --from-test --rows 5
"""
import argparse
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path

SERVING_URL = "http://localhost:5001/invocations"
HEADERS = {"Content-Type": "application/json"}

# A realistic sample with typical JM1 feature values
SAMPLE_INPUT = {
    "loc": 45.0,
    "cyclomatic_complexity": 8.0,
    "essential_complexity": 4.0,
    "design_complexity": 5.0,
    "halstead_length": 210.0,
    "halstead_volume": 850.0,
    "halstead_level": 0.05,
    "halstead_difficulty": 20.0,
    "halstead_intelligence": 42.0,
    "halstead_effort": 17000.0,
    "halstead_time_est": 0.9,
    "time_to_program": 945.0,
    "code_lines": 38.0,
    "comment_lines": 4.0,
    "blank_lines": 3.0,
    "code_and_comment_lines": 2.0,
    "unique_operators": 15.0,
    "unique_operands": 28.0,
    "total_operators": 90.0,
    "total_operands": 120.0,
    "branch_count": 18.0,
    "comment_ratio": 0.0870,
    "blank_ratio": 0.0652,
    "operand_richness": 0.2295,
    "operator_richness": 0.1630,
    "complexity_ratio": 0.4444,
}


def build_payload(records: list[dict]) -> dict:
    """
    MLflow serving expects requests in dataframe_split format:
    {
        "dataframe_split": {
            "columns": [...],
            "data": [[...], [...]]
        }
    }
    """
    df = pd.DataFrame(records)
    return {
        "dataframe_split": {
            "columns": df.columns.tolist(),
            "data": df.values.tolist(),
        }
    }


def predict(records: list[dict]) -> list[float]:
    """
    Send records to the model server and return defect probabilities.
    """
    payload = build_payload(records)
    response = requests.post(SERVING_URL, headers=HEADERS, json=payload, timeout=10)

    if response.status_code != 200:
        raise RuntimeError(
            f"Prediction request failed: {response.status_code}\n{response.text}"
        )

    result = response.json()

    # MLflow returns predictions as {"predictions": [...]}
    predictions = result.get("predictions", result)
    if isinstance(predictions, list) and isinstance(predictions[0], list):
        # xgboost flavor returns [[prob_class0, prob_class1], ...]
        return [row[1] for row in predictions]
    return predictions


def print_results(records: list[dict], probabilities: list[float], threshold: float = 0.5):
    print(f"\n{'#':>3}  {'Defect Prob':>12}  {'Prediction':>12}  {'Risk Level'}")
    print("-" * 50)
    for i, (record, prob) in enumerate(zip(records, probabilities)):
        prediction = "DEFECT" if prob >= threshold else "clean"
        risk = "HIGH" if prob >= 0.7 else ("MEDIUM" if prob >= 0.4 else "LOW")
        print(f"  {i+1:>1}  {prob:>12.4f}  {prediction:>12}  {risk}")


def load_from_test(rows: int = 5) -> list[dict]:
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")
    sample = X_test.head(rows)
    print(f"Scoring {rows} rows from test set.")
    print(f"Actual labels: {y_test['defects'].head(rows).tolist()}")
    return sample.to_dict(orient="records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None,
                        help="Path to JSON file with input records")
    parser.add_argument("--from-test", action="store_true",
                        help="Score rows from the saved test set")
    parser.add_argument("--rows", type=int, default=5,
                        help="Number of test rows to score (used with --from-test)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for defect classification")
    args = parser.parse_args()

    if args.input:
        with open(args.input) as f:
            records = json.load(f)
        if isinstance(records, dict):
            records = [records]
    elif args.from_test:
        records = load_from_test(args.rows)
    else:
        print("Using built-in sample input.")
        records = [SAMPLE_INPUT]

    try:
        probabilities = predict(records)
        print_results(records, probabilities, threshold=args.threshold)
    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Cannot connect to model server at {SERVING_URL}")
        print(f"       Start it first: uv run python -m serving.server")
