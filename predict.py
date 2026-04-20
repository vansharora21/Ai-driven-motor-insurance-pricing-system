from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_loader import build_inference_frame
from src.frequency_model import predict_frequency
from src.model_artifacts import load_artifacts
from src.pricing_engine import calculate_premium
from src.severity_model import predict_severity


def predict_premiums(input_df: pd.DataFrame, artifacts: dict[str, Any] | None = None) -> pd.DataFrame:
    """Score arbitrary policy rows using the saved production artifacts."""
    artifacts = artifacts or load_artifacts()
    metadata = artifacts["metadata"]

    prepared_df = build_inference_frame(input_df, metadata)
    annual_frequency = predict_frequency(artifacts["frequency_model"], prepared_df)
    expected_severity = predict_severity(artifacts["severity_model"], prepared_df)

    scored = calculate_premium(
        prepared_df,
        annual_frequency,
        expected_severity,
        pricing_config=metadata.get("pricing_config"),
        risk_thresholds=metadata.get("risk_thresholds"),
    )
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with the trained motor pricing models.")
    parser.add_argument("--input", required=True, help="Path to a CSV file containing policy features.")
    parser.add_argument(
        "--output",
        default="results/premium_reports/predictions.csv",
        help="Where to save the scored predictions.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_df = pd.read_csv(input_path)
    scored_df = predict_premiums(input_df)
    scored_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path.resolve()}")


if __name__ == "__main__":
    main()
