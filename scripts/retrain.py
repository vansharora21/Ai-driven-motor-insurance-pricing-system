"""Retrain the pricing models on data collected via Supabase.

Pulls consented quotes (`training_data`) and insurer portfolios
(`b2b_portfolios`) from Supabase, converts them into the policy/severity
frames the training pipeline expects, then runs the same pipeline used for
the original freMTPL2 training.

Usage:
    python scripts/retrain.py                 # retrain on all collected data
    python scripts/retrain.py --min-rows 500  # refuse to retrain below a floor

The retrained artifacts overwrite models/*.joblib + model_metadata.json.
Commit them to the repo and redeploy to ship the new model.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src.db import fetch_training_data
from src.feature_engineering import INPUT_COLUMNS, POLICY_ID_COLUMN
from scripts.train import run_training_pipeline

# Minimum number of policy rows before a retrain is allowed. Retraining on a
# handful of quotes would overfit and silently degrade the production model.
DEFAULT_MIN_ROWS = 200


def _build_policy_frame(records: list[dict]) -> pd.DataFrame:
    """Convert Supabase records into a policy-level frame for frequency modeling.

    Each record becomes one policy row. `claim_occurred` (or a positive
    `claim_amount`) maps to ClaimNb; `premium_paid` is kept for later
    calibration work but is not a model feature.
    """
    rows = []
    for record in records:
        claim_occurred = bool(record.get("claim_occurred")) or float(record.get("claim_amount") or 0) > 0
        rows.append(
            {
                POLICY_ID_COLUMN: int(record.get("id", 0)) if isinstance(record.get("id"), int) else 0,
                "ClaimNb": int(claim_occurred),
                "Exposure": float(record.get("Exposure") or 1.0),
                "VehPower": float(record.get("VehPower") or 6.0),
                "VehAge": float(record.get("VehAge") or 6.0),
                "DrivAge": float(record.get("DrivAge") or 40.0),
                "BonusMalus": float(record.get("BonusMalus") or 60.0),
                "VehBrand": str(record.get("VehBrand") or "B12"),
                "VehGas": str(record.get("VehGas") or "Regular"),
                "Area": str(record.get("Area") or "C"),
                "Density": float(record.get("Density") or 500.0),
                "Region": str(record.get("Region") or "Centre"),
                "premium_paid": float(record.get("premium_paid") or 0.0),
            }
        )
    return pd.DataFrame(rows)


def _build_severity_frame(records: list[dict]) -> pd.DataFrame:
    """Convert Supabase records into a claim-level frame for severity modeling.

    Only records with an actual paid claim amount produce a severity row.
    """
    rows = []
    for record in records:
        claim_amount = float(record.get("claim_amount") or 0.0)
        if claim_amount <= 0:
            continue
        rows.append(
            {
                POLICY_ID_COLUMN: int(record.get("id", 0)) if isinstance(record.get("id"), int) else 0,
                "ClaimAmount": claim_amount,
                "Exposure": float(record.get("Exposure") or 1.0),
                "VehPower": float(record.get("VehPower") or 6.0),
                "VehAge": float(record.get("VehAge") or 6.0),
                "DrivAge": float(record.get("DrivAge") or 40.0),
                "BonusMalus": float(record.get("BonusMalus") or 60.0),
                "VehBrand": str(record.get("VehBrand") or "B12"),
                "VehGas": str(record.get("VehGas") or "Regular"),
                "Area": str(record.get("Area") or "C"),
                "Density": float(record.get("Density") or 500.0),
                "Region": str(record.get("Region") or "Centre"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain pricing models on Supabase-collected data.")
    parser.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS, help="Minimum policy rows required.")
    args = parser.parse_args()

    try:
        data = fetch_training_data()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    training_records = data["training_data"]
    b2b_records = data["b2b_portfolios"]
    total_policies = len(training_records) + len(b2b_records)

    print(f"Collected: {len(training_records)} consented quotes, {len(b2b_records)} B2B portfolio rows.")
    if total_policies < args.min_rows:
        print(
            f"Refusing to retrain: only {total_policies} policy rows "
            f"(minimum {args.min_rows}). Collect more data first.",
            file=sys.stderr,
        )
        sys.exit(1)

    policy_df = _build_policy_frame(training_records + b2b_records)
    severity_df = _build_severity_frame(training_records + b2b_records)

    if policy_df.empty:
        print("ERROR: no usable policy rows after conversion.", file=sys.stderr)
        sys.exit(1)

    # The pipeline expects the engineered feature columns to exist.
    for column in INPUT_COLUMNS:
        if column not in policy_df.columns:
            policy_df[column] = np.nan

    data_quality = {
        "frequency_rows": int(len(policy_df)),
        "severity_rows": int(len(severity_df)),
        "policies_with_claims": int((policy_df["ClaimNb"] > 0).sum()),
        "policies_with_paid_amounts": int((policy_df["premium_paid"] > 0).sum()),
        "positive_claim_count_without_paid_amount": 0,
        "severity_rows_without_matching_policy": 0,
        "claim_rows_used_for_severity_model": int(len(severity_df)),
    }

    print(f"Retraining on {len(policy_df)} policies / {len(severity_df)} claims...")
    result = run_training_pipeline(policy_df, severity_df, data_quality)

    print("Retraining complete.")
    print(f"Frequency model: {result['metrics']['frequency'].get('selected_model')}")
    print(f"Severity model:  {result['metrics']['severity'].get('selected_model')}")
    print("Artifacts written to models/. Commit them and redeploy to ship the new model.")


if __name__ == "__main__":
    main()