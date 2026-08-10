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
from pathlib import Path

import pandas as pd

from src.data_loader import prepare_model_datasets
from src.db import fetch_training_data
from src.feature_engineering import POLICY_ID_COLUMN, engineer_features
from scripts.train import run_training_pipeline

# Minimum number of policy rows before a retrain is allowed. Retraining on a
# handful of quotes would overfit and silently degrade the production model.
DEFAULT_MIN_ROWS = 200

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FREQUENCY_PATH = PROJECT_ROOT / "data" / "freMTPL2freq.csv"
SEVERITY_PATH = PROJECT_ROOT / "data" / "freMTPL2sev.csv"


def _get(record: dict, key: str):
    """Case-insensitive lookup for Supabase records.

    Postgres folds unquoted identifiers to lowercase, so Supabase returns
    lowercase keys (exposure, vehpower, ...) while the rest of the codebase
    uses the canonical capitalized names (Exposure, VehPower, ...).
    """
    if key in record:
        return record[key]
    lowered = key.lower()
    for record_key, value in record.items():
        if record_key.lower() == lowered:
            return value
    return None


def _policy_id(record: dict, fallback: int) -> int:
    """Supabase ids are UUIDs; the pipeline needs a unique integer policy id."""
    raw = _get(record, "id")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return fallback


def _build_policy_frame(records: list[dict]) -> pd.DataFrame:
    """Convert Supabase records into a policy-level frame for frequency modeling.

    Each record becomes one policy row. `claim_occurred` (or a positive
    `claim_amount`) maps to ClaimNb; `premium_paid` is kept for later
    calibration work but is not a model feature.
    """
    rows = []
    for index, record in enumerate(records):
        claim_occurred = bool(_get(record, "claim_occurred")) or float(_get(record, "claim_amount") or 0) > 0
        rows.append(
            {
                POLICY_ID_COLUMN: _policy_id(record, index + 1),
                "ClaimNb": int(claim_occurred),
                "Exposure": float(_get(record, "Exposure") or 1.0),
                "VehPower": float(_get(record, "VehPower") or 6.0),
                "VehAge": float(_get(record, "VehAge") or 6.0),
                "DrivAge": float(_get(record, "DrivAge") or 40.0),
                "BonusMalus": float(_get(record, "BonusMalus") or 60.0),
                "VehBrand": str(_get(record, "VehBrand") or "B12"),
                "VehGas": str(_get(record, "VehGas") or "Regular"),
                "Area": str(_get(record, "Area") or "C"),
                "Density": float(_get(record, "Density") or 500.0),
                "Region": str(_get(record, "Region") or "Centre"),
                "premium_paid": float(_get(record, "premium_paid") or 0.0),
            }
        )
    return pd.DataFrame(rows)


def _build_severity_frame(records: list[dict]) -> pd.DataFrame:
    """Convert Supabase records into a claim-level frame for severity modeling.

    Only records with an actual paid claim amount produce a severity row.
    """
    rows = []
    for index, record in enumerate(records):
        claim_amount = float(_get(record, "claim_amount") or 0.0)
        if claim_amount <= 0:
            continue
        rows.append(
            {
                POLICY_ID_COLUMN: _policy_id(record, index + 1),
                "ClaimAmount": claim_amount,
                "Exposure": float(_get(record, "Exposure") or 1.0),
                "VehPower": float(_get(record, "VehPower") or 6.0),
                "VehAge": float(_get(record, "VehAge") or 6.0),
                "DrivAge": float(_get(record, "DrivAge") or 40.0),
                "BonusMalus": float(_get(record, "BonusMalus") or 60.0),
                "VehBrand": str(_get(record, "VehBrand") or "B12"),
                "VehGas": str(_get(record, "VehGas") or "Regular"),
                "Area": str(_get(record, "Area") or "C"),
                "Density": float(_get(record, "Density") or 500.0),
                "Region": str(_get(record, "Region") or "Centre"),
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

    # Apply the same feature engineering used by the original training pipeline
    # (adds LogDensity, fills/clips numeric columns, normalizes categoricals).
    policy_df = engineer_features(policy_df)
    policy_df["has_claim"] = (policy_df["ClaimNb"] > 0).astype(int)
    if not severity_df.empty:
        severity_df = engineer_features(severity_df)

    # Blend in the bundled freMTPL2 baseline when present. This preserves the
    # full feature space (Area A-F, brands, regions) that the frontend sends,
    # while the Supabase portfolios contribute real-world outcomes.
    if FREQUENCY_PATH.exists() and SEVERITY_PATH.exists():
        print("Including bundled freMTPL2 baseline data...")
        fremtpl_policy, fremtpl_severity, fremtpl_quality = prepare_model_datasets(
            FREQUENCY_PATH, SEVERITY_PATH
        )
        policy_df = pd.concat([policy_df, fremtpl_policy], ignore_index=True)
        severity_df = pd.concat([severity_df, fremtpl_severity], ignore_index=True)
        policy_df["has_claim"] = (policy_df["ClaimNb"] > 0).astype(int)
        print(
            f"  freMTPL2: {fremtpl_quality['frequency_rows']} policies, "
            f"{fremtpl_quality['claim_rows_used_for_severity_model']} claims"
        )

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