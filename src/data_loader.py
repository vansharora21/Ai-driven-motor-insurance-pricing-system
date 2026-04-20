from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_feature_config
from src.feature_engineering import (
    CANONICAL_COLUMN_ALIASES,
    CATEGORICAL_FEATURES,
    EXPOSURE_COLUMN,
    FREQUENCY_TARGET,
    INPUT_COLUMNS,
    POLICY_ID_COLUMN,
    SEVERITY_TARGET,
    engineer_features,
)

FEATURE_CONFIG = get_feature_config()
EXPOSURE_LOWER_BOUND = float(FEATURE_CONFIG["exposure_lower_bound"])

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREQUENCY_PATH = PROJECT_ROOT / "data" / "freMTPL2freq.csv"
DEFAULT_SEVERITY_PATH = PROJECT_ROOT / "data" / "freMTPL2sev.csv"

FREQUENCY_REQUIRED_COLUMNS = [
    POLICY_ID_COLUMN,
    FREQUENCY_TARGET,
    EXPOSURE_COLUMN,
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "VehBrand",
    "VehGas",
    "Area",
    "Density",
    "Region",
]
SEVERITY_REQUIRED_COLUMNS = [POLICY_ID_COLUMN, SEVERITY_TARGET]


class DataValidationError(ValueError):
    """Raised when dataset or inference input validation fails."""


def _validate_columns(df: pd.DataFrame, required_columns: list[str], dataset_name: str) -> None:
    """Ensure all required columns exist before any type coercion or modeling steps."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise DataValidationError(
            f"{dataset_name} is missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )


def _validate_numeric_columns(df: pd.DataFrame, numeric_columns: list[str], dataset_name: str) -> None:
    """Ensure numeric columns are not entirely non-numeric after coercion."""
    non_numeric_columns: list[str] = []
    for column in numeric_columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().sum() == 0 and df[column].notna().sum() > 0:
            non_numeric_columns.append(column)

    if non_numeric_columns:
        raise DataValidationError(
            f"{dataset_name} contains non-numeric values in expected numeric columns: {non_numeric_columns}"
        )


def _validate_missing_values(
    df: pd.DataFrame,
    columns: list[str],
    dataset_name: str,
    allow_missing: set[str] | None = None,
) -> None:
    """Reject missing values in required columns unless explicitly allowed."""
    allow_missing = allow_missing or set()
    missing_counts = {
        column: int(df[column].isna().sum())
        for column in columns
        if column not in allow_missing and int(df[column].isna().sum()) > 0
    }
    if missing_counts:
        raise DataValidationError(
            f"{dataset_name} has missing values in required columns: {missing_counts}. "
            "Please impute or remove invalid rows before training/inference."
        )


def validate_dataframe_schema(
    df: pd.DataFrame,
    dataset_name: str,
    required_columns: list[str],
    numeric_columns: list[str],
    allow_missing: set[str] | None = None,
) -> None:
    """Validate schema consistency for datasets used in training or inference."""
    _validate_columns(df, required_columns, dataset_name)
    _validate_numeric_columns(df, numeric_columns, dataset_name)
    _validate_missing_values(df, required_columns, dataset_name, allow_missing=allow_missing)


def load_frequency_data(filepath: str | Path = DEFAULT_FREQUENCY_PATH) -> pd.DataFrame:
    """Load the raw freMTPL2 frequency dataset."""
    df = pd.read_csv(filepath)
    validate_dataframe_schema(
        df,
        dataset_name="Frequency dataset",
        required_columns=FREQUENCY_REQUIRED_COLUMNS,
        numeric_columns=[POLICY_ID_COLUMN, FREQUENCY_TARGET, EXPOSURE_COLUMN, "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"],
    )
    return df


def load_severity_data(filepath: str | Path = DEFAULT_SEVERITY_PATH) -> pd.DataFrame:
    """Load the raw freMTPL2 severity dataset."""
    df = pd.read_csv(filepath)
    validate_dataframe_schema(
        df,
        dataset_name="Severity dataset",
        required_columns=SEVERITY_REQUIRED_COLUMNS,
        numeric_columns=[POLICY_ID_COLUMN, SEVERITY_TARGET],
    )
    return df


def clean_frequency_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean policy-level frequency data and return model-ready policy records.

    Cleaning rules are intentionally conservative:
    - numeric columns are coerced with explicit clipping/filling
    - categorical fields are normalized to non-empty strings
    - engineered features are added through a single shared transformation path
    """
    frequency_df = df.copy()
    _validate_columns(frequency_df, FREQUENCY_REQUIRED_COLUMNS, "Frequency dataset")

    numeric_columns = [POLICY_ID_COLUMN, FREQUENCY_TARGET, EXPOSURE_COLUMN, "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
    for column in numeric_columns:
        frequency_df[column] = pd.to_numeric(frequency_df[column], errors="coerce")

    frequency_df[POLICY_ID_COLUMN] = frequency_df[POLICY_ID_COLUMN].round().astype("Int64")
    frequency_df = frequency_df.dropna(subset=[POLICY_ID_COLUMN]).copy()
    frequency_df[POLICY_ID_COLUMN] = frequency_df[POLICY_ID_COLUMN].astype(int)

    frequency_df[FREQUENCY_TARGET] = frequency_df[FREQUENCY_TARGET].fillna(0).clip(lower=0).round().astype(int)
    # Exposure is clipped to a positive floor to avoid divide-by-zero during frequency modeling.
    frequency_df[EXPOSURE_COLUMN] = frequency_df[EXPOSURE_COLUMN].fillna(1.0).clip(lower=EXPOSURE_LOWER_BOUND)

    for column in CATEGORICAL_FEATURES:
        frequency_df[column] = frequency_df[column].fillna("Unknown").astype(str).str.strip()
        frequency_df.loc[frequency_df[column] == "", column] = "Unknown"

    cleaned = engineer_features(frequency_df)
    cleaned["has_claim"] = (cleaned[FREQUENCY_TARGET] > 0).astype(int)
    cleaned["observed_claim_frequency"] = cleaned[FREQUENCY_TARGET] / cleaned[EXPOSURE_COLUMN]
    return cleaned


def clean_severity_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean claim-level severity data and keep strictly positive paid claims."""
    severity_df = df.copy()
    _validate_columns(severity_df, SEVERITY_REQUIRED_COLUMNS, "Severity dataset")

    severity_df[POLICY_ID_COLUMN] = pd.to_numeric(severity_df[POLICY_ID_COLUMN], errors="coerce").round().astype("Int64")
    severity_df[SEVERITY_TARGET] = pd.to_numeric(severity_df[SEVERITY_TARGET], errors="coerce")
    severity_df = severity_df.dropna(subset=[POLICY_ID_COLUMN, SEVERITY_TARGET]).copy()
    severity_df[POLICY_ID_COLUMN] = severity_df[POLICY_ID_COLUMN].astype(int)
    severity_df[SEVERITY_TARGET] = severity_df[SEVERITY_TARGET].clip(lower=0)
    severity_df = severity_df[severity_df[SEVERITY_TARGET] > 0].copy()
    return severity_df


def aggregate_severity_by_policy(severity_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate total claim amount by policy for portfolio-level pricing."""
    policy_severity_totals = severity_df.groupby(POLICY_ID_COLUMN, as_index=False)[SEVERITY_TARGET].sum()
    policy_severity_totals = policy_severity_totals.rename(columns={SEVERITY_TARGET: "TotalClaimAmount"})
    return policy_severity_totals


def prepare_model_datasets(
    frequency_path: str | Path = DEFAULT_FREQUENCY_PATH,
    severity_path: str | Path = DEFAULT_SEVERITY_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Load, clean, and align the freMTPL2 datasets for training.

    Returns:
        policy_df: one row per policy for frequency modeling and final pricing
        claim_df: one row per claim for severity modeling
        data_quality: summary of merge quality and dropped rows
    """
    frequency_df = clean_frequency_data(load_frequency_data(frequency_path))
    severity_df = clean_severity_data(load_severity_data(severity_path))

    severity_by_policy = aggregate_severity_by_policy(severity_df)
    policy_df = frequency_df.merge(severity_by_policy, on=POLICY_ID_COLUMN, how="left")
    policy_df["TotalClaimAmount"] = policy_df["TotalClaimAmount"].fillna(0.0)
    policy_df["AverageClaimAmount"] = np.where(
        policy_df[FREQUENCY_TARGET] > 0,
        policy_df["TotalClaimAmount"] / policy_df[FREQUENCY_TARGET].clip(lower=1),
        0.0,
    )

    # Severity model is trained at claim level, so each claim row must carry policy attributes.
    claim_level_features = frequency_df[[POLICY_ID_COLUMN] + INPUT_COLUMNS[1:] + ["LogDensity"]].copy()
    claim_df = severity_df.merge(claim_level_features, on=POLICY_ID_COLUMN, how="inner")
    claim_df = claim_df[claim_df[SEVERITY_TARGET] > 0].copy()

    data_quality = {
        "frequency_rows": int(len(frequency_df)),
        "severity_rows": int(len(severity_df)),
        "policies_with_claims": int((frequency_df[FREQUENCY_TARGET] > 0).sum()),
        "policies_with_paid_amounts": int((policy_df["TotalClaimAmount"] > 0).sum()),
        "positive_claim_count_without_paid_amount": int(
            ((policy_df[FREQUENCY_TARGET] > 0) & (policy_df["TotalClaimAmount"] <= 0)).sum()
        ),
        "severity_rows_without_matching_policy": int((~severity_df[POLICY_ID_COLUMN].isin(frequency_df[POLICY_ID_COLUMN])).sum()),
        "claim_rows_used_for_severity_model": int(len(claim_df)),
    }

    return policy_df, claim_df, data_quality


def build_inference_frame(df: pd.DataFrame, metadata: dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Prepare user-provided policy rows for inference with saved models.

    This function validates minimum required columns, supports alias-based
    column names, and applies the same feature engineering used during training.
    """
    if df is None or df.empty:
        raise DataValidationError("Input CSV is empty. Please upload a file with at least one policy row.")

    required_inference_columns = [EXPOSURE_COLUMN, "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region"]
    alias_map = CANONICAL_COLUMN_ALIASES

    missing_required_columns: list[str] = []
    for canonical_column in required_inference_columns:
        aliases = alias_map.get(canonical_column, [canonical_column])
        if not any(alias in df.columns for alias in aliases):
            missing_required_columns.append(canonical_column)

    if missing_required_columns:
        raise DataValidationError(
            "Input CSV is missing required pricing columns: "
            f"{missing_required_columns}. "
            "Expected core columns include Exposure, VehPower, VehAge, DrivAge, BonusMalus, VehBrand, VehGas, Area, Density, and Region."
        )

    defaults = metadata or {}
    numeric_defaults = defaults.get("numeric_defaults", {})
    categorical_defaults = defaults.get("categorical_defaults", {})
    prepared = engineer_features(
        df.copy(),
        numeric_defaults=numeric_defaults,
        categorical_defaults=categorical_defaults,
    )

    validate_dataframe_schema(
        prepared,
        dataset_name="Prepared inference dataset",
        required_columns=[POLICY_ID_COLUMN] + required_inference_columns,
        numeric_columns=[POLICY_ID_COLUMN, EXPOSURE_COLUMN, "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"],
        allow_missing={POLICY_ID_COLUMN},
    )

    if POLICY_ID_COLUMN not in prepared.columns:
        prepared[POLICY_ID_COLUMN] = np.arange(1, len(prepared) + 1)

    return prepared
