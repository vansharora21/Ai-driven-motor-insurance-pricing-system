from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import get_feature_config

POLICY_ID_COLUMN = "IDpol"
EXPOSURE_COLUMN = "Exposure"
FREQUENCY_TARGET = "ClaimNb"
SEVERITY_TARGET = "ClaimAmount"

RAW_NUMERIC_FEATURES = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
DERIVED_NUMERIC_FEATURES = ["LogDensity"]
NUMERIC_FEATURES = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "LogDensity"]
CATEGORICAL_FEATURES = ["VehBrand", "VehGas", "Area", "Region"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
INPUT_COLUMNS = [
    POLICY_ID_COLUMN,
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

FEATURE_CONFIG = get_feature_config()
DEFAULT_NUMERIC_VALUES = FEATURE_CONFIG["numeric_defaults"]
DEFAULT_CATEGORICAL_VALUES = FEATURE_CONFIG["categorical_defaults"]
NUMERIC_CLIP_BOUNDS = FEATURE_CONFIG["numeric_clip_bounds"]
EXPOSURE_LOWER_BOUND = float(FEATURE_CONFIG["exposure_lower_bound"])
OPTIONAL_SIMULATED_FEATURES = FEATURE_CONFIG["optional_simulated_features"]

CANONICAL_COLUMN_ALIASES = {
    POLICY_ID_COLUMN: [POLICY_ID_COLUMN, "policy_id", "driver_id"],
    EXPOSURE_COLUMN: [EXPOSURE_COLUMN, "exposure"],
    "VehPower": ["VehPower", "vehicle_power", "veh_power", "power"],
    "VehAge": ["VehAge", "vehicle_age", "veh_age"],
    "DrivAge": ["DrivAge", "driver_age", "age"],
    "BonusMalus": ["BonusMalus", "bonus_malus", "bonusmalus"],
    "VehBrand": ["VehBrand", "vehicle_brand", "veh_brand", "brand"],
    "VehGas": ["VehGas", "vehicle_fuel", "fuel_type", "veh_gas"],
    "Area": ["Area", "area"],
    "Density": ["Density", "density", "population_density"],
    "Region": ["Region", "region"],
    FREQUENCY_TARGET: [FREQUENCY_TARGET, "claim_nb", "claim_count"],
    SEVERITY_TARGET: [SEVERITY_TARGET, "claim_amount"],
}

SIMULATED_FEATURE_ALIASES = {
    "simulated_vehicle_type": ["simulated_vehicle_type", "vehicle_type"],
    "simulated_daily_mileage": ["simulated_daily_mileage", "daily_mileage", "annual_mileage"],
    "simulated_night_driving_level": ["simulated_night_driving_level", "night_driving_level"],
    "simulated_harsh_braking_level": ["simulated_harsh_braking_level", "harsh_braking_level"],
    "simulated_accidents_last_2yr": ["simulated_accidents_last_2yr", "accidents_last_2yr"],
    "simulated_claim_history": ["simulated_claim_history", "claim_history"],
}


def _copy_first_matching_column(df: pd.DataFrame, target_column: str, aliases: list[str]) -> None:
    if target_column in df.columns:
        return

    for alias in aliases:
        if alias in df.columns:
            df[target_column] = df[alias]
            return


def _normalize_input_schema(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for target_column, aliases in CANONICAL_COLUMN_ALIASES.items():
        _copy_first_matching_column(normalized, target_column, aliases)

    for target_column, aliases in SIMULATED_FEATURE_ALIASES.items():
        _copy_first_matching_column(normalized, target_column, aliases)

    return normalized


def _resolve_numeric_default(df: pd.DataFrame, column: str, numeric_defaults: dict[str, float] | None) -> float:
    if numeric_defaults and column in numeric_defaults:
        return float(numeric_defaults[column])

    candidate = pd.to_numeric(df.get(column), errors="coerce") if column in df.columns else pd.Series(dtype=float)
    if not candidate.empty:
        median = float(candidate.median())
        if not np.isnan(median):
            return median

    return float(DEFAULT_NUMERIC_VALUES[column])


def _resolve_categorical_default(df: pd.DataFrame, column: str, categorical_defaults: dict[str, str] | None) -> str:
    if categorical_defaults and column in categorical_defaults:
        return str(categorical_defaults[column])

    if column in df.columns:
        non_empty = df[column].dropna().astype(str).str.strip()
        non_empty = non_empty[non_empty != ""]
        if not non_empty.empty:
            return str(non_empty.mode().iloc[0])

    return DEFAULT_CATEGORICAL_VALUES[column]


def _apply_clip_bounds(df: pd.DataFrame, column: str) -> None:
    bounds = NUMERIC_CLIP_BOUNDS.get(column, {})
    lower = bounds.get("min")
    upper = bounds.get("max")
    df[column] = df[column].clip(lower=lower, upper=upper)


def _standardize_optional_simulated_features(df: pd.DataFrame) -> None:
    numeric_simulated_columns = {
        "simulated_daily_mileage",
        "simulated_accidents_last_2yr",
        "simulated_claim_history",
    }
    present_simulated_columns = [column for column in OPTIONAL_SIMULATED_FEATURES if column in df.columns]
    if not present_simulated_columns:
        return

    for column in present_simulated_columns:
        if column in numeric_simulated_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            df[column] = df[column].fillna("NotProvided").astype(str).str.strip()
            df.loc[df[column] == "", column] = "NotProvided"

    df["has_simulated_behavioral_inputs"] = df[present_simulated_columns].notna().any(axis=1).astype(int)


def engineer_features(
    df: pd.DataFrame,
    numeric_defaults: dict[str, float] | None = None,
    categorical_defaults: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Apply consistent feature cleaning for both training and inference.

    The freMTPL2 dataset columns remain the canonical model schema. Any older
    telematics-style inputs are preserved only as explicitly simulated fields and
    are not used by the pricing models.
    """
    df = _normalize_input_schema(df.copy())

    if POLICY_ID_COLUMN not in df.columns:
        df[POLICY_ID_COLUMN] = np.arange(1, len(df) + 1)

    df[POLICY_ID_COLUMN] = pd.to_numeric(df[POLICY_ID_COLUMN], errors="coerce")
    df[POLICY_ID_COLUMN] = df[POLICY_ID_COLUMN].fillna(pd.Series(np.arange(1, len(df) + 1), index=df.index))
    df[POLICY_ID_COLUMN] = df[POLICY_ID_COLUMN].round().astype(int)

    numeric_columns = [EXPOSURE_COLUMN] + RAW_NUMERIC_FEATURES
    for column in numeric_columns:
        if column not in df.columns:
            df[column] = _resolve_numeric_default(df, column, numeric_defaults)

        df[column] = pd.to_numeric(df[column], errors="coerce")
        default_value = _resolve_numeric_default(df, column, numeric_defaults)
        df[column] = df[column].fillna(default_value)
        _apply_clip_bounds(df, column)

    df[EXPOSURE_COLUMN] = df[EXPOSURE_COLUMN].clip(lower=EXPOSURE_LOWER_BOUND)
    df["LogDensity"] = np.log1p(df["Density"])

    for column in CATEGORICAL_FEATURES:
        default_value = _resolve_categorical_default(df, column, categorical_defaults)
        if column not in df.columns:
            df[column] = default_value
        df[column] = df[column].fillna(default_value).astype(str).str.strip()
        df.loc[df[column] == "", column] = default_value

    _standardize_optional_simulated_features(df)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Create the shared sklearn preprocessor used by both models."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_default_metadata(df: pd.DataFrame) -> dict[str, dict[str, float | str] | dict[str, list[str]] | list[str]]:
    """Capture training-time defaults for later inference."""
    numeric_defaults = {column: float(df[column].median()) for column in [EXPOSURE_COLUMN] + RAW_NUMERIC_FEATURES}
    categorical_defaults = {
        column: str(df[column].dropna().astype(str).mode().iloc[0]) if not df[column].dropna().empty else DEFAULT_CATEGORICAL_VALUES[column]
        for column in CATEGORICAL_FEATURES
    }
    return {
        "numeric_defaults": numeric_defaults,
        "categorical_defaults": categorical_defaults,
        "supported_input_aliases": {column: aliases for column, aliases in CANONICAL_COLUMN_ALIASES.items()},
        "optional_simulated_features": OPTIONAL_SIMULATED_FEATURES,
    }
