from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

DEFAULT_NUMERIC_VALUES = {
    EXPOSURE_COLUMN: 1.0,
    "VehPower": 6.0,
    "VehAge": 5.0,
    "DrivAge": 40.0,
    "BonusMalus": 50.0,
    "Density": 1000.0,
}
DEFAULT_CATEGORICAL_VALUES = {
    "VehBrand": "B1",
    "VehGas": "Regular",
    "Area": "C",
    "Region": "Ile-de-France",
}


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


def engineer_features(
    df: pd.DataFrame,
    numeric_defaults: dict[str, float] | None = None,
    categorical_defaults: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Apply consistent feature cleaning for both training and inference."""
    df = df.copy()

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

    df[EXPOSURE_COLUMN] = df[EXPOSURE_COLUMN].clip(lower=1e-6)
    df["VehPower"] = df["VehPower"].clip(lower=1)
    df["VehAge"] = df["VehAge"].clip(lower=0, upper=120)
    df["DrivAge"] = df["DrivAge"].clip(lower=18, upper=100)
    df["BonusMalus"] = df["BonusMalus"].clip(lower=1, upper=300)
    df["Density"] = df["Density"].clip(lower=0)
    df["LogDensity"] = np.log1p(df["Density"])

    for column in CATEGORICAL_FEATURES:
        default_value = _resolve_categorical_default(df, column, categorical_defaults)
        if column not in df.columns:
            df[column] = default_value
        df[column] = df[column].fillna(default_value).astype(str).str.strip()
        df.loc[df[column] == "", column] = default_value

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


def build_default_metadata(df: pd.DataFrame) -> dict[str, dict[str, float | str]]:
    """Capture training-time defaults for later inference."""
    numeric_defaults = {column: float(df[column].median()) for column in [EXPOSURE_COLUMN] + RAW_NUMERIC_FEATURES}
    categorical_defaults = {
        column: str(df[column].dropna().astype(str).mode().iloc[0]) if not df[column].dropna().empty else DEFAULT_CATEGORICAL_VALUES[column]
        for column in CATEGORICAL_FEATURES
    }
    return {
        "numeric_defaults": numeric_defaults,
        "categorical_defaults": categorical_defaults,
    }
