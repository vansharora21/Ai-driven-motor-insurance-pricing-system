from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import DataValidationError, build_inference_frame, validate_dataframe_schema


def test_validate_dataframe_schema_detects_missing_required_column(sample_policy_df: pd.DataFrame) -> None:
    invalid_df = sample_policy_df.drop(columns=["VehPower"])

    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_dataframe_schema(
            invalid_df,
            dataset_name="Frequency dataset",
            required_columns=["IDpol", "ClaimNb", "Exposure", "VehPower"],
            numeric_columns=["IDpol", "ClaimNb", "Exposure", "VehPower"],
        )


def test_build_inference_frame_rejects_missing_input_columns() -> None:
    missing_columns_df = pd.DataFrame(
        {
            "Exposure": [0.6],
            "VehPower": [7],
            "VehAge": [4],
            "DrivAge": [40],
        }
    )

    with pytest.raises(DataValidationError, match="missing required pricing columns"):
        build_inference_frame(missing_columns_df)


def test_build_inference_frame_accepts_alias_columns() -> None:
    alias_df = pd.DataFrame(
        {
            "policy_id": [1001],
            "exposure": [0.8],
            "vehicle_power": [8],
            "vehicle_age": [5],
            "driver_age": [42],
            "bonus_malus": [55],
            "vehicle_brand": ["B12"],
            "fuel_type": ["Regular"],
            "area": ["C"],
            "population_density": [390],
            "region": ["Centre"],
        }
    )

    prepared = build_inference_frame(alias_df)
    assert "IDpol" in prepared.columns
    assert "Exposure" in prepared.columns
    assert prepared.loc[0, "IDpol"] == 1001
