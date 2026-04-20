from __future__ import annotations

import pandas as pd

from src.feature_engineering import build_preprocessor, engineer_features


def test_engineer_features_creates_log_density(sample_policy_df: pd.DataFrame) -> None:
    engineered = engineer_features(sample_policy_df)
    assert "LogDensity" in engineered.columns
    assert (engineered["LogDensity"] >= 0).all()


def test_engineer_features_fills_empty_categories(sample_policy_df: pd.DataFrame) -> None:
    dirty_df = sample_policy_df.copy()
    dirty_df.loc[0, "VehBrand"] = ""
    engineered = engineer_features(dirty_df)
    assert engineered.loc[0, "VehBrand"] != ""


def test_build_preprocessor_transforms_feature_matrix(sample_policy_df: pd.DataFrame) -> None:
    engineered = engineer_features(sample_policy_df)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(engineered)
    assert transformed.shape[0] == len(engineered)
