from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineering import engineer_features
from src.frequency_model import predict_frequency, train_frequency_model
from src.severity_model import predict_severity, train_severity_model


def test_frequency_model_prediction_shape(sample_policy_df: pd.DataFrame) -> None:
    train_df = engineer_features(sample_policy_df)
    model = train_frequency_model(train_df)
    predictions = predict_frequency(model, train_df)
    assert len(predictions) == len(train_df)
    assert (predictions > 0).all()


def test_severity_model_prediction_shape(sample_claim_df: pd.DataFrame) -> None:
    train_df = engineer_features(sample_claim_df)
    model = train_severity_model(train_df)
    predictions = predict_severity(model, train_df)
    assert len(predictions) == len(train_df)
    assert (predictions > 0).all()


def test_frequency_random_forest_prediction_shape(sample_policy_df: pd.DataFrame) -> None:
    train_df = engineer_features(sample_policy_df)
    model = train_frequency_model(train_df, model_name="random_forest")
    predictions = predict_frequency(model, train_df)
    assert len(predictions) == len(train_df)
    assert (predictions > 0).all()


def test_severity_invalid_model_name_raises(sample_claim_df: pd.DataFrame) -> None:
    train_df = engineer_features(sample_claim_df)
    with pytest.raises(ValueError, match="Unsupported severity_model"):
        train_severity_model(train_df, model_name="invalid_model")
