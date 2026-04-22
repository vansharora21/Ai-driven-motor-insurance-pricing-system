from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_poisson_deviance, mean_squared_error
from sklearn.pipeline import Pipeline

from src.config import get_feature_config, get_model_config
from src.feature_engineering import EXPOSURE_COLUMN, FREQUENCY_TARGET, MODEL_FEATURES, build_preprocessor
from src.model_factory import create_frequency_regressor

FEATURE_CONFIG = get_feature_config()
EXPOSURE_LOWER_BOUND = float(FEATURE_CONFIG["exposure_lower_bound"])


def train_frequency_model(
    df: pd.DataFrame,
    alpha: float | None = None,
    max_iter: int | None = None,
    model_name: str | None = None,
    model_config: dict[str, Any] | None = None,
) -> Pipeline:
    """
    Train a model for claim frequency per exposure unit.

    The target is ClaimNb / Exposure and exposure is passed as sample weight,
    matching standard actuarial frequency modeling practice.
    """
    resolved_model_config = model_config.copy() if model_config is not None else get_model_config()
    if model_name is not None:
        resolved_model_config["frequency_model"] = model_name

    regressor = create_frequency_regressor(
        model_config=resolved_model_config,
        alpha=alpha,
        max_iter=max_iter,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor),
        ]
    )

    exposure_values = df[EXPOSURE_COLUMN].clip(lower=EXPOSURE_LOWER_BOUND)
    target_frequency = df[FREQUENCY_TARGET] / exposure_values
    model.fit(df[MODEL_FEATURES], target_frequency, regressor__sample_weight=exposure_values)
    return model


def predict_frequency(model: Pipeline, df: pd.DataFrame) -> pd.Series:
    """Predict annualized claim frequency for each policy row."""
    predicted_frequency = model.predict(df[MODEL_FEATURES])
    predicted_frequency = np.clip(predicted_frequency, 1e-9, None)
    return pd.Series(predicted_frequency, index=df.index, name="predicted_annual_frequency")


def predict_claim_count(model: Pipeline, df: pd.DataFrame) -> pd.Series:
    """Convert annual frequency into expected claim count for the policy exposure."""
    annual_frequency = predict_frequency(model, df)
    return annual_frequency * df[EXPOSURE_COLUMN]


def evaluate_frequency_model(model: Pipeline, df: pd.DataFrame) -> dict[str, Any]:
    """Evaluate holdout claim-count performance using RMSE and Poisson deviance."""
    regressor_name = model.named_steps["regressor"].__class__.__name__
    observed_claim_count = df[FREQUENCY_TARGET].astype(float)
    predicted_claim_count = predict_claim_count(model, df).clip(lower=1e-9)

    rmse = float(np.sqrt(mean_squared_error(observed_claim_count, predicted_claim_count)))
    poisson_deviance = float(mean_poisson_deviance(observed_claim_count, predicted_claim_count))

    return {
        "model_name": regressor_name,
        "evaluation_split": "test",
        "sample_size": int(len(df)),
        "metrics": {
            "rmse": rmse,
            "poisson_deviance": poisson_deviance,
        },
        "distribution": {
            "observed_mean": float(observed_claim_count.mean()),
            "predicted_mean": float(predicted_claim_count.mean()),
        },
    }
