from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance, mean_squared_error
from sklearn.pipeline import Pipeline

from src.config import get_feature_config, get_model_config
from src.feature_engineering import EXPOSURE_COLUMN, FREQUENCY_TARGET, MODEL_FEATURES, build_preprocessor

MODEL_CONFIG = get_model_config()
FEATURE_CONFIG = get_feature_config()
EXPOSURE_LOWER_BOUND = float(FEATURE_CONFIG["exposure_lower_bound"])


def train_frequency_model(
    df: pd.DataFrame,
    alpha: float | None = None,
    max_iter: int | None = None,
) -> Pipeline:
    """Train a Poisson frequency model on claims per unit exposure."""
    frequency_config = MODEL_CONFIG["frequency"]
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                PoissonRegressor(
                    alpha=float(alpha if alpha is not None else frequency_config["alpha"]),
                    max_iter=int(max_iter if max_iter is not None else frequency_config["max_iter"]),
                ),
            ),
        ]
    )

    exposure = df[EXPOSURE_COLUMN].clip(lower=EXPOSURE_LOWER_BOUND)
    target_frequency = df[FREQUENCY_TARGET] / exposure
    model.fit(df[MODEL_FEATURES], target_frequency, regressor__sample_weight=exposure)
    return model


def predict_frequency(model: Pipeline, df: pd.DataFrame) -> pd.Series:
    """Predict annual claim frequency for each policy."""
    predictions = model.predict(df[MODEL_FEATURES])
    predictions = np.clip(predictions, 1e-9, None)
    return pd.Series(predictions, index=df.index, name="predicted_annual_frequency")


def predict_claim_count(model: Pipeline, df: pd.DataFrame) -> pd.Series:
    """Convert annual frequency into expected claim count for the policy exposure."""
    annual_frequency = predict_frequency(model, df)
    return annual_frequency * df[EXPOSURE_COLUMN]


def evaluate_frequency_model(model: Pipeline, df: pd.DataFrame) -> dict[str, float]:
    """Evaluate predicted claim counts on a holdout dataset."""
    actual_claim_count = df[FREQUENCY_TARGET].astype(float)
    predicted_claim_count = predict_claim_count(model, df).clip(lower=1e-9)

    rmse = float(np.sqrt(mean_squared_error(actual_claim_count, predicted_claim_count)))
    poisson_deviance = float(mean_poisson_deviance(actual_claim_count, predicted_claim_count))

    return {
        "rmse": rmse,
        "mean_poisson_deviance": poisson_deviance,
        "observed_average_claim_count": float(actual_claim_count.mean()),
        "predicted_average_claim_count": float(predicted_claim_count.mean()),
    }
