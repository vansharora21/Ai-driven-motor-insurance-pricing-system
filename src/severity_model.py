from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import GammaRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from src.config import get_model_config
from src.feature_engineering import MODEL_FEATURES, SEVERITY_TARGET, build_preprocessor

MODEL_CONFIG = get_model_config()


def train_severity_model(
    df: pd.DataFrame,
    alpha: float | None = None,
    max_iter: int | None = None,
) -> Pipeline:
    """
    Train a Gamma regression model on positive claim amounts.

    Gamma regression requires strictly positive targets, so non-positive claim
    amounts are excluded before training.
    """
    positive_claim_df = df[df[SEVERITY_TARGET] > 0].copy()
    if positive_claim_df.empty:
        raise ValueError("Severity training data is empty after filtering positive claims.")

    severity_config = MODEL_CONFIG["severity"]
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                GammaRegressor(
                    alpha=float(alpha if alpha is not None else severity_config["alpha"]),
                    max_iter=int(max_iter if max_iter is not None else severity_config["max_iter"]),
                ),
            ),
        ]
    )
    model.fit(positive_claim_df[MODEL_FEATURES], positive_claim_df[SEVERITY_TARGET])
    return model


def predict_severity(model: Pipeline, df: pd.DataFrame) -> pd.Series:
    """Predict expected claim severity for each row in the input dataframe."""
    predicted_severity = model.predict(df[MODEL_FEATURES])
    predicted_severity = np.clip(predicted_severity, 1e-9, None)
    return pd.Series(predicted_severity, index=df.index, name="predicted_claim_severity")


def evaluate_severity_model(model: Pipeline, df: pd.DataFrame) -> dict[str, float]:
    """Evaluate holdout severity performance using MAE and RMSE."""
    positive_eval_df = df[df[SEVERITY_TARGET] > 0].copy()
    if positive_eval_df.empty:
        raise ValueError("Severity evaluation data is empty after filtering positive claims.")

    observed_severity = positive_eval_df[SEVERITY_TARGET].astype(float)
    predicted_severity = predict_severity(model, positive_eval_df)

    mae = float(mean_absolute_error(observed_severity, predicted_severity))
    rmse = float(np.sqrt(mean_squared_error(observed_severity, predicted_severity)))

    return {
        "mae": mae,
        "rmse": rmse,
        "observed_average_severity": float(observed_severity.mean()),
        "predicted_average_severity": float(predicted_severity.mean()),
    }
