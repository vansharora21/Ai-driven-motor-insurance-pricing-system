from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from src.config import get_model_config
from src.feature_engineering import MODEL_FEATURES, SEVERITY_TARGET, build_preprocessor
from src.model_factory import create_severity_regressor


def train_severity_model(
    df: pd.DataFrame,
    alpha: float | None = None,
    max_iter: int | None = None,
    model_name: str | None = None,
    model_config: dict[str, Any] | None = None,
) -> Pipeline:
    """
    Train a severity model on positive claim amounts.

    Severity models require strictly positive targets, so non-positive claim
    amounts are excluded before training.
    """
    positive_claim_df = df[df[SEVERITY_TARGET] > 0].copy()
    if positive_claim_df.empty:
        raise ValueError("Severity training data is empty after filtering positive claims.")

    resolved_model_config = model_config.copy() if model_config is not None else get_model_config()
    if model_name is not None:
        resolved_model_config["severity_model"] = model_name

    regressor = create_severity_regressor(
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
