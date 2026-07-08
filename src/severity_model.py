from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_gamma_deviance

from src.base_model import BaseActuarialModel
from src.config import get_model_config
from src.feature_engineering import MODEL_FEATURES, SEVERITY_TARGET
from src.model_factory import create_severity_regressor


class SeverityModel(BaseActuarialModel):
    target_column = SEVERITY_TARGET

    def _set_model_name(self, name: str) -> None:
        self.model_config["severity_model"] = name

    def _create_regressor(self) -> Any:
        return create_severity_regressor(
            model_config=self.model_config,
            alpha=self.alpha,
            max_iter=self.max_iter,
        )

    def fit(self, df: pd.DataFrame) -> Any:
        positive_df = df[df[SEVERITY_TARGET] > 0].copy()
        if positive_df.empty:
            raise ValueError(
                "Severity training data is empty after filtering positive claims."
            )
        self._pipeline = self.build_pipeline()
        self._pipeline.fit(
            positive_df[MODEL_FEATURES],
            positive_df[SEVERITY_TARGET],
        )
        return self._pipeline

    def evaluate(self, df: pd.DataFrame) -> dict[str, Any]:
        positive_df = df[df[SEVERITY_TARGET] > 0].copy()
        if positive_df.empty:
            raise ValueError(
                "Severity evaluation data is empty after filtering positive claims."
            )
        return super().evaluate(positive_df)

    def _extra_metrics(
        self, y_true: pd.Series, y_pred: pd.Series
    ) -> dict[str, float]:
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "gamma_deviance": float(mean_gamma_deviance(y_true, y_pred)),
        }

    def _prediction_name(self) -> str:
        return "predicted_claim_severity"


def train_severity_model(
    df: pd.DataFrame,
    alpha: float | None = None,
    max_iter: int | None = None,
    model_name: str | None = None,
    model_config: dict[str, Any] | None = None,
) -> Any:
    resolved_config = (
        model_config.copy() if model_config is not None else get_model_config()
    )
    model = SeverityModel(
        model_config=resolved_config,
        model_name=model_name,
        alpha=alpha,
        max_iter=max_iter,
    )
    return model.fit(df)


def predict_severity(model: Any, df: pd.DataFrame) -> pd.Series:
    base_model = model if isinstance(model, SeverityModel) else SeverityModel()
    base_model._pipeline = model if hasattr(model, "predict") else model
    return base_model.predict(df)


def evaluate_severity_model(
    model: Any, df: pd.DataFrame
) -> dict[str, Any]:
    base_model = model if isinstance(model, SeverityModel) else SeverityModel()
    if hasattr(model, "predict"):
        base_model._pipeline = model
    return base_model.evaluate(df)
