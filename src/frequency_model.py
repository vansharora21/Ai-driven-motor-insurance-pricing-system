from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import mean_poisson_deviance

from src.base_model import BaseActuarialModel
from src.config import get_feature_config, get_model_config
from src.feature_engineering import EXPOSURE_COLUMN, FREQUENCY_TARGET, MODEL_FEATURES
from src.model_factory import create_frequency_regressor

FEATURE_CONFIG = get_feature_config()
EXPOSURE_LOWER_BOUND = float(FEATURE_CONFIG["exposure_lower_bound"])


class FrequencyModel(BaseActuarialModel):
    target_column = FREQUENCY_TARGET

    def _set_model_name(self, name: str) -> None:
        self.model_config["frequency_model"] = name

    def _create_regressor(self) -> Any:
        return create_frequency_regressor(
            model_config=self.model_config,
            alpha=self.alpha,
            max_iter=self.max_iter,
        )

    def _get_sample_weight(self, df: pd.DataFrame) -> pd.Series | None:
        return df[EXPOSURE_COLUMN].clip(lower=EXPOSURE_LOWER_BOUND)

    def fit(self, df: pd.DataFrame) -> Any:
        exposure = df[EXPOSURE_COLUMN].clip(lower=EXPOSURE_LOWER_BOUND)
        target_frequency = df[FREQUENCY_TARGET] / exposure
        self._pipeline = self.build_pipeline()
        self._pipeline.fit(
            df[MODEL_FEATURES],
            target_frequency,
            regressor__sample_weight=exposure,
        )
        return self._pipeline

    def _prediction_name(self) -> str:
        return "predicted_annual_frequency"

    def _extra_metrics(
        self, y_true: pd.Series, y_pred: pd.Series
    ) -> dict[str, float]:
        return {
            "poisson_deviance": float(
                mean_poisson_deviance(y_true, y_pred)
            )
        }


def train_frequency_model(
    df: pd.DataFrame,
    alpha: float | None = None,
    max_iter: int | None = None,
    model_name: str | None = None,
    model_config: dict[str, Any] | None = None,
) -> Any:
    resolved_config = (
        model_config.copy() if model_config is not None else get_model_config()
    )
    model = FrequencyModel(
        model_config=resolved_config,
        model_name=model_name,
        alpha=alpha,
        max_iter=max_iter,
    )
    return model.fit(df)


def predict_frequency(model: Any, df: pd.DataFrame) -> pd.Series:
    base_model = model if isinstance(model, FrequencyModel) else FrequencyModel()
    base_model._pipeline = model if hasattr(model, "predict") else model
    return base_model.predict(df)


def predict_claim_count(model: Any, df: pd.DataFrame) -> pd.Series:
    from src.feature_engineering import EXPOSURE_COLUMN

    annual_frequency = predict_frequency(model, df)
    return annual_frequency * df[EXPOSURE_COLUMN]


def evaluate_frequency_model(
    model: Any, df: pd.DataFrame
) -> dict[str, Any]:
    from src.base_model import BaseActuarialModel

    base_model = model if isinstance(model, BaseActuarialModel) else FrequencyModel()
    if hasattr(model, "predict"):
        base_model._pipeline = model
    return base_model.evaluate(df)
