from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.feature_engineering import MODEL_FEATURES, build_preprocessor
from src.logger import get_logger

logger = get_logger(__name__)


class BaseActuarialModel(ABC):
    """Shared base for frequency and severity models to reduce duplication."""

    def __init__(
        self,
        model_config: dict[str, Any] | None = None,
        model_name: str | None = None,
        alpha: float | None = None,
        max_iter: int | None = None,
    ) -> None:
        self.model_config = (model_config or {}).copy()
        if model_name is not None:
            self._set_model_name(model_name)
        self.alpha = alpha
        self.max_iter = max_iter
        self._pipeline: Pipeline | None = None

    @abstractmethod
    def _set_model_name(self, name: str) -> None:
        ...

    @abstractmethod
    def _create_regressor(self) -> Any:
        ...

    @property
    @abstractmethod
    def target_column(self) -> str:
        ...

    def build_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("regressor", self._create_regressor()),
            ]
        )

    def fit(self, df: pd.DataFrame) -> Pipeline:
        self._pipeline = self.build_pipeline()
        X = df[MODEL_FEATURES]
        y = df[self.target_column]
        sample_weight = self._get_sample_weight(df)
        if sample_weight is not None:
            self._pipeline.fit(X, y, regressor__sample_weight=sample_weight)
        else:
            self._pipeline.fit(X, y)
        logger.info(
            "Trained %s on %d rows", self.__class__.__name__, len(df)
        )
        return self._pipeline

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self._pipeline is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        raw = self._pipeline.predict(df[MODEL_FEATURES])
        clipped = np.clip(raw, 1e-9, None)
        return pd.Series(clipped, index=df.index, name=self._prediction_name())

    def evaluate(self, df: pd.DataFrame) -> dict[str, Any]:
        from sklearn.metrics import mean_squared_error

        y_true = df[self.target_column].astype(float)
        y_pred = self.predict(df)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        regressor_name = (
            self._pipeline.named_steps["regressor"].__class__.__name__
            if self._pipeline
            else "unknown"
        )
        result: dict[str, Any] = {
            "model_name": regressor_name,
            "evaluation_split": "test",
            "sample_size": int(len(df)),
            "metrics": {"rmse": rmse},
            "distribution": {
                "observed_mean": float(y_true.mean()),
                "predicted_mean": float(y_pred.mean()),
            },
        }
        result["metrics"].update(self._extra_metrics(y_true, y_pred))
        return result

    def _get_sample_weight(self, df: pd.DataFrame) -> pd.Series | None:
        return None

    def _prediction_name(self) -> str:
        return f"predicted_{self.__class__.__name__.lower().replace('model', '')}"

    def _extra_metrics(
        self, y_true: pd.Series, y_pred: pd.Series
    ) -> dict[str, float]:
        return {}
