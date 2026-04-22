from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import GammaRegressor, PoissonRegressor

from src.config import get_model_config, get_random_seed

FREQUENCY_MODEL_OPTIONS = {"poisson", "random_forest", "xgboost"}
SEVERITY_MODEL_OPTIONS = {"gamma", "random_forest", "xgboost"}


def _resolve_model_name(model_name: Any, valid_options: set[str], label: str) -> str:
    resolved = str(model_name).strip().lower()
    if resolved not in valid_options:
        valid = ", ".join(sorted(valid_options))
        raise ValueError(f"Unsupported {label} '{model_name}'. Allowed values: {valid}.")
    return resolved


def _load_xgb_regressor():
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "XGBoost model selected but package 'xgboost' is not installed. "
            "Install it with: pip install xgboost"
        ) from exc
    return XGBRegressor


def create_frequency_regressor(
    model_config: dict[str, Any] | None = None,
    alpha: float | None = None,
    max_iter: int | None = None,
) -> Any:
    config = model_config or get_model_config()
    frequency_model_name = _resolve_model_name(
        config.get("frequency_model", "poisson"),
        FREQUENCY_MODEL_OPTIONS,
        "frequency_model",
    )
    frequency_config = config.get("frequency", {})
    random_seed = get_random_seed()

    if frequency_model_name == "poisson":
        return PoissonRegressor(
            alpha=float(alpha if alpha is not None else frequency_config.get("alpha", 1e-4)),
            max_iter=int(max_iter if max_iter is not None else frequency_config.get("max_iter", 1000)),
        )

    if frequency_model_name == "random_forest":
        rf_config = frequency_config.get("random_forest", {})
        return RandomForestRegressor(
            n_estimators=int(rf_config.get("n_estimators", 300)),
            max_depth=rf_config.get("max_depth", None),
            min_samples_leaf=int(rf_config.get("min_samples_leaf", 1)),
            n_jobs=int(rf_config.get("n_jobs", -1)),
            random_state=int(rf_config.get("random_state", random_seed)),
        )

    xgb_config = frequency_config.get("xgboost", {})
    XGBRegressor = _load_xgb_regressor()
    return XGBRegressor(
        objective=xgb_config.get("objective", "count:poisson"),
        n_estimators=int(xgb_config.get("n_estimators", 300)),
        learning_rate=float(xgb_config.get("learning_rate", 0.05)),
        max_depth=int(xgb_config.get("max_depth", 6)),
        subsample=float(xgb_config.get("subsample", 0.8)),
        colsample_bytree=float(xgb_config.get("colsample_bytree", 0.8)),
        reg_lambda=float(xgb_config.get("reg_lambda", 1.0)),
        random_state=int(xgb_config.get("random_state", random_seed)),
        n_jobs=int(xgb_config.get("n_jobs", -1)),
    )


def create_severity_regressor(
    model_config: dict[str, Any] | None = None,
    alpha: float | None = None,
    max_iter: int | None = None,
) -> Any:
    config = model_config or get_model_config()
    severity_model_name = _resolve_model_name(
        config.get("severity_model", "gamma"),
        SEVERITY_MODEL_OPTIONS,
        "severity_model",
    )
    severity_config = config.get("severity", {})
    random_seed = get_random_seed()

    if severity_model_name == "gamma":
        return GammaRegressor(
            alpha=float(alpha if alpha is not None else severity_config.get("alpha", 1e-4)),
            max_iter=int(max_iter if max_iter is not None else severity_config.get("max_iter", 1000)),
        )

    if severity_model_name == "random_forest":
        rf_config = severity_config.get("random_forest", {})
        return RandomForestRegressor(
            n_estimators=int(rf_config.get("n_estimators", 300)),
            max_depth=rf_config.get("max_depth", None),
            min_samples_leaf=int(rf_config.get("min_samples_leaf", 1)),
            n_jobs=int(rf_config.get("n_jobs", -1)),
            random_state=int(rf_config.get("random_state", random_seed)),
        )

    xgb_config = severity_config.get("xgboost", {})
    XGBRegressor = _load_xgb_regressor()
    return XGBRegressor(
        objective=xgb_config.get("objective", "reg:gamma"),
        n_estimators=int(xgb_config.get("n_estimators", 400)),
        learning_rate=float(xgb_config.get("learning_rate", 0.05)),
        max_depth=int(xgb_config.get("max_depth", 6)),
        subsample=float(xgb_config.get("subsample", 0.8)),
        colsample_bytree=float(xgb_config.get("colsample_bytree", 0.8)),
        reg_lambda=float(xgb_config.get("reg_lambda", 1.0)),
        random_state=int(xgb_config.get("random_state", random_seed)),
        n_jobs=int(xgb_config.get("n_jobs", -1)),
    )
