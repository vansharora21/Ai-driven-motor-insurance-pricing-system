from __future__ import annotations

import json
import os
import random
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "modeling_config.json"

DEFAULT_MODELING_CONFIG: dict[str, Any] = {
    "random_seed": 42,
    "evaluation": {
        "test_size": 0.2,
    },
    "models": {
        "frequency": {
            "alpha": 1e-4,
            "max_iter": 1000,
        },
        "severity": {
            "alpha": 1e-4,
            "max_iter": 1000,
        },
    },
    "feature_engineering": {
        "exposure_lower_bound": 1e-6,
        "numeric_defaults": {
            "Exposure": 0.49,
            "VehPower": 6.0,
            "VehAge": 6.0,
            "DrivAge": 44.0,
            "BonusMalus": 50.0,
            "Density": 393.0,
        },
        "categorical_defaults": {
            "VehBrand": "B12",
            "VehGas": "Regular",
            "Area": "C",
            "Region": "Centre",
        },
        "numeric_clip_bounds": {
            "Exposure": {"min": 1e-6, "max": 2.5},
            "VehPower": {"min": 1.0, "max": 20.0},
            "VehAge": {"min": 0.0, "max": 100.0},
            "DrivAge": {"min": 18.0, "max": 100.0},
            "BonusMalus": {"min": 50.0, "max": 350.0},
            "Density": {"min": 0.0, "max": 27000.0},
        },
        "optional_simulated_features": [
            "simulated_vehicle_type",
            "simulated_daily_mileage",
            "simulated_night_driving_level",
            "simulated_harsh_braking_level",
            "simulated_accidents_last_2yr",
            "simulated_claim_history",
        ],
    },
    "pricing": {
        "expense_loading": 0.30,
        "fixed_expense": 50.0,
        "minimum_premium": 50.0,
        "risk_score_scale": 100.0,
        "risk_score_baseline_floor": 1.0,
        "annualized_expected_loss_thresholds": {
            "low_max": 150.0,
            "medium_max": 400.0,
        },
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@lru_cache(maxsize=4)
def load_modeling_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load modeling configuration from configs/."""
    path = Path(config_path)
    config = deepcopy(DEFAULT_MODELING_CONFIG)
    if path.exists():
        with path.open("r", encoding="utf-8") as config_file:
            loaded = json.load(config_file)
        config = _deep_merge(config, loaded)
    return deepcopy(config)


def get_random_seed(config: dict[str, Any] | None = None) -> int:
    source = config or load_modeling_config()
    return int(source["random_seed"])


def get_evaluation_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    source = config or load_modeling_config()
    return deepcopy(source["evaluation"])


def get_model_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    source = config or load_modeling_config()
    return deepcopy(source["models"])


def get_feature_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    source = config or load_modeling_config()
    return deepcopy(source["feature_engineering"])


def get_pricing_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    source = config or load_modeling_config()
    return deepcopy(source["pricing"])


def set_global_determinism(seed: int | None = None) -> int:
    resolved_seed = int(seed if seed is not None else get_random_seed())
    os.environ["PYTHONHASHSEED"] = str(resolved_seed)
    random.seed(resolved_seed)
    np.random.seed(resolved_seed)
    return resolved_seed
