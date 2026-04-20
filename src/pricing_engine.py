from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from src.feature_engineering import EXPOSURE_COLUMN

DEFAULT_PRICING_CONFIG = {
    "expense_loading": 0.30,
    "fixed_expense": 50.0,
    "minimum_premium": 50.0,
}


def compute_risk_thresholds(pure_premium: pd.Series) -> dict[str, float]:
    """Derive simple portfolio-relative risk thresholds from the pure premium distribution."""
    return {
        "low_to_medium": float(pure_premium.quantile(0.33)),
        "medium_to_high": float(pure_premium.quantile(0.67)),
    }


def assign_risk_category(pure_premium: pd.Series, thresholds: Mapping[str, float] | None = None) -> pd.Series:
    """Map premium levels into Low / Medium / High risk segments."""
    if not thresholds:
        return pd.Series(["Unassigned"] * len(pure_premium), index=pure_premium.index, name="risk_category")

    low_to_medium = thresholds["low_to_medium"]
    medium_to_high = thresholds["medium_to_high"]

    categories = np.where(
        pure_premium <= low_to_medium,
        "Low",
        np.where(pure_premium <= medium_to_high, "Medium", "High"),
    )
    return pd.Series(categories, index=pure_premium.index, name="risk_category")


def calculate_premium(
    df: pd.DataFrame,
    annual_frequency: pd.Series,
    expected_severity: pd.Series,
    pricing_config: Mapping[str, float] | None = None,
    risk_thresholds: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Convert model outputs into pure premium and loaded premium components."""
    pricing = dict(DEFAULT_PRICING_CONFIG)
    if pricing_config:
        pricing.update(pricing_config)

    scored = df.copy()
    scored["predicted_annual_frequency"] = annual_frequency
    scored["predicted_claim_count"] = scored["predicted_annual_frequency"] * scored[EXPOSURE_COLUMN]
    scored["predicted_claim_severity"] = expected_severity
    scored["pure_premium"] = scored["predicted_claim_count"] * scored["predicted_claim_severity"]
    scored["loaded_premium"] = scored["pure_premium"] * (1.0 + pricing["expense_loading"]) + pricing["fixed_expense"]
    scored["final_premium"] = scored["loaded_premium"].clip(lower=pricing["minimum_premium"])
    scored["risk_category"] = assign_risk_category(scored["pure_premium"], risk_thresholds)
    return scored
