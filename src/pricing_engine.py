from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from src.config import get_feature_config, get_pricing_config
from src.feature_engineering import EXPOSURE_COLUMN

FEATURE_CONFIG = get_feature_config()
DEFAULT_PRICING_CONFIG = get_pricing_config()
EXPOSURE_LOWER_BOUND = float(FEATURE_CONFIG["exposure_lower_bound"])


def compute_risk_thresholds(
    pure_premium: pd.Series | None = None,
    pricing_config: Mapping[str, float | Mapping[str, float]] | None = None,
) -> dict[str, float]:
    """
    Return stable underwriting thresholds based on annualized expected loss.

    The thresholds are absolute business settings rather than portfolio quantiles,
    so the same policy profile is segmented consistently across datasets.
    """
    pricing = dict(DEFAULT_PRICING_CONFIG)
    if pricing_config:
        pricing.update(pricing_config)
    thresholds = pricing["annualized_expected_loss_thresholds"]
    return {
        "low_max": float(thresholds["low_max"]),
        "medium_max": float(thresholds["medium_max"]),
    }


def compute_portfolio_baselines(
    annual_frequency: pd.Series,
    expected_severity: pd.Series,
    pricing_config: Mapping[str, float] | None = None,
) -> dict[str, float]:
    pricing = dict(DEFAULT_PRICING_CONFIG)
    if pricing_config:
        pricing.update(pricing_config)

    baseline_floor = float(pricing["risk_score_baseline_floor"])
    annual_frequency_series = pd.Series(annual_frequency, copy=False).astype(float).clip(lower=0.0)
    expected_severity_series = pd.Series(expected_severity, copy=False).astype(float).clip(lower=0.0)
    annualized_expected_loss = annual_frequency_series * expected_severity_series

    return {
        "annual_frequency": max(float(annual_frequency_series.mean()), baseline_floor),
        "claim_severity": max(float(expected_severity_series.mean()), baseline_floor),
        "annualized_expected_loss": max(float(annualized_expected_loss.mean()), baseline_floor),
    }


def assign_risk_category(
    annualized_expected_loss: pd.Series,
    thresholds: Mapping[str, float] | None = None,
) -> pd.Series:
    """Map annualized expected loss into Low / Medium / High underwriting bands."""
    resolved_thresholds = dict(thresholds or compute_risk_thresholds())
    low_max = float(resolved_thresholds["low_max"])
    medium_max = float(resolved_thresholds["medium_max"])

    categories = np.where(
        annualized_expected_loss <= low_max,
        "Low",
        np.where(annualized_expected_loss <= medium_max, "Medium", "High"),
    )
    return pd.Series(categories, index=annualized_expected_loss.index, name="risk_category")


def calculate_premium(
    df: pd.DataFrame,
    annual_frequency: pd.Series,
    expected_severity: pd.Series,
    pricing_config: Mapping[str, float] | None = None,
    risk_thresholds: Mapping[str, float] | None = None,
    portfolio_baselines: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Convert model outputs into actuarially consistent premium components."""
    pricing = dict(DEFAULT_PRICING_CONFIG)
    if pricing_config:
        pricing.update(pricing_config)

    scored = df.copy()
    exposure = pd.to_numeric(scored[EXPOSURE_COLUMN], errors="coerce").fillna(1.0).clip(lower=EXPOSURE_LOWER_BOUND)
    annual_frequency_series = pd.Series(annual_frequency, index=scored.index, copy=False).astype(float).clip(lower=0.0)
    expected_severity_series = pd.Series(expected_severity, index=scored.index, copy=False).astype(float).clip(lower=0.0)

    resolved_baselines = (
        {key: float(value) for key, value in portfolio_baselines.items()}
        if portfolio_baselines
        else compute_portfolio_baselines(annual_frequency_series, expected_severity_series, pricing)
    )
    baseline_floor = float(pricing["risk_score_baseline_floor"])
    annualized_loss_baseline = max(float(resolved_baselines["annualized_expected_loss"]), baseline_floor)

    scored["predicted_annual_frequency"] = annual_frequency_series
    scored["predicted_claim_count"] = scored["predicted_annual_frequency"] * exposure
    scored["predicted_claim_severity"] = expected_severity_series

    # Annualized loss is exposure-neutral, so it is the right basis for stable risk bands.
    scored["annualized_expected_loss"] = scored["predicted_annual_frequency"] * scored["predicted_claim_severity"]

    # Expected loss for the policy term is frequency x severity x exposure.
    scored["expected_loss"] = scored["predicted_claim_count"] * scored["predicted_claim_severity"]
    scored["pure_premium"] = scored["expected_loss"]
    scored["technical_premium"] = scored["pure_premium"] * (1.0 + float(pricing["expense_loading"])) + float(pricing["fixed_expense"])
    scored["loaded_premium"] = scored["technical_premium"]
    scored["final_premium"] = scored["technical_premium"].clip(lower=float(pricing["minimum_premium"]))

    scored["frequency_relativity"] = scored["predicted_annual_frequency"] / max(float(resolved_baselines["annual_frequency"]), baseline_floor)
    scored["severity_relativity"] = scored["predicted_claim_severity"] / max(float(resolved_baselines["claim_severity"]), baseline_floor)
    scored["risk_score"] = float(pricing["risk_score_scale"]) * scored["annualized_expected_loss"] / annualized_loss_baseline
    scored["risk_category"] = assign_risk_category(
        scored["annualized_expected_loss"],
        thresholds=risk_thresholds or compute_risk_thresholds(pricing_config=pricing),
    )
    return scored
