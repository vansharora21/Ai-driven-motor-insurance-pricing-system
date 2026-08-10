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
    # The pure_premium parameter is intentionally kept for backward API compatibility.
    _ = pure_premium
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
    """Compute stable portfolio-level baselines used for relativity and risk scoring."""
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
    """
    Convert model outputs into premium components used for underwriting.

    Returned columns include predicted frequency/severity, expected losses,
    premium components, risk score, and final risk category.
    """
    pricing = dict(DEFAULT_PRICING_CONFIG)
    if pricing_config:
        pricing.update(pricing_config)

    # Models are trained on EUR data; convert monetary outputs to INR for the
    # Indian market. Frequency is unitless (claims/year) and stays unchanged.
    fx_rate = float(pricing.get("fx_rate_to_inr", 1.0))

    scored_df = df.copy()
    exposure_values = pd.to_numeric(scored_df[EXPOSURE_COLUMN], errors="coerce").fillna(1.0).clip(lower=EXPOSURE_LOWER_BOUND)
    annual_frequency_series = pd.Series(annual_frequency, index=scored_df.index, copy=False).astype(float).clip(lower=0.0)
    expected_severity_series = pd.Series(expected_severity, index=scored_df.index, copy=False).astype(float).clip(lower=0.0) * fx_rate

    resolved_baselines = (
        {
            key: float(value) * (fx_rate if key in ("claim_severity", "annualized_expected_loss") else 1.0)
            for key, value in portfolio_baselines.items()
        }
        if portfolio_baselines
        else compute_portfolio_baselines(annual_frequency_series, expected_severity_series, pricing)
    )
    baseline_floor = float(pricing["risk_score_baseline_floor"])
    annualized_loss_baseline = max(float(resolved_baselines["annualized_expected_loss"]), baseline_floor)

    scored_df["predicted_annual_frequency"] = annual_frequency_series
    scored_df["predicted_claim_count"] = scored_df["predicted_annual_frequency"] * exposure_values
    scored_df["predicted_claim_severity"] = expected_severity_series

    # Annualized loss is exposure-neutral, so it is the right basis for stable risk bands.
    scored_df["annualized_expected_loss"] = scored_df["predicted_annual_frequency"] * scored_df["predicted_claim_severity"]

    # Expected loss for the policy term is frequency x severity x exposure.
    scored_df["expected_loss"] = scored_df["predicted_claim_count"] * scored_df["predicted_claim_severity"]
    scored_df["pure_premium"] = scored_df["expected_loss"]
    scored_df["technical_premium"] = scored_df["pure_premium"] * (1.0 + float(pricing["expense_loading"])) + float(pricing["fixed_expense"])
    scored_df["loaded_premium"] = scored_df["technical_premium"]
    scored_df["final_premium"] = scored_df["technical_premium"].clip(lower=float(pricing["minimum_premium"]))

    scored_df["frequency_relativity"] = scored_df["predicted_annual_frequency"] / max(float(resolved_baselines["annual_frequency"]), baseline_floor)
    scored_df["severity_relativity"] = scored_df["predicted_claim_severity"] / max(float(resolved_baselines["claim_severity"]), baseline_floor)
    scored_df["risk_score"] = float(pricing["risk_score_scale"]) * scored_df["annualized_expected_loss"] / annualized_loss_baseline
    scored_df["risk_category"] = assign_risk_category(
        scored_df["annualized_expected_loss"],
        thresholds=risk_thresholds or compute_risk_thresholds(pricing_config=pricing),
    )
    return scored_df
