from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineering import engineer_features
from src.pricing_engine import assign_risk_category, calculate_premium


def test_assign_risk_category_boundaries() -> None:
    annualized_loss = pd.Series([120.0, 250.0, 700.0])
    categories = assign_risk_category(annualized_loss, thresholds={"low_max": 150.0, "medium_max": 400.0})
    assert categories.tolist() == ["Low", "Medium", "High"]


def test_calculate_premium_outputs_required_columns(sample_policy_df: pd.DataFrame) -> None:
    policies = engineer_features(sample_policy_df)
    annual_frequency = pd.Series(np.full(len(policies), 0.2), index=policies.index)
    expected_severity = pd.Series(np.full(len(policies), 1000.0), index=policies.index)

    scored = calculate_premium(policies, annual_frequency, expected_severity)
    required_columns = [
        "predicted_annual_frequency",
        "predicted_claim_count",
        "predicted_claim_severity",
        "annualized_expected_loss",
        "final_premium",
        "risk_category",
    ]
    for column in required_columns:
        assert column in scored.columns

    assert (scored["final_premium"] >= 50.0).all()
