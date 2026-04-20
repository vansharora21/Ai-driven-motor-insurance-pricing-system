from __future__ import annotations

import pandas as pd

from src.feature_engineering import engineer_features


def simulate_stress_scenario(
    df: pd.DataFrame,
    scenario_name: str = "Adverse Urban Inflation",
    exposure_multiplier: float = 1.0,
    bonus_malus_shift: float = 10.0,
    density_multiplier: float = 1.15,
) -> pd.DataFrame:
    """
    Optional scenario analysis helper for the real freMTPL2 feature set.

    The stress test increases bonus-malus and density to approximate a harder
    underwriting environment while preserving the rest of the portfolio mix.
    """
    stressed = df.copy()
    stressed["Exposure"] = stressed["Exposure"] * exposure_multiplier
    stressed["BonusMalus"] = stressed["BonusMalus"] + bonus_malus_shift
    stressed["Density"] = stressed["Density"] * density_multiplier
    stressed["scenario_name"] = scenario_name
    return engineer_features(stressed)


if __name__ == "__main__":
    from predict import predict_premiums
    from src.data_loader import prepare_model_datasets

    policy_df, _, _ = prepare_model_datasets()
    base_portfolio = predict_premiums(policy_df.head(5000))
    stressed_portfolio = predict_premiums(simulate_stress_scenario(policy_df.head(5000)))
    print(base_portfolio["final_premium"].mean(), stressed_portfolio["final_premium"].mean())
