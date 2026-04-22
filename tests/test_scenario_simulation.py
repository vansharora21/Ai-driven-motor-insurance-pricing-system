from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.feature_engineering import engineer_features
from src.frequency_model import train_frequency_model
from src.pricing_engine import compute_portfolio_baselines, compute_risk_thresholds
from src.scenario_simulation import simulate_scenarios
from src.severity_model import train_severity_model


def test_simulate_scenarios_generates_report_and_plot(
    sample_policy_df: pd.DataFrame,
    sample_claim_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    frequency_df = engineer_features(sample_policy_df)
    severity_df = engineer_features(sample_claim_df)

    frequency_model = train_frequency_model(
        frequency_df,
        model_name="poisson",
        model_config={"frequency_model": "poisson", "frequency": {"alpha": 1e-4, "max_iter": 200}},
    )
    severity_model = train_severity_model(
        severity_df,
        model_name="gamma",
        model_config={"severity_model": "gamma", "severity": {"alpha": 1e-4, "max_iter": 200}},
    )

    artifacts = {
        "frequency_model": frequency_model,
        "severity_model": severity_model,
        "metadata": {
            "random_seed": 42,
            "pricing_config": {},
            "risk_thresholds": compute_risk_thresholds(),
            "portfolio_baselines": compute_portfolio_baselines(
                annual_frequency=pd.Series([0.1, 0.15, 0.2]),
                expected_severity=pd.Series([1200.0, 1800.0, 2400.0]),
            ),
        },
    }

    input_policy = sample_policy_df.head(1).copy()
    output_path = tmp_path / "scenario_analysis.csv"
    plot_path = tmp_path / "scenario_premium_curves.png"

    report = simulate_scenarios(
        input_policy,
        artifacts=artifacts,
        output_path=output_path,
        save_plot=True,
        plot_path=plot_path,
    )

    assert output_path.exists()
    assert plot_path.exists()
    assert len(report) == 7

    expected_scenarios = {
        "baseline",
        "vehicle_power_decrease",
        "vehicle_power_increase",
        "driver_age_decrease",
        "driver_age_increase",
        "bonus_malus_decrease",
        "bonus_malus_increase",
    }
    assert expected_scenarios == set(report["scenario_name"])

    required_columns = {
        "policy_id",
        "scenario_name",
        "changed_feature",
        "changed_value",
        "predicted_frequency",
        "predicted_severity",
        "final_premium",
        "premium_delta",
        "premium_delta_pct",
    }
    assert required_columns.issubset(report.columns)
