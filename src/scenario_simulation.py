from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import get_feature_config, set_global_determinism
from src.data_loader import build_inference_frame
from src.feature_engineering import POLICY_ID_COLUMN
from src.frequency_model import predict_frequency
from src.model_artifacts import PLOTS_DIR, REPORTS_DIR, load_artifacts
from src.pricing_engine import calculate_premium
from src.severity_model import predict_severity

SCENARIO_REPORT_PATH = REPORTS_DIR / "scenario_analysis.csv"
SCENARIO_PLOT_PATH = PLOTS_DIR / "scenario_premium_curves.png"


def _to_single_row_dataframe(input_policy: pd.DataFrame | pd.Series | Mapping[str, Any]) -> pd.DataFrame:
    if isinstance(input_policy, pd.DataFrame):
        if input_policy.empty:
            raise ValueError("input_policy dataframe is empty.")
        if len(input_policy) != 1:
            raise ValueError("simulate_scenarios expects exactly one policy row.")
        return input_policy.copy().reset_index(drop=True)

    if isinstance(input_policy, pd.Series):
        return input_policy.to_frame().T.reset_index(drop=True)

    if isinstance(input_policy, Mapping):
        return pd.DataFrame([dict(input_policy)])

    raise TypeError("input_policy must be a pandas DataFrame, Series, or mapping.")


def _clip_feature_value(feature_name: str, raw_value: float) -> float:
    feature_config = get_feature_config()
    bounds = feature_config.get("numeric_clip_bounds", {}).get(feature_name, {})
    lower = bounds.get("min")
    upper = bounds.get("max")

    value = float(raw_value)
    if lower is not None:
        value = max(value, float(lower))
    if upper is not None:
        value = min(value, float(upper))
    return value


def _build_variation_rows(base_policy: pd.DataFrame) -> pd.DataFrame:
    baseline = base_policy.copy()
    baseline["scenario_name"] = "baseline"
    baseline["changed_feature"] = "none"
    baseline["changed_value"] = np.nan

    scenario_specs = [
        ("vehicle_power_decrease", "VehPower", lambda value: value * 0.8),
        ("vehicle_power_increase", "VehPower", lambda value: value * 1.2),
        ("driver_age_decrease", "DrivAge", lambda value: value - 10.0),
        ("driver_age_increase", "DrivAge", lambda value: value + 10.0),
        ("bonus_malus_decrease", "BonusMalus", lambda value: value - 20.0),
        ("bonus_malus_increase", "BonusMalus", lambda value: value + 20.0),
    ]

    variations: list[pd.DataFrame] = [baseline]
    for scenario_name, feature_name, transform in scenario_specs:
        scenario_row = base_policy.copy()
        current_value = float(pd.to_numeric(scenario_row.at[0, feature_name], errors="coerce"))
        changed_value = _clip_feature_value(feature_name, transform(current_value))

        scenario_row[feature_name] = changed_value
        scenario_row["scenario_name"] = scenario_name
        scenario_row["changed_feature"] = feature_name
        scenario_row["changed_value"] = changed_value
        variations.append(scenario_row)

    return pd.concat(variations, ignore_index=True)


def _score_scenarios(scenario_rows: pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    metadata = artifacts.get("metadata", {})
    set_global_determinism(int(metadata.get("random_seed", 42)))

    prepared = build_inference_frame(scenario_rows, metadata)
    annual_frequency = predict_frequency(artifacts["frequency_model"], prepared)
    expected_severity = predict_severity(artifacts["severity_model"], prepared)

    scored = calculate_premium(
        prepared,
        annual_frequency,
        expected_severity,
        pricing_config=metadata.get("pricing_config"),
        risk_thresholds=metadata.get("risk_thresholds"),
        portfolio_baselines=metadata.get("portfolio_baselines"),
    )

    scored["scenario_name"] = scenario_rows["scenario_name"].values
    scored["changed_feature"] = scenario_rows["changed_feature"].values
    scored["changed_value"] = scenario_rows["changed_value"].values
    return scored


def _save_scenario_premium_curves(scenario_df: pd.DataFrame, save_path: Path) -> Path:
    sns.set_theme(style="whitegrid", context="notebook")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    feature_order = ["VehPower", "DrivAge", "BonusMalus"]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)

    for axis, feature_name in zip(axes, feature_order):
        feature_frame = scenario_df[
            (scenario_df["changed_feature"] == feature_name) | (scenario_df["changed_feature"] == "none")
        ].copy()
        if feature_name == "VehPower":
            feature_frame["feature_value"] = feature_frame["VehPower"]
        elif feature_name == "DrivAge":
            feature_frame["feature_value"] = feature_frame["DrivAge"]
        else:
            feature_frame["feature_value"] = feature_frame["BonusMalus"]

        feature_frame = feature_frame.sort_values(by="feature_value")
        sns.lineplot(data=feature_frame, x="feature_value", y="final_premium", marker="o", ax=axis, color="#2563eb")
        axis.set_title(f"{feature_name} vs Final Premium")
        axis.set_xlabel(feature_name)
        axis.set_ylabel("Final Premium")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(figure)
    return save_path


def simulate_scenarios(
    input_policy: pd.DataFrame | pd.Series | Mapping[str, Any],
    artifacts: dict[str, Any] | None = None,
    output_path: str | Path = SCENARIO_REPORT_PATH,
    save_plot: bool = True,
    plot_path: str | Path = SCENARIO_PLOT_PATH,
) -> pd.DataFrame:
    """Simulate underwriting scenarios and save premium impacts for one policy row."""
    loaded_artifacts = artifacts or load_artifacts()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    base_policy = _to_single_row_dataframe(input_policy)
    scenario_rows = _build_variation_rows(base_policy)
    scored = _score_scenarios(scenario_rows, loaded_artifacts)

    baseline_premium = float(scored.loc[scored["scenario_name"] == "baseline", "final_premium"].iloc[0])
    scored["premium_delta"] = scored["final_premium"] - baseline_premium
    scored["premium_delta_pct"] = (scored["premium_delta"] / max(baseline_premium, 1e-9)) * 100.0

    if POLICY_ID_COLUMN in scored.columns:
        policy_ids = pd.to_numeric(scored[POLICY_ID_COLUMN], errors="coerce").fillna(0).round().astype(int)
    else:
        policy_ids = pd.Series([1] * len(scored), index=scored.index, dtype=int)

    report = pd.DataFrame(
        {
            "policy_id": policy_ids,
            "scenario_name": scored["scenario_name"],
            "changed_feature": scored["changed_feature"],
            "changed_value": scored["changed_value"],
            "predicted_frequency": scored["predicted_annual_frequency"].astype(float),
            "predicted_severity": scored["predicted_claim_severity"].astype(float),
            "final_premium": scored["final_premium"].astype(float),
            "premium_delta": scored["premium_delta"].astype(float),
            "premium_delta_pct": scored["premium_delta_pct"].astype(float),
        }
    )

    report.to_csv(output_file, index=False, float_format="%.6f")

    if save_plot:
        _save_scenario_premium_curves(scored, Path(plot_path))

    return report
