from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from predict import predict_premiums
from src.data_loader import DataValidationError
from src.model_artifacts import load_artifacts

st.set_page_config(page_title="Motor Insurance Pricing", layout="wide")


@st.cache_resource
def load_prediction_artifacts():
    """Load trained artifacts once per Streamlit session."""
    return load_artifacts()


def _range(metadata: dict[str, Any], key: str, default_min: float, default_max: float) -> tuple[float, float]:
    ranges = metadata.get("numeric_ranges", {})
    column_range = ranges.get(key, {})
    return float(column_range.get("min", default_min)), float(column_range.get("max", default_max))


def _default(metadata: dict[str, Any], key: str, fallback: float) -> float:
    defaults = metadata.get("numeric_defaults", {})
    return float(defaults.get(key, fallback))


def _option(options: dict[str, list[str]], key: str, fallback: str = "Unknown") -> list[str]:
    values = options.get(key, [])
    return values if values else [fallback]


def _validate_single_policy_input(policy_row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if policy_row["Exposure"] <= 0:
        errors.append("Exposure must be greater than 0.")
    if policy_row["DrivAge"] < 18:
        errors.append("Driver age must be at least 18.")
    if policy_row["VehPower"] <= 0:
        errors.append("Vehicle power must be greater than 0.")
    if policy_row["Density"] < 0:
        errors.append("Population density cannot be negative.")
    return errors


def _risk_badge(risk_category: str) -> str:
    risk_color = {
        "Low": "#1f8a70",
        "Medium": "#d49406",
        "High": "#c0392b",
    }.get(risk_category, "#34495e")
    return (
        f"<div style='display:inline-block;padding:0.35rem 0.65rem;border-radius:0.5rem;"
        f"background:{risk_color};color:white;font-weight:600'>{risk_category}</div>"
    )


def _render_project_intro() -> None:
    st.title("French Motor Insurance Premium Predictor")
    st.caption(
        "Inference-only Streamlit app using pre-trained frequency and severity models "
        "from the freMTPL2 French motor insurance dataset."
    )

    with st.expander("About this project", expanded=True):
        st.markdown(
            "This app predicts actuarial pricing components for motor insurance policies.\n\n"
            "Step 1: Enter policy attributes in the input form.\n"
            "Step 2: Click Predict Premium to run saved models.\n"
            "Step 3: Review frequency, severity, expected loss, premium, and risk category."
        )


def _render_single_prediction_tab(artifacts: dict[str, Any]) -> None:
    metadata = artifacts["metadata"]
    options = metadata.get("categorical_options", {})

    exposure_min, exposure_max = _range(metadata, "Exposure", 0.01, 2.5)
    power_min, power_max = _range(metadata, "VehPower", 1.0, 20.0)
    veh_age_min, veh_age_max = _range(metadata, "VehAge", 0.0, 100.0)
    driv_age_min, driv_age_max = _range(metadata, "DrivAge", 18.0, 100.0)
    bonus_min, bonus_max = _range(metadata, "BonusMalus", 50.0, 350.0)
    density_min, density_max = _range(metadata, "Density", 0.0, 27000.0)

    st.subheader("Input form")
    with st.form("policy_input_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            exposure = st.number_input(
                "Exposure",
                min_value=max(0.01, float(exposure_min)),
                max_value=float(max(exposure_max, 2.5)),
                value=_default(metadata, "Exposure", 0.5),
                step=0.01,
                help="Policy exposure in years. 1.0 means one full policy-year.",
            )
            driv_age = st.slider(
                "Driver age",
                min_value=int(driv_age_min),
                max_value=int(driv_age_max),
                value=int(round(_default(metadata, "DrivAge", 40))),
                help="Age of the main policyholder.",
            )
            bonus_malus = st.slider(
                "Bonus-Malus",
                min_value=int(bonus_min),
                max_value=int(bonus_max),
                value=int(round(_default(metadata, "BonusMalus", 60))),
                help="Claims bonus-malus coefficient from prior insurance history.",
            )

        with col2:
            veh_power = st.slider(
                "Vehicle power",
                min_value=int(power_min),
                max_value=int(power_max),
                value=int(round(_default(metadata, "VehPower", 6))),
                help="Vehicle horsepower class variable used in freMTPL2.",
            )
            veh_age = st.slider(
                "Vehicle age",
                min_value=int(veh_age_min),
                max_value=int(veh_age_max),
                value=int(round(_default(metadata, "VehAge", 6))),
                help="Vehicle age in years.",
            )
            density = st.number_input(
                "Population density",
                min_value=int(density_min),
                max_value=int(density_max),
                value=int(round(_default(metadata, "Density", 400))),
                step=10,
                help="Policyholder area density value from dataset encoding.",
            )

        with col3:
            veh_brand = st.selectbox("Vehicle brand", _option(options, "VehBrand"), help="Vehicle brand code.")
            veh_gas = st.selectbox("Fuel type", _option(options, "VehGas"), help="Fuel category.")
            area = st.selectbox("Area", _option(options, "Area"), help="Area risk segment label.")
            region = st.selectbox("Region", _option(options, "Region"), help="French region label.")

        submitted = st.form_submit_button("Predict Premium", type="primary", use_container_width=True)

    if not submitted:
        st.info("Fill in the policy attributes and click Predict Premium.")
        return

    policy_row = {
        "IDpol": 0,
        "Exposure": float(exposure),
        "VehPower": int(veh_power),
        "VehAge": int(veh_age),
        "DrivAge": int(driv_age),
        "BonusMalus": int(bonus_malus),
        "VehBrand": str(veh_brand),
        "VehGas": str(veh_gas),
        "Area": str(area),
        "Density": int(density),
        "Region": str(region),
    }

    validation_errors = _validate_single_policy_input(policy_row)
    if validation_errors:
        st.error("Input validation failed. Please review the following:")
        for error in validation_errors:
            st.markdown(f"- {error}")
        return

    try:
        scored = predict_premiums(pd.DataFrame([policy_row]), artifacts)
        result = scored.iloc[0]
    except DataValidationError as exc:
        st.error(f"Input validation failed: {exc}")
        return
    except Exception as exc:
        st.error(f"Prediction failed unexpectedly: {exc}")
        return

    st.subheader("Prediction results")
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    metric_col1.metric("Predicted frequency", f"{result['predicted_annual_frequency']:.4f}")
    metric_col2.metric("Predicted severity", f"{result['predicted_claim_severity']:,.2f}")
    metric_col3.metric("Expected loss", f"{result['expected_loss']:,.2f}")
    metric_col4.metric("Final premium", f"{result['final_premium']:,.2f}")
    metric_col5.metric("Expected claim count", f"{result['predicted_claim_count']:.4f}")

    st.markdown("Risk category")
    st.markdown(_risk_badge(str(result["risk_category"])), unsafe_allow_html=True)

    premium_breakdown = pd.DataFrame(
        {
            "component": ["Expected loss", "Pure premium", "Technical premium", "Final premium"],
            "value": [
                float(result["expected_loss"]),
                float(result["pure_premium"]),
                float(result["technical_premium"]),
                float(result["final_premium"]),
            ],
        }
    ).set_index("component")
    st.caption("Premium breakdown")
    st.bar_chart(premium_breakdown)


def _render_batch_tab(artifacts: dict[str, Any]) -> None:
    st.subheader("Optional batch upload")
    st.markdown(
        "Upload a CSV containing: IDpol (optional), Exposure, VehPower, VehAge, DrivAge, "
        "BonusMalus, VehBrand, VehGas, Area, Density, Region."
    )

    template_df = pd.DataFrame(
        [
            {
                "IDpol": 10001,
                "Exposure": 1.0,
                "VehPower": 6,
                "VehAge": 5,
                "DrivAge": 40,
                "BonusMalus": 60,
                "VehBrand": "B12",
                "VehGas": "Regular",
                "Area": "C",
                "Density": 500,
                "Region": "Centre",
            }
        ]
    )
    st.download_button(
        "Download input template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="policy_input_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload policy CSV", type=["csv"])
    if uploaded_file is None:
        return

    try:
        batch_df = pd.read_csv(uploaded_file)
        scored_batch = predict_premiums(batch_df, artifacts)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        st.error(f"Invalid CSV format: {exc}")
        return
    except DataValidationError as exc:
        st.error(f"Input validation failed: {exc}")
        return
    except Exception as exc:
        st.error(f"Unexpected scoring error: {exc}")
        return

    st.success(f"Scored {len(scored_batch)} policies.")
    risk_counts = (
        scored_batch["risk_category"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
        .rename_axis("risk_category")
        .to_frame("count")
    )

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Low risk", int(risk_counts.loc["Low", "count"]))
    summary_col2.metric("Medium risk", int(risk_counts.loc["Medium", "count"]))
    summary_col3.metric("High risk", int(risk_counts.loc["High", "count"]))

    st.caption("Risk distribution")
    st.bar_chart(risk_counts)

    display_columns = [
        "IDpol",
        "Exposure",
        "predicted_annual_frequency",
        "predicted_claim_count",
        "predicted_claim_severity",
        "annualized_expected_loss",
        "expected_loss",
        "final_premium",
        "risk_score",
        "risk_category",
    ]
    st.dataframe(scored_batch[display_columns], use_container_width=True)

    st.download_button(
        label="Download predictions",
        data=scored_batch.to_csv(index=False).encode("utf-8"),
        file_name="freMTPL2_batch_predictions.csv",
        mime="text/csv",
        type="primary",
    )


def _render_model_info_tab(metadata: dict[str, Any], metrics: dict[str, Any]) -> None:
    st.subheader("How to use this app")
    st.markdown(
        "1. Provide policy features in the single-policy input form or via CSV upload.\n"
        "2. Click Predict Premium (or upload a CSV) to score policies.\n"
        "3. Interpret outputs as follows:\n"
        "- Predicted frequency: expected annual claims per policy-year.\n"
        "- Predicted severity: expected claim amount when a claim occurs.\n"
        "- Expected loss: expected claim cost for the policy term.\n"
        "- Final premium: loaded premium after pricing adjustments.\n"
        "- Risk category: Low/Medium/High underwriting segment."
    )

    st.subheader("Model and data summary")
    dataset_summary = metadata.get("dataset_summary", {})
    data_quality = metadata.get("data_quality", {})

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Policy rows", f"{dataset_summary.get('policy_rows', 0):,}")
    summary_col2.metric("Claim rows", f"{dataset_summary.get('claim_rows', 0):,}")
    summary_col3.metric(
        "Positive claim count without paid amount",
        f"{data_quality.get('positive_claim_count_without_paid_amount', 0):,}",
    )

    if metrics:
        st.subheader("Saved evaluation metrics")
        frequency_metrics = metrics.get("frequency", {})
        severity_metrics = metrics.get("severity", {})

        frequency_metric_values = frequency_metrics.get("metrics", frequency_metrics)
        severity_metric_values = severity_metrics.get("metrics", severity_metrics)

        freq_col1, freq_col2 = st.columns(2)
        freq_col1.metric("Frequency RMSE", f"{float(frequency_metric_values.get('rmse', 0.0)):.4f}")
        freq_col2.metric(
            "Frequency Poisson deviance",
            f"{float(frequency_metric_values.get('poisson_deviance', frequency_metric_values.get('mean_poisson_deviance', 0.0))):.4f}",
        )

        sev_col1, sev_col2 = st.columns(2)
        sev_col1.metric("Severity MAE", f"{float(severity_metric_values.get('mae', 0.0)):,.2f}")
        sev_col2.metric("Severity RMSE", f"{float(severity_metric_values.get('rmse', 0.0)):,.2f}")

    st.subheader("Saved training plots")
    plot_col1, plot_col2 = st.columns(2)
    premium_plot = Path("results/plots/premium_distribution.png")
    risk_plot = Path("results/plots/risk_distribution.png")
    frequency_plot = Path("results/plots/frequency_calibration.png")
    severity_plot = Path("results/plots/severity_actual_vs_pred.png")

    with plot_col1:
        if premium_plot.exists():
            st.image(str(premium_plot), caption="Final premium distribution", use_container_width=True)
        if frequency_plot.exists():
            st.image(str(frequency_plot), caption="Frequency calibration", use_container_width=True)

    with plot_col2:
        if risk_plot.exists():
            st.image(str(risk_plot), caption="Risk category distribution", use_container_width=True)
        if severity_plot.exists():
            st.image(str(severity_plot), caption="Severity actual vs predicted", use_container_width=True)


def main() -> None:
    _render_project_intro()

    try:
        artifacts = load_prediction_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Train models first using python train.py, then rerun this app.")
        st.stop()
    except Exception as exc:
        st.error(f"Failed to load model artifacts: {exc}")
        st.stop()

    metadata = artifacts.get("metadata", {})
    metrics = artifacts.get("metrics", {})

    tab_predict, tab_batch, tab_info = st.tabs(
        ["Single Policy Prediction", "Batch Upload", "Instructions and Model Info"]
    )
    with tab_predict:
        _render_single_prediction_tab(artifacts)
    with tab_batch:
        _render_batch_tab(artifacts)
    with tab_info:
        _render_model_info_tab(metadata, metrics)


if __name__ == "__main__":
    main()
