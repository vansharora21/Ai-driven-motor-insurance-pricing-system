from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from predict import predict_premiums
from src.model_artifacts import load_artifacts

st.set_page_config(page_title="Motor Insurance Pricing", layout="wide")


@st.cache_resource
def load_prediction_artifacts():
    """Load trained artifacts once per Streamlit session."""
    return load_artifacts()


try:
    artifacts = load_prediction_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Train the models first with `python train.py`.")
    st.stop()

metadata = artifacts["metadata"]
metrics = artifacts.get("metrics", {})
numeric_defaults = metadata["numeric_defaults"]
numeric_ranges = metadata["numeric_ranges"]
categorical_options = metadata["categorical_options"]

st.title("Motor Insurance Pricing Engine")
st.markdown(
    "Inference-only pricing app powered by the freMTPL2 French motor insurance dataset. "
    "The app loads saved production artifacts and does not retrain models at startup."
)

tab_calc, tab_upload, tab_portfolio = st.tabs(
    ["Premium Calculator", "Batch Risk Analyzer", "Portfolio Analytics"]
)

with tab_calc:
    st.header("Policy Input")
    col1, col2, col3 = st.columns(3)

    with col1:
        exposure = st.number_input(
            "Exposure",
            min_value=0.01,
            max_value=float(max(2.5, numeric_ranges["Exposure"]["max"])),
            value=float(numeric_defaults["Exposure"]),
            step=0.01,
        )
        driv_age = st.number_input(
            "Driver Age",
            min_value=int(numeric_ranges["DrivAge"]["min"]),
            max_value=int(numeric_ranges["DrivAge"]["max"]),
            value=int(round(numeric_defaults["DrivAge"])),
            step=1,
        )
        bonus_malus = st.number_input(
            "Bonus-Malus",
            min_value=int(numeric_ranges["BonusMalus"]["min"]),
            max_value=int(numeric_ranges["BonusMalus"]["max"]),
            value=int(round(numeric_defaults["BonusMalus"])),
            step=1,
        )

    with col2:
        veh_power = st.number_input(
            "Vehicle Power",
            min_value=int(numeric_ranges["VehPower"]["min"]),
            max_value=int(numeric_ranges["VehPower"]["max"]),
            value=int(round(numeric_defaults["VehPower"])),
            step=1,
        )
        veh_age = st.number_input(
            "Vehicle Age",
            min_value=int(numeric_ranges["VehAge"]["min"]),
            max_value=int(numeric_ranges["VehAge"]["max"]),
            value=int(round(numeric_defaults["VehAge"])),
            step=1,
        )
        density = st.number_input(
            "Population Density",
            min_value=int(numeric_ranges["Density"]["min"]),
            max_value=int(numeric_ranges["Density"]["max"]),
            value=int(round(numeric_defaults["Density"])),
            step=1,
        )

    with col3:
        veh_brand = st.selectbox("Vehicle Brand", categorical_options["VehBrand"])
        veh_gas = st.selectbox("Fuel Type", categorical_options["VehGas"])
        area = st.selectbox("Area", categorical_options["Area"])
        region = st.selectbox("Region", categorical_options["Region"])

    if st.button("Calculate Premium", type="primary", use_container_width=True):
        input_df = pd.DataFrame(
            [
                {
                    "IDpol": 0,
                    "Exposure": exposure,
                    "VehPower": veh_power,
                    "VehAge": veh_age,
                    "DrivAge": driv_age,
                    "BonusMalus": bonus_malus,
                    "VehBrand": veh_brand,
                    "VehGas": veh_gas,
                    "Area": area,
                    "Density": density,
                    "Region": region,
                }
            ]
        )

        scored = predict_premiums(input_df, artifacts)
        result = scored.iloc[0]

        st.subheader("Pricing Results")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Final Premium", f"{result['final_premium']:,.2f}")
        metric_col2.metric("Pure Premium", f"{result['pure_premium']:,.2f}")
        metric_col3.metric("Expected Claim Count", f"{result['predicted_claim_count']:.4f}")
        metric_col4.metric("Risk Category", result["risk_category"])

        detail_col1, detail_col2 = st.columns(2)
        detail_col1.metric("Annual Claim Frequency", f"{result['predicted_annual_frequency']:.4f}")
        detail_col2.metric("Expected Claim Severity", f"{result['predicted_claim_severity']:,.2f}")

with tab_upload:
    st.header("Batch Risk Analyzer")
    st.markdown(
        "Upload a CSV with columns "
        "`IDpol` (optional), `Exposure`, `VehPower`, `VehAge`, `DrivAge`, "
        "`BonusMalus`, `VehBrand`, `VehGas`, `Area`, `Density`, and `Region`."
    )

    uploaded_file = st.file_uploader("Upload Policy CSV", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        scored_batch = predict_premiums(batch_df, artifacts)

        st.success(f"Scored {len(scored_batch)} policies.")

        high_risk = int((scored_batch["risk_category"] == "High").sum())
        medium_risk = int((scored_batch["risk_category"] == "Medium").sum())
        low_risk = int((scored_batch["risk_category"] == "Low").sum())

        summary_col1, summary_col2, summary_col3 = st.columns(3)
        summary_col1.metric("High Risk", high_risk)
        summary_col2.metric("Medium Risk", medium_risk)
        summary_col3.metric("Low Risk", low_risk)

        display_columns = [
            "IDpol",
            "Exposure",
            "predicted_annual_frequency",
            "predicted_claim_count",
            "predicted_claim_severity",
            "pure_premium",
            "final_premium",
            "risk_category",
        ]
        st.dataframe(scored_batch[display_columns], use_container_width=True)

        csv_data = scored_batch.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Predictions",
            data=csv_data,
            file_name="freMTPL2_batch_predictions.csv",
            mime="text/csv",
            type="primary",
        )

with tab_portfolio:
    st.header("Portfolio Analytics")
    dataset_summary = metadata.get("dataset_summary", {})
    data_quality = metadata.get("data_quality", {})

    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Policies", f"{dataset_summary.get('policy_rows', 0):,}")
    info_col2.metric("Claim Rows", f"{dataset_summary.get('claim_rows', 0):,}")
    info_col3.metric("Claims Missing Paid Amount", data_quality.get("positive_claim_count_without_paid_amount", 0))

    if metrics:
        st.subheader("Evaluation Metrics")
        freq_col1, freq_col2 = st.columns(2)
        freq_col1.metric("Frequency RMSE", f"{metrics['frequency']['rmse']:.4f}")
        freq_col2.metric("Poisson Deviance", f"{metrics['frequency']['mean_poisson_deviance']:.4f}")

        sev_col1, sev_col2 = st.columns(2)
        sev_col1.metric("Severity MAE", f"{metrics['severity']['mae']:,.2f}")
        sev_col2.metric("Severity RMSE", f"{metrics['severity']['rmse']:,.2f}")

    st.subheader("Saved Training Plots")
    plot_col1, plot_col2 = st.columns(2)

    premium_plot = Path("results/plots/premium_distribution.png")
    risk_plot = Path("results/plots/risk_distribution.png")
    frequency_plot = Path("results/plots/frequency_calibration.png")
    severity_plot = Path("results/plots/severity_actual_vs_pred.png")

    with plot_col1:
        if premium_plot.exists():
            st.image(str(premium_plot), caption="Final Premium Distribution", use_container_width=True)
        if frequency_plot.exists():
            st.image(str(frequency_plot), caption="Frequency Calibration", use_container_width=True)

    with plot_col2:
        if risk_plot.exists():
            st.image(str(risk_plot), caption="Risk Category Distribution", use_container_width=True)
        if severity_plot.exists():
            st.image(str(severity_plot), caption="Severity Actual vs Predicted", use_container_width=True)

    st.info("Run `python train.py` whenever you want to refresh the saved models, metrics, and plots.")
