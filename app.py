from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from predict import predict_premiums
from src.data_loader import DataValidationError
from src.model_artifacts import load_artifacts

st.set_page_config(page_title="Motor Insurance Pricing", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


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


def _normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _first_number(value: Any, fallback: float) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return float(fallback)

    text = str(value).strip().lower()
    if not text:
        return float(fallback)

    range_match = re.search(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)", text)
    if range_match:
        return float((float(range_match.group(1)) + float(range_match.group(2))) / 2.0)

    number_match = re.search(r"\d+(?:\.\d+)?", text)
    if number_match:
        return float(number_match.group(0))

    return float(fallback)


def _level_from_text(value: Any) -> str:
    text = _normalize_text(value)
    if "high" in text:
        return "High"
    if "medium" in text:
        return "Medium"
    if "low" in text:
        return "Low"
    return "Medium"


def _yes_no_flag(value: Any) -> int:
    text = _normalize_text(value)
    if text.startswith("y") or "yes" in text or text == "1":
        return 1
    return 0


def _vehicle_power_from_type(vehicle_type: Any) -> int:
    mapping = {
        "sedan": 6,
        "suv": 8,
        "hatchback": 5,
        "truck": 12,
        "motorcycle": 4,
        "other": 6,
    }
    return int(mapping.get(_normalize_text(vehicle_type), 6))


def _brand_from_vehicle_type(vehicle_type: Any) -> str:
    mapping = {
        "sedan": "B12",
        "suv": "B11",
        "hatchback": "B10",
        "truck": "B7",
        "motorcycle": "B2",
        "other": "B12",
    }
    return str(mapping.get(_normalize_text(vehicle_type), "B12"))


def _gas_from_vehicle_type(vehicle_type: Any) -> str:
    if _normalize_text(vehicle_type) in {"truck", "motorcycle"}:
        return "Diesel"
    return "Regular"


def _area_from_behavior(night_level: str, braking_level: str, accidents: float, claim_history: int) -> str:
    risk_score = 0
    risk_score += {"Low": 0, "Medium": 1, "High": 2}.get(night_level, 1)
    risk_score += {"Low": 0, "Medium": 1, "High": 2}.get(braking_level, 1)
    risk_score += int(round(accidents))
    risk_score += int(claim_history)
    if risk_score <= 1:
        return "C"
    if risk_score <= 3:
        return "D"
    if risk_score <= 5:
        return "E"
    return "F"


def _density_from_behavior(daily_mileage: float, accidents: float, claim_history: int) -> int:
    density = 350.0 + (daily_mileage * 12.0) + (accidents * 180.0) + (claim_history * 140.0)
    return int(min(max(density, 0.0), 27000.0))


def _bonus_malus_from_behavior(daily_mileage: float, accidents: float, claim_history: int, night_level: str, braking_level: str, experience_years: float) -> int:
    score = 50.0
    score += accidents * 14.0
    score += claim_history * 18.0
    score += {"Low": 0.0, "Medium": 8.0, "High": 16.0}.get(night_level, 8.0)
    score += {"Low": 0.0, "Medium": 6.0, "High": 12.0}.get(braking_level, 6.0)
    score += min(daily_mileage / 10.0, 25.0)
    score -= min(experience_years, 30.0) * 0.6
    return int(min(max(score, 50.0), 350.0))


def _sort_latest_csv(data_dir: Path = DATA_DIR) -> list[Path]:
    if not data_dir.exists():
        return []
    return sorted(
        [path for path in data_dir.glob("*.csv") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _load_latest_csv(data_dir: Path = DATA_DIR) -> Path | None:
    csv_paths = _sort_latest_csv(data_dir)
    return csv_paths[0] if csv_paths else None


def _google_form_to_pricing_frame(form_df: pd.DataFrame) -> pd.DataFrame:
    if form_df.empty:
        raise DataValidationError("Google Form CSV is empty.")

    normalized_lookup = {_normalize_text(column): column for column in form_df.columns}

    def _get_value(row: pd.Series, *candidates: str, default: Any = None) -> Any:
        for candidate in candidates:
            normalized = _normalize_text(candidate)
            if normalized in normalized_lookup:
                value = row[normalized_lookup[normalized]]
                if pd.notna(value) and str(value).strip() != "":
                    return value
        return default

    transformed_rows: list[dict[str, Any]] = []
    for index, row in form_df.reset_index(drop=True).iterrows():
        age = _first_number(_get_value(row, "Age (Must be between 18 and 100)", "age"), 40.0)
        vehicle_age = _first_number(_get_value(row, "How old is your vehicle (in years)?", "vehicle_age"), 6.0)
        daily_mileage = _first_number(_get_value(row, "On average, how many kilometers do you drive per day?", "daily_mileage"), 20.0)
        night_level = _level_from_text(_get_value(row, "Night Driving Frequency (10 PM – 5 AM)", "night_driving_level"))
        braking_level = _level_from_text(_get_value(row, "Harsh Braking Frequency (How often do you brake abruptly in traffic?)", "harsh_braking_level"))
        accidents = _first_number(_get_value(row, "How many traffic accidents have you been involved in during the last 2 years?", "accidents_last_2yr"), 0.0)
        claim_history = _yes_no_flag(_get_value(row, "Have you filed a motor insurance claim in the last 2 years?", "claim_history"))
        experience_years = _first_number(_get_value(row, "How many years of driving experience do you have?", "driving_experience_years"), max(age - 18.0, 0.0))
        vehicle_type = _get_value(row, "Vehicle Type", "vehicle_type", default="Other")
        source_identifier = _get_value(row, "Driver Identifier (Email or Unique ID)", "driver_id", "Email address", "Email Address", default=f"live-{index + 1}")

        exposure = float(min(max(daily_mileage / 20.0, 0.1), 2.5))
        veh_power = _vehicle_power_from_type(vehicle_type)
        bonus_malus = _bonus_malus_from_behavior(daily_mileage, accidents, claim_history, night_level, braking_level, experience_years)
        area = _area_from_behavior(night_level, braking_level, accidents, claim_history)
        density = _density_from_behavior(daily_mileage, accidents, claim_history)

        transformed_rows.append(
            {
                "source_record_type": "Google Form response",
                "source_driver_identifier": source_identifier,
                "source_email": _get_value(row, "Email address", "Email Address", default=""),
                "source_timestamp": _get_value(row, "Timestamp", default=""),
                "source_vehicle_type": vehicle_type,
                "source_daily_mileage": daily_mileage,
                "source_night_driving_level": night_level,
                "source_harsh_braking_level": braking_level,
                "source_accidents_last_2yr": accidents,
                "source_claim_history": claim_history,
                "source_driving_experience_years": experience_years,
                "IDpol": index + 1,
                "Exposure": exposure,
                "VehPower": veh_power,
                "VehAge": int(round(vehicle_age)),
                "DrivAge": int(round(age)),
                "BonusMalus": bonus_malus,
                "VehBrand": _brand_from_vehicle_type(vehicle_type),
                "VehGas": _gas_from_vehicle_type(vehicle_type),
                "Area": area,
                "Density": density,
                "Region": "Centre",
            }
        )

    return pd.DataFrame(transformed_rows)


def _plot_pricing_proof(result: pd.Series) -> plt.Figure:
    """Build a compact waterfall chart that makes the pricing calculation visible."""
    expected_loss = float(result["expected_loss"])
    pure_premium = float(result["pure_premium"])
    technical_premium = float(result["technical_premium"])
    final_premium = float(result["final_premium"])

    steps = [
        ("Expected loss", expected_loss),
        ("Expense loading", pure_premium * 0.30),
        ("Fixed expense", 50.0),
        ("Technical premium", technical_premium),
        ("Minimum premium adjustment", max(final_premium - technical_premium, 0.0)),
        ("Final premium", final_premium),
    ]

    figure, axis = plt.subplots(figsize=(11.5, 5.5))
    running_total = 0.0
    x_positions = list(range(len(steps)))
    bar_labels = []
    bar_colors = []
    bar_bottoms = []
    bar_heights = []

    for label, value in steps:
        bar_labels.append(label)
        if label in {"Expected loss", "Technical premium", "Final premium"}:
            bar_colors.append("#2563eb")
            bar_bottoms.append(0.0)
            bar_heights.append(value)
            running_total = value
        else:
            bar_colors.append("#d97706")
            bar_bottoms.append(running_total)
            bar_heights.append(value)
            running_total += value

    axis.bar(x_positions, bar_heights, bottom=bar_bottoms, color=bar_colors, width=0.7)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(bar_labels, rotation=20, ha="right")
    axis.set_ylabel("Euro")
    axis.set_title("Waterfall pricing proof for this policy")
    axis.grid(axis="y", alpha=0.25)

    for idx, (bottom, height) in enumerate(zip(bar_bottoms, bar_heights, strict=False)):
        axis.text(idx, bottom + height + max(final_premium * 0.015, 1.0), f"{height:,.2f}", ha="center", va="bottom", fontsize=9)

    axis.axhline(0, color="#334155", linewidth=0.8)
    figure.tight_layout()
    return figure


def _plot_batch_pricing_proof(scored_batch: pd.DataFrame) -> plt.Figure:
    """Plot batch outputs so the pricing relationship is visible on live data."""
    figure, axis = plt.subplots(figsize=(9.5, 6))

    palette = {"Low": "#1f8a70", "Medium": "#d49406", "High": "#c0392b"}
    for risk_category, color in palette.items():
        subset = scored_batch[scored_batch["risk_category"] == risk_category]
        if subset.empty:
            continue
        axis.scatter(
            subset["expected_loss"],
            subset["final_premium"],
            s=38,
            alpha=0.75,
            color=color,
            label=risk_category,
        )

    max_value = float(max(scored_batch["expected_loss"].max(), scored_batch["final_premium"].max()))
    axis.plot([0, max_value * 1.1], [0, max_value * 1.1], linestyle="--", color="#94a3b8", label="1:1 reference")
    axis.set_title("Live batch pricing output")
    axis.set_xlabel("Expected loss")
    axis.set_ylabel("Final premium")
    axis.legend(title="Risk category", frameon=False)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def _score_dataframe(dataframe: pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    try:
        return predict_premiums(dataframe, artifacts)
    except DataValidationError:
        if any(_normalize_text(column) in {_normalize_text("Age (Must be between 18 and 100)"), _normalize_text("Vehicle Type")} for column in dataframe.columns):
            live_frame = _google_form_to_pricing_frame(dataframe)
            scored = predict_premiums(live_frame, artifacts)
            return scored
        raise


def _render_scored_preview(scored_df: pd.DataFrame, title: str) -> None:
    st.subheader(title)

    if "source_driver_identifier" in scored_df.columns:
        preview_columns = [
            "source_driver_identifier",
            "source_email",
            "predicted_annual_frequency",
            "predicted_claim_severity",
            "expected_loss",
            "final_premium",
            "risk_category",
        ]
    else:
        preview_columns = [
            "IDpol",
            "predicted_annual_frequency",
            "predicted_claim_severity",
            "expected_loss",
            "final_premium",
            "risk_category",
        ]

    available_columns = [column for column in preview_columns if column in scored_df.columns]
    if available_columns:
        st.dataframe(scored_df[available_columns].head(20), use_container_width=True, hide_index=True)

    if len(scored_df) == 1:
        st.caption("Pricing proof chart")
        st.pyplot(_plot_pricing_proof(scored_df.iloc[0]), clear_figure=True)
    else:
        st.caption("Batch pricing proof")
        st.pyplot(_plot_batch_pricing_proof(scored_df), clear_figure=True)


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

    st.subheader("Pricing output")
    result_summary = pd.DataFrame(
        {
            "metric": [
                "Predicted frequency",
                "Predicted severity",
                "Expected loss",
                "Technical premium",
                "Final premium",
                "Risk category",
            ],
            "value": [
                f"{result['predicted_annual_frequency']:.4f}",
                f"{result['predicted_claim_severity']:,.2f}",
                f"{result['expected_loss']:,.2f}",
                f"{result['technical_premium']:,.2f}",
                f"{result['final_premium']:,.2f}",
                str(result["risk_category"]),
            ],
        }
    )
    st.dataframe(result_summary, hide_index=True, use_container_width=True)

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
    st.caption("Pricing proof chart")
    st.pyplot(_plot_pricing_proof(result), clear_figure=True)


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

    st.caption("Batch pricing proof")
    st.pyplot(_plot_batch_pricing_proof(scored_batch), clear_figure=True)

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


def _render_live_form_tab(artifacts: dict[str, Any]) -> None:
    st.subheader("Live Google Form responses")
    st.markdown(
        "This section reads the newest CSV in the data folder, which currently contains Google Form responses. "
        "Because the form collects behavioral survey fields rather than the freMTPL2 actuarial schema, the app maps "
        "those responses into transparent proxy pricing inputs before scoring."
    )

    latest_csv = _load_latest_csv()
    if latest_csv is None:
        st.info("No CSV file was found in the data folder.")
        return

    st.caption(f"Latest live file: {latest_csv.name}")

    try:
        live_raw_df = pd.read_csv(latest_csv)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        st.error(f"Could not read the live CSV: {exc}")
        return

    st.caption("Raw live intake preview")
    st.dataframe(live_raw_df.head(20), use_container_width=True, hide_index=True)

    try:
        live_model_input = _google_form_to_pricing_frame(live_raw_df)
        scored_live = predict_premiums(live_model_input, artifacts)
    except DataValidationError as exc:
        st.error(f"Live data could not be scored: {exc}")
        return
    except Exception as exc:
        st.error(f"Unexpected live scoring error: {exc}")
        return

    st.success(f"Scored {len(scored_live)} live responses from the latest form export.")
    _render_scored_preview(scored_live, "Live pricing output")

    if "source_driver_identifier" in scored_live.columns:
        st.caption("Mapped pricing inputs used by the model")
        mapped_columns = [
            "source_driver_identifier",
            "Exposure",
            "VehPower",
            "VehAge",
            "DrivAge",
            "BonusMalus",
            "VehBrand",
            "VehGas",
            "Area",
            "Density",
            "Region",
        ]
        st.dataframe(scored_live[mapped_columns].head(20), use_container_width=True, hide_index=True)
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

    tab_predict, tab_batch, tab_live, tab_info = st.tabs(
        ["Single Policy Prediction", "Batch Upload", "Live Google Form Data", "Instructions and Model Info"]
    )
    with tab_predict:
        _render_single_prediction_tab(artifacts)
    with tab_batch:
        _render_batch_tab(artifacts)
    with tab_live:
        _render_live_form_tab(artifacts)
    with tab_info:
        _render_model_info_tab(metadata, metrics)


if __name__ == "__main__":
    main()
