from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline

from src.config import get_random_seed
from src.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
REPORTS_DIR = PROJECT_ROOT / "results" / "premium_reports"
EVALUATION_DIR = PROJECT_ROOT / "results" / "evaluation"
MODEL_COMPARISON_RMSE_PLOT = PLOTS_DIR / "model_comparison_rmse.png"
PREDICTED_VS_ACTUAL_PLOT = PLOTS_DIR / "predicted_vs_actual.png"
ERROR_DISTRIBUTION_PLOT = PLOTS_DIR / "error_distribution.png"
FEATURE_IMPORTANCE_PLOT_PREFIX = "feature_importance_"
FEATURE_IMPORTANCE_SUPPORTED_REGRESSORS = {"RandomForestRegressor", "XGBRegressor"}

_PREDICTION_COLUMN_PAIRS: tuple[tuple[str, str], ...] = (
    ("actual", "predicted"),
    ("y_true", "y_pred"),
    ("observed", "predicted"),
    ("ClaimAmount", "predicted_claim_severity"),
    ("ClaimNb", "predicted_claim_count"),
)

_COLUMN_LABELS = {
    "actual": "Actual Value",
    "predicted": "Predicted Value",
    "y_true": "Actual Value",
    "y_pred": "Predicted Value",
    "observed": "Observed Value",
    "ClaimAmount": "Actual Claim Amount",
    "predicted_claim_severity": "Predicted Claim Amount",
    "ClaimNb": "Actual Claim Count",
    "predicted_claim_count": "Predicted Claim Count",
    "error": "Residual",
}


def ensure_output_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


def _set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "axes.titleweight": "semibold",
            "axes.labelweight": "medium",
        }
    )


def _save_current_figure(save_path: Path) -> Path:
    try:
        plt.savefig(save_path)
        return save_path
    except PermissionError:
        fallback_path = save_path.with_name(f"{save_path.stem}_latest{save_path.suffix}")
        plt.savefig(fallback_path)
        return fallback_path


def _save_dataframe(df: pd.DataFrame, output_path: Path) -> Path:
    try:
        df.to_csv(output_path, index=False)
        return output_path
    except PermissionError:
        fallback_path = output_path.with_name(f"{output_path.stem}_latest{output_path.suffix}")
        df.to_csv(fallback_path, index=False)
        return fallback_path


def _load_table_input(data: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.read_csv(Path(data))


def _load_model_comparison_input(model_comparison: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(model_comparison, Mapping):
        return dict(model_comparison)
    with Path(model_comparison).open("r", encoding="utf-8") as comparison_file:
        return json.load(comparison_file)


def _pretty_label(column_name: str) -> str:
    return _COLUMN_LABELS.get(column_name, column_name.replace("_", " ").strip().title())


def _resolve_tree_model_label(regressor: Any) -> str | None:
    class_name = regressor.__class__.__name__.lower()
    if class_name == "randomforestregressor":
        return "random_forest"
    if class_name == "xgbregressor":
        return "xgboost"
    return None


def _snake_case_name(name: str) -> str:
    normalized = []
    for character in name:
        if character.isupper() and normalized and normalized[-1] != "_":
            normalized.append("_")
        normalized.append(character.lower())
    return "".join(normalized).replace("__", "_").strip("_")


def _clean_feature_name(feature_name: str) -> str:
    cleaned_name = str(feature_name)
    if "__" in cleaned_name:
        _, cleaned_name = cleaned_name.split("__", 1)
    if cleaned_name.startswith("numeric__"):
        cleaned_name = cleaned_name.split("__", 1)[1]
    elif cleaned_name.startswith("categorical__"):
        cleaned_name = cleaned_name.split("__", 1)[1]
        if "_" in cleaned_name:
            base_name, category_value = cleaned_name.split("_", 1)
            cleaned_name = f"{base_name} = {category_value}"
    return cleaned_name.replace("_", " ")


def _extract_feature_importance_frame(model: Any) -> tuple[pd.DataFrame | None, str]:
    regressor = getattr(model, "named_steps", {}).get("regressor") if hasattr(model, "named_steps") else None
    regressor_name = regressor.__class__.__name__ if regressor is not None else model.__class__.__name__

    if regressor_name not in FEATURE_IMPORTANCE_SUPPORTED_REGRESSORS:
        return None, regressor_name

    feature_importances = getattr(regressor, "feature_importances_", None)
    preprocessor = getattr(model, "named_steps", {}).get("preprocessor") if hasattr(model, "named_steps") else None
    if feature_importances is None or preprocessor is None or not hasattr(preprocessor, "get_feature_names_out"):
        return None, regressor_name

    feature_names = preprocessor.get_feature_names_out()
    if len(feature_names) != len(feature_importances):
        raise ValueError(
            "Feature importance values do not align with the transformed feature names."
        )

    importance_frame = pd.DataFrame(
        {
            "feature": [_clean_feature_name(feature_name) for feature_name in feature_names],
            "importance": pd.Series(feature_importances, dtype=float),
        }
    ).sort_values(by="importance", ascending=False)
    return importance_frame, regressor_name


def plot_feature_importance(
    model: Any,
    save_path: Path | None = None,
    model_label: str | None = None,
    top_n: int = 10,
    print_top_n: bool = True,
) -> Path | None:
    """Plot feature importance for supported tree-based regressors."""
    ensure_output_dirs()
    _set_plot_style()

    importance_frame, regressor_name = _extract_feature_importance_frame(model)
    if importance_frame is None:
        return None

    resolved_label = model_label or _snake_case_name(regressor_name)
    target_path = save_path or (PLOTS_DIR / f"feature_importance_{resolved_label}.png")

    top_features = importance_frame.head(max(1, top_n)).copy()
    plot_df = top_features.sort_values(by="importance", ascending=True)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=plot_df, x="importance", y="feature", color="#2563eb")
    ax.set_title(f"Feature Importance - {resolved_label.replace('_', ' ').title()}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    sns.despine()
    plt.tight_layout()
    output_path = _save_current_figure(target_path)
    plt.close()

    if print_top_n:
        top_preview = top_features.head(min(top_n, len(top_features))).copy()
        logger.info("Top %d features for %s (%s):", len(top_preview), resolved_label, regressor_name)
        for _, row in top_preview.iterrows():
            logger.info("  %s: %.6f", row["feature"], row["importance"])

    return output_path


def _resolve_prediction_columns(
    predictions: pd.DataFrame,
    actual_column: str | None = None,
    predicted_column: str | None = None,
) -> tuple[str, str]:
    if actual_column is not None and predicted_column is not None:
        if actual_column not in predictions.columns or predicted_column not in predictions.columns:
            raise ValueError(
                f"Predictions dataframe must contain '{actual_column}' and '{predicted_column}' columns."
            )
        return actual_column, predicted_column

    for resolved_actual, resolved_predicted in _PREDICTION_COLUMN_PAIRS:
        if resolved_actual in predictions.columns and resolved_predicted in predictions.columns:
            return resolved_actual, resolved_predicted

    raise ValueError(
        "Could not infer actual and predicted columns. Provide actual_column and predicted_column explicitly."
    )


def _prepare_prediction_frame(
    predictions: pd.DataFrame | str | Path,
    actual_column: str | None = None,
    predicted_column: str | None = None,
    sample_size: int = 5000,
) -> pd.DataFrame:
    frame = _load_table_input(predictions)
    resolved_actual, resolved_predicted = _resolve_prediction_columns(frame, actual_column, predicted_column)

    plot_df = frame[[resolved_actual, resolved_predicted]].copy()
    plot_df[resolved_actual] = pd.to_numeric(plot_df[resolved_actual], errors="coerce")
    plot_df[resolved_predicted] = pd.to_numeric(plot_df[resolved_predicted], errors="coerce")
    plot_df = plot_df.dropna(subset=[resolved_actual, resolved_predicted])

    if plot_df.empty:
        raise ValueError("Predictions dataframe does not contain any numeric actual/predicted pairs.")

    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(sample_size, random_state=get_random_seed())

    plot_df = plot_df.rename(columns={resolved_actual: "actual", resolved_predicted: "predicted"})
    plot_df["error"] = plot_df["actual"] - plot_df["predicted"]
    return plot_df


def _build_model_comparison_frame(model_comparison: Mapping[str, Any] | str | Path) -> pd.DataFrame:
    payload = _load_model_comparison_input(model_comparison)
    records: list[dict[str, Any]] = []

    for run_name, run_payload in payload.items():
        if not isinstance(run_payload, Mapping) or run_payload.get("status") != "success":
            continue

        metrics = run_payload.get("metrics", {})
        for component_name in ("frequency", "severity"):
            component_metrics = metrics.get(component_name, {})
            rmse = component_metrics.get("metrics", {}).get("rmse")
            if rmse is None:
                continue

            records.append(
                {
                    "run_name": run_name,
                    "combo_label": f"{run_payload.get('frequency_model', 'frequency')} + {run_payload.get('severity_model', 'severity')}",
                    "component": component_name.title(),
                    "rmse": float(rmse),
                }
            )

    comparison_df = pd.DataFrame.from_records(records)
    if comparison_df.empty:
        raise ValueError("No successful model comparison RMSE values were found in the input payload.")
    return comparison_df


def extract_feature_importance(model: Pipeline) -> pd.DataFrame | None:
    """Extract transformed feature importances from a fitted tree-based pipeline."""
    if not isinstance(model, Pipeline):
        return None

    if "preprocessor" not in model.named_steps or "regressor" not in model.named_steps:
        return None

    regressor = model.named_steps["regressor"]
    model_label = _resolve_tree_model_label(regressor)
    if model_label is None or not hasattr(regressor, "feature_importances_"):
        return None

    preprocessor = model.named_steps["preprocessor"]
    if not hasattr(preprocessor, "get_feature_names_out"):
        return None

    feature_names = preprocessor.get_feature_names_out()
    importances = getattr(regressor, "feature_importances_", None)
    if importances is None:
        return None

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": pd.Series(importances, dtype=float),
        }
    ).sort_values(by="importance", ascending=False, ignore_index=True)
    importance_df.attrs["model_label"] = model_label
    importance_df.attrs["estimator_name"] = regressor.__class__.__name__
    return importance_df


def plot_feature_importance(
    model: Pipeline,
    save_path: Path | None = None,
    top_n: int = 10,
    model_label: str | None = None,
) -> Path | None:
    """Plot feature importance for supported tree-based models and skip GLM models."""
    ensure_output_dirs()
    _set_plot_style()

    importance_df = extract_feature_importance(model)
    if importance_df is None:
        return None

    regressor = model.named_steps["regressor"]
    resolved_label = model_label or importance_df.attrs.get("model_label") or _resolve_tree_model_label(regressor)
    if resolved_label is None:
        return None

    save_path = save_path or (PLOTS_DIR / f"{FEATURE_IMPORTANCE_PLOT_PREFIX}{resolved_label}.png")

    plot_df = importance_df.head(top_n).sort_values(by="importance", ascending=True)
    figure_height = max(5.5, 0.45 * len(plot_df) + 2.0)

    plt.figure(figsize=(10.5, figure_height))
    ax = sns.barplot(data=plot_df, x="importance", y="feature", color="#2563eb")
    ax.set_title(f"{resolved_label.replace('_', ' ').title()} Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlim(0, max(importance_df["importance"].head(top_n).max() * 1.15, 0.01))
    sns.despine()
    plt.tight_layout()
    output_path = _save_current_figure(save_path)
    plt.close()
    return output_path


def plot_model_comparison_rmse(
    model_comparison: Mapping[str, Any] | str | Path,
    save_path: Path | None = None,
) -> Path:
    """Plot RMSE across model-combination runs for both frequency and severity components."""
    ensure_output_dirs()
    _set_plot_style()
    save_path = save_path or MODEL_COMPARISON_RMSE_PLOT

    comparison_df = _build_model_comparison_frame(model_comparison)

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=comparison_df,
        x="combo_label",
        y="rmse",
        hue="component",
        palette={"Frequency": "#1f77b4", "Severity": "#ff7f0e"},
    )
    ax.set_title("Model Comparison RMSE")
    ax.set_xlabel("Model combination")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Metric component", frameon=False)
    sns.despine()
    plt.tight_layout()
    output_path = _save_current_figure(save_path)
    plt.close()
    return output_path


def plot_predicted_vs_actual(
    predictions: pd.DataFrame | str | Path,
    save_path: Path | None = None,
    actual_column: str | None = None,
    predicted_column: str | None = None,
    sample_size: int = 5000,
) -> Path:
    """Plot predicted versus actual values with a 45-degree reference line."""
    ensure_output_dirs()
    _set_plot_style()
    save_path = save_path or PREDICTED_VS_ACTUAL_PLOT

    plot_df = _prepare_prediction_frame(
        predictions,
        actual_column=actual_column,
        predicted_column=predicted_column,
        sample_size=sample_size,
    )

    axis_min = float(min(plot_df["actual"].min(), plot_df["predicted"].min()))
    axis_max = float(max(plot_df["actual"].max(), plot_df["predicted"].max()))
    padding = max((axis_max - axis_min) * 0.05, 1.0)
    lower_bound = axis_min - padding
    upper_bound = axis_max + padding

    plt.figure(figsize=(8.5, 8.5))
    ax = sns.scatterplot(data=plot_df, x="actual", y="predicted", alpha=0.35, s=24, color="#2563eb")
    ax.plot([lower_bound, upper_bound], [lower_bound, upper_bound], linestyle="--", color="#111827", linewidth=1.6)
    ax.set_xlim(lower_bound, upper_bound)
    ax.set_ylim(lower_bound, upper_bound)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Predicted vs Actual")
    ax.set_xlabel(_pretty_label(actual_column or "actual"))
    ax.set_ylabel(_pretty_label(predicted_column or "predicted"))
    sns.despine()
    plt.tight_layout()
    output_path = _save_current_figure(save_path)
    plt.close()
    return output_path


def plot_error_distribution(
    predictions: pd.DataFrame | str | Path,
    save_path: Path | None = None,
    actual_column: str | None = None,
    predicted_column: str | None = None,
    sample_size: int = 5000,
) -> Path:
    """Plot the residual distribution for predicted values."""
    ensure_output_dirs()
    _set_plot_style()
    save_path = save_path or ERROR_DISTRIBUTION_PLOT

    plot_df = _prepare_prediction_frame(
        predictions,
        actual_column=actual_column,
        predicted_column=predicted_column,
        sample_size=sample_size,
    )

    plt.figure(figsize=(9, 6))
    ax = sns.histplot(plot_df["error"], bins=40, kde=True, color="#c2410c", edgecolor="white")
    ax.axvline(0, color="#111827", linestyle="--", linewidth=1.6, label="Zero error")
    ax.axvline(plot_df["error"].mean(), color="#2563eb", linestyle="-", linewidth=1.4, label="Mean error")
    ax.set_title("Prediction Error Distribution")
    ax.set_xlabel(_pretty_label("error"))
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    output_path = _save_current_figure(save_path)
    plt.close()
    return output_path


def generate_model_comparison_visualizations(
    model_comparison: Mapping[str, Any] | str | Path,
    predictions: pd.DataFrame | str | Path,
    output_dir: Path | None = None,
    actual_column: str | None = None,
    predicted_column: str | None = None,
) -> dict[str, Path]:
    """Generate the model comparison and prediction diagnostics in one call."""
    ensure_output_dirs()
    target_dir = output_dir or PLOTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    return {
        "model_comparison_rmse": plot_model_comparison_rmse(
            model_comparison,
            save_path=target_dir / "model_comparison_rmse.png",
        ),
        "predicted_vs_actual": plot_predicted_vs_actual(
            predictions,
            save_path=target_dir / "predicted_vs_actual.png",
            actual_column=actual_column,
            predicted_column=predicted_column,
        ),
        "error_distribution": plot_error_distribution(
            predictions,
            save_path=target_dir / "error_distribution.png",
            actual_column=actual_column,
            predicted_column=predicted_column,
        ),
    }


def plot_frequency_calibration(actual: pd.Series, predicted: pd.Series, save_path: Path | None = None) -> Path:
    """Plot binned actual vs predicted claim counts for the frequency model."""
    ensure_output_dirs()
    save_path = save_path or (PLOTS_DIR / "frequency_calibration.png")

    calibration_df = pd.DataFrame({"actual": actual, "predicted": predicted}).copy()
    calibration_df["bucket"] = pd.qcut(
        calibration_df["predicted"].rank(method="first"),
        q=min(10, len(calibration_df)),
        labels=False,
        duplicates="drop",
    )
    summary = calibration_df.groupby("bucket", observed=False).agg(
        actual_mean=("actual", "mean"),
        predicted_mean=("predicted", "mean"),
    )

    plt.figure(figsize=(9, 6))
    plt.plot(summary.index + 1, summary["actual_mean"], marker="o", label="Actual")
    plt.plot(summary.index + 1, summary["predicted_mean"], marker="o", label="Predicted")
    plt.title("Frequency Calibration by Prediction Decile")
    plt.xlabel("Prediction Decile")
    plt.ylabel("Average Claim Count")
    plt.legend()
    plt.tight_layout()
    output_path = _save_current_figure(save_path)
    plt.close()
    return output_path


def plot_severity_predictions(actual: pd.Series, predicted: pd.Series, save_path: Path | None = None) -> Path:
    """Scatter plot of actual vs predicted claim severity on a sampled holdout set."""
    save_path = save_path or (PLOTS_DIR / "severity_actual_vs_pred.png")
    return plot_predicted_vs_actual(
        pd.DataFrame({"ClaimAmount": actual, "predicted_claim_severity": predicted}),
        save_path=save_path,
        actual_column="ClaimAmount",
        predicted_column="predicted_claim_severity",
    )


def plot_premium_distribution(df: pd.DataFrame, save_path: Path | None = None) -> Path:
    """Plot the final premium distribution on a log x-axis for heavy-tailed data."""
    ensure_output_dirs()
    save_path = save_path or (PLOTS_DIR / "premium_distribution.png")

    plt.figure(figsize=(10, 6))
    sns.histplot(df["final_premium"], bins=40, color="#1f8a70")
    plt.xscale("log")
    plt.title("Distribution of Final Premiums")
    plt.xlabel("Final Premium (log scale)")
    plt.ylabel("Policy Count")
    plt.tight_layout()
    output_path = _save_current_figure(save_path)
    plt.close()
    return output_path


def plot_risk_distribution(df: pd.DataFrame, save_path: Path | None = None) -> Path:
    """Plot the count of Low / Medium / High risk policies."""
    ensure_output_dirs()
    save_path = save_path or (PLOTS_DIR / "risk_distribution.png")

    plt.figure(figsize=(8, 5))
    order = ["Low", "Medium", "High"]
    sns.countplot(
        data=df,
        x="risk_category",
        order=order,
        hue="risk_category",
        hue_order=order,
        palette=["#6db784", "#f4c95d", "#d95d39"],
        legend=False,
    )
    plt.title("Risk Category Distribution")
    plt.xlabel("Risk Category")
    plt.ylabel("Policy Count")
    plt.tight_layout()
    output_path = _save_current_figure(save_path)
    plt.close()
    return output_path


def save_top_premiums(df: pd.DataFrame, filename: str = "top_premiums.csv", top_n: int = 1000) -> Path:
    """Persist the highest-premium policies for review."""
    ensure_output_dirs()
    report_columns = [
        "IDpol",
        "Exposure",
        "predicted_annual_frequency",
        "predicted_claim_count",
        "predicted_claim_severity",
        "annualized_expected_loss",
        "expected_loss",
        "pure_premium",
        "final_premium",
        "risk_score",
        "risk_category",
    ]
    report_df = df[report_columns].sort_values(by="final_premium", ascending=False).head(top_n)
    output_path = REPORTS_DIR / filename
    return _save_dataframe(report_df, output_path)
