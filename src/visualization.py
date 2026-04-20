from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import get_random_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
REPORTS_DIR = PROJECT_ROOT / "results" / "premium_reports"
EVALUATION_DIR = PROJECT_ROOT / "results" / "evaluation"


def ensure_output_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


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
    ensure_output_dirs()
    save_path = save_path or (PLOTS_DIR / "severity_actual_vs_pred.png")

    plot_df = pd.DataFrame({"actual": actual, "predicted": predicted})
    if len(plot_df) > 5000:
        plot_df = plot_df.sample(5000, random_state=get_random_seed())

    axis_limit = max(plot_df["actual"].max(), plot_df["predicted"].max())
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=plot_df, x="actual", y="predicted", alpha=0.35, s=25)
    plt.plot([0, axis_limit], [0, axis_limit], linestyle="--", color="black")
    plt.title("Severity Model: Actual vs Predicted Claim Amount")
    plt.xlabel("Actual Claim Amount")
    plt.ylabel("Predicted Claim Amount")
    plt.tight_layout()
    output_path = _save_current_figure(save_path)
    plt.close()
    return output_path


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
