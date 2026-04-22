from __future__ import annotations

import json
from copy import deepcopy
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from src.config import get_experiment_config, get_model_config, get_pricing_config, get_random_seed
from src.feature_engineering import POLICY_ID_COLUMN
from src.frequency_model import evaluate_frequency_model, predict_frequency, train_frequency_model
from src.model_artifacts import EVALUATION_DIR, PLOTS_DIR, REPORTS_DIR
from src.pricing_engine import calculate_premium
from src.severity_model import evaluate_severity_model, predict_severity, train_severity_model
from src.visualization import plot_feature_importance

FREQUENCY_EXPERIMENT_MODELS = ["poisson", "random_forest", "xgboost"]
SEVERITY_EXPERIMENT_MODELS = ["gamma", "random_forest", "xgboost"]
MODEL_COMPARISON_PATH = EVALUATION_DIR / "model_comparison.json"
MODEL_PREMIUM_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
MODEL_FAMILY_COLUMNS = {
    "glm": ("poisson", "gamma"),
    "rf": ("random_forest", "random_forest"),
    "xgb": ("xgboost", "xgboost"),
}


def _abbreviate_model_name(model_name: str) -> str:
    aliases = {
        "random_forest": "rf",
        "xgboost": "xgb",
    }
    return aliases.get(model_name, model_name)


def _experiment_key(frequency_model_name: str, severity_model_name: str) -> str:
    return f"{_abbreviate_model_name(frequency_model_name)}_{_abbreviate_model_name(severity_model_name)}"


def _resolve_model_list(raw_values: object, fallback: list[str]) -> list[str]:
    if not isinstance(raw_values, list):
        return fallback
    resolved = [str(value).strip().lower() for value in raw_values if str(value).strip()]
    return resolved or fallback


def _sample_for_experiment(
    df: pd.DataFrame,
    sample_fraction: float,
    max_rows: int,
    random_seed: int,
) -> pd.DataFrame:
    target_rows = len(df)
    if 0.0 < sample_fraction < 1.0:
        target_rows = min(target_rows, max(1, int(len(df) * sample_fraction)))
    if max_rows > 0:
        target_rows = min(target_rows, max_rows)
    if target_rows >= len(df):
        return df
    return df.sample(n=target_rows, random_state=random_seed)


def _build_experiment_model_config(experiment_config: dict[str, Any]) -> dict[str, Any]:
    model_config = deepcopy(get_model_config())
    rf_estimators = int(experiment_config.get("random_forest_n_estimators", 0))
    xgb_estimators = int(experiment_config.get("xgboost_n_estimators", 0))

    if rf_estimators > 0:
        model_config.setdefault("frequency", {}).setdefault("random_forest", {})["n_estimators"] = rf_estimators
        model_config.setdefault("severity", {}).setdefault("random_forest", {})["n_estimators"] = rf_estimators

    if xgb_estimators > 0:
        model_config.setdefault("frequency", {}).setdefault("xgboost", {})["n_estimators"] = xgb_estimators
        model_config.setdefault("severity", {}).setdefault("xgboost", {})["n_estimators"] = xgb_estimators

    return model_config


def run_model_comparison_experiments(
    frequency_train: pd.DataFrame,
    frequency_test: pd.DataFrame,
    severity_train: pd.DataFrame,
    severity_test: pd.DataFrame,
    output_path: str | Path = MODEL_COMPARISON_PATH,
    comparison_output_path: str | Path = MODEL_PREMIUM_COMPARISON_PATH,
    scoring_df: pd.DataFrame | None = None,
) -> Path:
    """
    Compare all configured frequency/severity model-family combinations.

    Each run stores the selected model names, metrics, runtime, and status.
    XGBoost combinations are skipped with a clear message when xgboost is not installed.
    """
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_comparison_output_path = Path(comparison_output_path)
    resolved_comparison_output_path.parent.mkdir(parents=True, exist_ok=True)

    experiment_config = get_experiment_config()
    frequency_models = _resolve_model_list(experiment_config.get("frequency_models"), FREQUENCY_EXPERIMENT_MODELS)
    severity_models = _resolve_model_list(experiment_config.get("severity_models"), SEVERITY_EXPERIMENT_MODELS)
    sample_fraction = float(experiment_config.get("sample_fraction", 1.0))
    max_frequency_rows = int(experiment_config.get("max_frequency_rows", 0))
    max_severity_rows = int(experiment_config.get("max_severity_rows", 0))
    random_seed = get_random_seed()
    experiment_model_config = _build_experiment_model_config(experiment_config)

    experiment_frequency_train = _sample_for_experiment(
        frequency_train,
        sample_fraction=sample_fraction,
        max_rows=max_frequency_rows,
        random_seed=random_seed,
    )
    experiment_severity_train = _sample_for_experiment(
        severity_train,
        sample_fraction=sample_fraction,
        max_rows=max_severity_rows,
        random_seed=random_seed,
    )

    print(
        "Model comparison sample sizes: "
        f"frequency_train={len(experiment_frequency_train)}/{len(frequency_train)}, "
        f"severity_train={len(experiment_severity_train)}/{len(severity_train)}"
    )

    scoring_frame = scoring_df.copy() if scoring_df is not None else frequency_test.copy()
    if POLICY_ID_COLUMN in scoring_frame.columns:
        policy_ids = pd.to_numeric(scoring_frame[POLICY_ID_COLUMN], errors="coerce").fillna(0).round().astype(int)
    else:
        policy_ids = pd.Series(range(1, len(scoring_frame) + 1), index=scoring_frame.index, dtype=int)

    premium_comparison_df = pd.DataFrame({"policy_id": policy_ids}, index=scoring_frame.index)
    for model_family in MODEL_FAMILY_COLUMNS:
        premium_comparison_df[f"{model_family}_predicted_frequency"] = pd.NA
        premium_comparison_df[f"{model_family}_predicted_severity"] = pd.NA
        premium_comparison_df[f"{model_family}_premium"] = pd.NA

    pricing_config = get_pricing_config()

    def _resolve_model_family(frequency_model_name: str, severity_model_name: str) -> str | None:
        for family, combo in MODEL_FAMILY_COLUMNS.items():
            if combo == (frequency_model_name, severity_model_name):
                return family
        return None

    comparison_results: dict[str, dict[str, object]] = {}
    total_runs = len(frequency_models) * len(severity_models)
    generated_feature_importance_labels: set[str] = set()

    def _save_feature_importance_plot(model: object, role: str, model_name: str) -> None:
        feature_importance_label = f"{role}_{model_name}"
        if feature_importance_label in generated_feature_importance_labels:
            return

        saved_path = plot_feature_importance(
            model,
            save_path=PLOTS_DIR / f"feature_importance_{feature_importance_label}.png",
            model_label=feature_importance_label,
        )
        if saved_path is not None:
            generated_feature_importance_labels.add(feature_importance_label)
            print(f"Saved feature importance plot to {saved_path.resolve()}")

    for run_index, (frequency_model_name, severity_model_name) in enumerate(
        product(frequency_models, severity_models),
        start=1,
    ):
        run_label = f"[{run_index}/{total_runs}] {frequency_model_name} + {severity_model_name}"
        print(f"Model comparison run {run_label}: training...")
        started_at = perf_counter()
        key = _experiment_key(frequency_model_name, severity_model_name)

        try:
            frequency_model = train_frequency_model(
                experiment_frequency_train,
                model_name=frequency_model_name,
                model_config=experiment_model_config,
            )
            severity_model = train_severity_model(
                experiment_severity_train,
                model_name=severity_model_name,
                model_config=experiment_model_config,
            )

            frequency_metrics = evaluate_frequency_model(frequency_model, frequency_test)
            severity_metrics = evaluate_severity_model(severity_model, severity_test)
            _save_feature_importance_plot(frequency_model, "frequency", frequency_model_name)
            _save_feature_importance_plot(severity_model, "severity", severity_model_name)

            model_family = _resolve_model_family(frequency_model_name, severity_model_name)
            if model_family is not None:
                predicted_frequency = predict_frequency(frequency_model, scoring_frame)
                predicted_severity = predict_severity(severity_model, scoring_frame)
                scored = calculate_premium(
                    scoring_frame,
                    predicted_frequency,
                    predicted_severity,
                    pricing_config=pricing_config,
                )
                premium_comparison_df[f"{model_family}_predicted_frequency"] = scored["predicted_annual_frequency"].astype(float)
                premium_comparison_df[f"{model_family}_predicted_severity"] = scored["predicted_claim_severity"].astype(float)
                premium_comparison_df[f"{model_family}_premium"] = scored["final_premium"].astype(float)

            elapsed_seconds = float(perf_counter() - started_at)

            comparison_results[key] = {
                "status": "success",
                "frequency_model": frequency_model_name,
                "severity_model": severity_model_name,
                "duration_seconds": elapsed_seconds,
                "metrics": {
                    "frequency": frequency_metrics,
                    "severity": severity_metrics,
                },
            }
            print(f"Model comparison run {run_label}: success in {elapsed_seconds:.2f}s")
        except ImportError as exc:
            elapsed_seconds = float(perf_counter() - started_at)
            comparison_results[key] = {
                "status": "skipped",
                "frequency_model": frequency_model_name,
                "severity_model": severity_model_name,
                "duration_seconds": elapsed_seconds,
                "error": str(exc),
            }
            print(f"Model comparison run {run_label}: skipped ({exc})")
        except Exception as exc:  # pragma: no cover - defensive logging path
            elapsed_seconds = float(perf_counter() - started_at)
            comparison_results[key] = {
                "status": "failed",
                "frequency_model": frequency_model_name,
                "severity_model": severity_model_name,
                "duration_seconds": elapsed_seconds,
                "error": str(exc),
            }
            print(f"Model comparison run {run_label}: failed ({exc})")

    with resolved_output_path.open("w", encoding="utf-8") as comparison_file:
        json.dump(comparison_results, comparison_file, indent=2)

    ordered_columns = ["policy_id"]
    for family in MODEL_FAMILY_COLUMNS:
        ordered_columns.extend(
            [
                f"{family}_predicted_frequency",
                f"{family}_predicted_severity",
                f"{family}_premium",
            ]
        )

    premium_comparison_df = premium_comparison_df[ordered_columns]
    premium_comparison_df.to_csv(resolved_comparison_output_path, index=False, float_format="%.6f")
    print(f"Saved premium model comparison report to {resolved_comparison_output_path.resolve()}")

    return resolved_output_path
