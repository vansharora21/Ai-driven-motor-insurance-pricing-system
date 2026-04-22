from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    get_experiment_config,
    get_evaluation_config,
    get_pricing_config,
    get_random_seed,
    load_modeling_config,
    set_global_determinism,
)
from src.data_loader import prepare_model_datasets
from src.experiments import run_model_comparison_experiments
from src.experiment_tracking import persist_experiment_run
from src.feature_engineering import (
    CATEGORICAL_FEATURES,
    EXPOSURE_COLUMN,
    INPUT_COLUMNS,
    MODEL_FEATURES,
    RAW_NUMERIC_FEATURES,
    build_default_metadata,
)
from src.frequency_model import evaluate_frequency_model, predict_claim_count, predict_frequency, train_frequency_model
from src.model_artifacts import save_artifacts
from src.pricing_engine import calculate_premium, compute_portfolio_baselines, compute_risk_thresholds
from src.severity_model import evaluate_severity_model, predict_severity, train_severity_model
from src.visualization import (
    generate_model_comparison_visualizations,
    plot_frequency_calibration,
    plot_predicted_vs_actual,
    plot_premium_distribution,
    plot_risk_distribution,
    plot_error_distribution,
    plot_severity_predictions,
    save_top_premiums,
)


def _build_metadata(
    policy_df: pd.DataFrame,
    severity_df: pd.DataFrame,
    data_quality: dict[str, int],
    pricing_config: dict[str, object],
    risk_thresholds: dict[str, float],
    portfolio_baselines: dict[str, float],
    modeling_config: dict[str, object],
    random_seed: int,
) -> dict[str, object]:
    """Build metadata bundle required for reproducible inference and UI defaults."""
    defaults = build_default_metadata(policy_df)

    return {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "input_columns": INPUT_COLUMNS,
        "model_features": MODEL_FEATURES,
        "numeric_defaults": defaults["numeric_defaults"],
        "categorical_defaults": defaults["categorical_defaults"],
        "supported_input_aliases": defaults["supported_input_aliases"],
        "optional_simulated_features": defaults["optional_simulated_features"],
        "categorical_options": {
            feature: sorted(policy_df[feature].dropna().astype(str).unique().tolist())
            for feature in CATEGORICAL_FEATURES
        },
        "numeric_ranges": {
            column: {
                "min": float(policy_df[column].min()),
                "max": float(policy_df[column].max()),
            }
            for column in [EXPOSURE_COLUMN] + RAW_NUMERIC_FEATURES
        },
        "risk_thresholds": risk_thresholds,
        "portfolio_baselines": portfolio_baselines,
        "pricing_config": pricing_config,
        "modeling_config": modeling_config,
        "dataset_summary": {
            "policy_rows": int(len(policy_df)),
            "claim_rows": int(len(severity_df)),
        },
        "data_quality": data_quality,
    }


def main() -> None:
    """Train, evaluate, persist artifacts, and emit experiment-tracking outputs."""
    modeling_config = load_modeling_config()
    random_seed = set_global_determinism(get_random_seed(modeling_config))
    evaluation_config = get_evaluation_config(modeling_config)
    experiment_config = get_experiment_config(modeling_config)
    pricing_config = get_pricing_config(modeling_config)

    print("Loading and preparing freMTPL2 datasets...")
    policy_df, severity_df, data_quality = prepare_model_datasets()

    print("Splitting frequency and severity training sets...")
    # Frequency model uses policy-level claim counts, stratified by claim occurrence.
    frequency_train, frequency_test = train_test_split(
        policy_df,
        test_size=float(evaluation_config["test_size"]),
        random_state=random_seed,
        stratify=policy_df["has_claim"],
    )
    # Severity model uses claim-level paid amounts and does not require stratification.
    severity_train, severity_test = train_test_split(
        severity_df,
        test_size=float(evaluation_config["test_size"]),
        random_state=random_seed,
    )

    print("Training evaluation models...")
    frequency_eval_model = train_frequency_model(frequency_train)
    severity_eval_model = train_severity_model(severity_train)

    print("Evaluating holdout performance...")
    frequency_metrics = evaluate_frequency_model(frequency_eval_model, frequency_test)
    severity_metrics = evaluate_severity_model(severity_eval_model, severity_test)

    comparison_path: Path | None = None
    if bool(experiment_config.get("enabled", True)):
        print("Running model comparison experiments...")
        comparison_path = run_model_comparison_experiments(
            frequency_train=frequency_train,
            frequency_test=frequency_test,
            severity_train=severity_train,
            severity_test=severity_test,
        )
    else:
        print("Model comparison experiments disabled in config.")

    test_predicted_claim_count = predict_claim_count(frequency_eval_model, frequency_test)
    test_predicted_severity = predict_severity(severity_eval_model, severity_test)

    plot_frequency_calibration(frequency_test["ClaimNb"], test_predicted_claim_count)
    plot_severity_predictions(severity_test["ClaimAmount"], test_predicted_severity)

    comparison_predictions = pd.DataFrame(
        {
            "ClaimAmount": severity_test["ClaimAmount"],
            "predicted_claim_severity": test_predicted_severity,
        }
    )
    if comparison_path is not None:
        comparison_visualizations = generate_model_comparison_visualizations(
            comparison_path,
            comparison_predictions,
            actual_column="ClaimAmount",
            predicted_column="predicted_claim_severity",
        )
        print(
            "Saved comparison plots to "
            + ", ".join(str(path.resolve()) for path in comparison_visualizations.values())
        )
    else:
        plot_predicted_vs_actual(
            comparison_predictions,
            save_path=Path("results/plots/predicted_vs_actual.png"),
            actual_column="ClaimAmount",
            predicted_column="predicted_claim_severity",
        )
        plot_error_distribution(
            comparison_predictions,
            save_path=Path("results/plots/error_distribution.png"),
            actual_column="ClaimAmount",
            predicted_column="predicted_claim_severity",
        )

    print("Retraining final models on the full datasets...")
    frequency_model = train_frequency_model(policy_df)
    severity_model = train_severity_model(severity_df)

    # Portfolio baselines stabilize relativities and risk scores across prediction batches.
    portfolio_frequency = predict_frequency(frequency_model, policy_df)
    portfolio_severity = predict_severity(severity_model, policy_df)
    portfolio_baselines = compute_portfolio_baselines(
        portfolio_frequency,
        portfolio_severity,
        pricing_config=pricing_config,
    )
    risk_thresholds = compute_risk_thresholds(pricing_config=pricing_config)
    scored_portfolio = calculate_premium(
        policy_df,
        portfolio_frequency,
        portfolio_severity,
        pricing_config=pricing_config,
        risk_thresholds=risk_thresholds,
        portfolio_baselines=portfolio_baselines,
    )

    plot_premium_distribution(scored_portfolio)
    plot_risk_distribution(scored_portfolio)
    top_premiums_path = save_top_premiums(scored_portfolio)

    metrics = {
        "frequency": {
            "selected_model": str(modeling_config.get("models", {}).get("frequency_model", "poisson")),
            **frequency_metrics,
        },
        "severity": {
            "selected_model": str(modeling_config.get("models", {}).get("severity_model", "gamma")),
            **severity_metrics,
        },
    }
    metadata = _build_metadata(
        policy_df,
        severity_df,
        data_quality,
        pricing_config,
        risk_thresholds,
        portfolio_baselines,
        modeling_config,
        random_seed,
    )
    save_artifacts(frequency_model, severity_model, metadata, metrics)

    experiment_path = persist_experiment_run(
        metadata=metadata,
        metrics=metrics,
        model_config=modeling_config.get("models", {}),
        data_quality=data_quality,
    )

    print("Training complete.")
    print(f"Saved model artifacts to {Path('models').resolve()}")
    print(f"Saved evaluation metrics to {Path('results/evaluation/metrics.json').resolve()}")
    if comparison_path is not None:
        print(f"Saved model comparison metrics to {comparison_path.resolve()}")
    print(f"Saved premium report to {top_premiums_path.resolve()}")
    print(f"Saved experiment run summary to {experiment_path.resolve()}")


if __name__ == "__main__":
    main()
