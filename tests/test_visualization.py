from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.visualization import (
    generate_model_comparison_visualizations,
    plot_error_distribution,
    plot_model_comparison_rmse,
    plot_predicted_vs_actual,
)


def test_plot_model_comparison_rmse(tmp_path: Path) -> None:
    comparison_payload = {
        "poisson_gamma": {
            "status": "success",
            "frequency_model": "poisson",
            "severity_model": "gamma",
            "metrics": {
                "frequency": {"metrics": {"rmse": 0.24}},
                "severity": {"metrics": {"rmse": 9554.78}},
            },
        },
        "rf_rf": {
            "status": "success",
            "frequency_model": "random_forest",
            "severity_model": "random_forest",
            "metrics": {
                "frequency": {"metrics": {"rmse": 0.72}},
                "severity": {"metrics": {"rmse": 70816.61}},
            },
        },
        "xgb_gamma": {
            "status": "skipped",
            "frequency_model": "xgboost",
            "severity_model": "gamma",
            "metrics": {},
        },
    }

    output_path = tmp_path / "model_comparison_rmse.png"
    saved_path = plot_model_comparison_rmse(comparison_payload, save_path=output_path)

    assert saved_path == output_path
    assert output_path.exists()


def test_prediction_diagnostics_generate_expected_outputs(tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        {
            "ClaimAmount": [100.0, 250.0, 400.0, 800.0],
            "predicted_claim_severity": [90.0, 260.0, 350.0, 820.0],
        }
    )

    scatter_path = tmp_path / "predicted_vs_actual.png"
    error_path = tmp_path / "error_distribution.png"

    saved_scatter = plot_predicted_vs_actual(
        predictions,
        save_path=scatter_path,
        actual_column="ClaimAmount",
        predicted_column="predicted_claim_severity",
    )
    saved_error = plot_error_distribution(
        predictions,
        save_path=error_path,
        actual_column="ClaimAmount",
        predicted_column="predicted_claim_severity",
    )

    assert saved_scatter == scatter_path
    assert saved_error == error_path
    assert scatter_path.exists()
    assert error_path.exists()


def test_generate_model_comparison_visualizations(tmp_path: Path) -> None:
    comparison_file = tmp_path / "model_comparison.json"
    comparison_file.write_text(
        json.dumps(
            {
                "poisson_gamma": {
                    "status": "success",
                    "frequency_model": "poisson",
                    "severity_model": "gamma",
                    "metrics": {
                        "frequency": {"metrics": {"rmse": 0.24}},
                        "severity": {"metrics": {"rmse": 9554.78}},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    predictions = pd.DataFrame(
        {
            "actual": [1.0, 2.0, 3.0],
            "predicted": [1.1, 1.8, 3.2],
        }
    )

    output_dir = tmp_path / "plots"
    outputs = generate_model_comparison_visualizations(comparison_file, predictions, output_dir=output_dir)

    assert set(outputs) == {"model_comparison_rmse", "predicted_vs_actual", "error_distribution"}
    for saved_path in outputs.values():
        assert saved_path.exists()
        assert saved_path.parent == output_dir
