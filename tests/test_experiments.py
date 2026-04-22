from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.experiments import run_model_comparison_experiments
from src.feature_engineering import engineer_features


def test_run_model_comparison_generates_expected_keys(
    sample_policy_df: pd.DataFrame,
    sample_claim_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    frequency_df = engineer_features(sample_policy_df)
    severity_df = engineer_features(sample_claim_df)

    output_path = tmp_path / "model_comparison.json"
    saved_path = run_model_comparison_experiments(
        frequency_train=frequency_df,
        frequency_test=frequency_df,
        severity_train=severity_df,
        severity_test=severity_df,
        output_path=output_path,
    )

    assert saved_path == output_path
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "poisson_gamma" in payload
    assert "rf_rf" in payload
    assert "xgb_xgb" in payload

    for result in payload.values():
        assert "status" in result
        assert "frequency_model" in result
        assert "severity_model" in result
