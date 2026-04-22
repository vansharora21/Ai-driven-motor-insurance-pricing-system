from __future__ import annotations

from pathlib import Path

from src.feature_engineering import engineer_features
from src.frequency_model import train_frequency_model
from src.visualization import extract_feature_importance, plot_feature_importance


def test_tree_model_feature_importance_is_extracted_and_plotted(sample_policy_df, tmp_path: Path) -> None:
    feature_df = engineer_features(sample_policy_df)
    model = train_frequency_model(
        feature_df,
        model_name="random_forest",
        model_config={
            "frequency_model": "random_forest",
            "frequency": {"random_forest": {"n_estimators": 10, "n_jobs": 1}},
        },
    )

    importance_df = extract_feature_importance(model)
    assert importance_df is not None
    assert not importance_df.empty
    assert {"feature", "importance"}.issubset(importance_df.columns)
    assert importance_df["importance"].iloc[0] > 0

    save_path = tmp_path / "feature_importance_random_forest.png"
    output_path = plot_feature_importance(model, save_path=save_path, model_label="random_forest")

    assert output_path == save_path
    assert save_path.exists()


def test_glm_model_feature_importance_is_skipped(sample_policy_df, tmp_path: Path) -> None:
    feature_df = engineer_features(sample_policy_df)
    model = train_frequency_model(feature_df, model_name="poisson")

    assert extract_feature_importance(model) is None
    assert plot_feature_importance(
        model,
        save_path=tmp_path / "feature_importance_poisson.png",
        model_label="poisson",
    ) is None