from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import get_random_seed

OPTIONAL_ANOMALY_FEATURES = [
    "predicted_claim_count",
    "predicted_claim_severity",
    "final_premium",
    "BonusMalus",
    "VehAge",
]


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.01) -> tuple[pd.DataFrame, IsolationForest]:
    """
    Optional helper for post-pricing anomaly detection.

    This module is intentionally separate from the production training pipeline.
    It does not inject fake fraud. Instead, it can be run on an already scored
    portfolio to highlight policies whose premium drivers look unusual.
    """
    missing_columns = [column for column in OPTIONAL_ANOMALY_FEATURES if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing anomaly-detection columns: {missing_columns}")

    scored = df.copy()
    model = IsolationForest(contamination=contamination, random_state=get_random_seed())
    predictions = model.fit_predict(scored[OPTIONAL_ANOMALY_FEATURES].fillna(0))
    scored["anomaly_flag"] = np.where(predictions == -1, 1, 0)
    return scored, model


if __name__ == "__main__":
    from predict import predict_premiums
    from src.data_loader import prepare_model_datasets

    policy_df, _, _ = prepare_model_datasets()
    scored_portfolio = predict_premiums(policy_df.head(5000))
    anomalies, _ = detect_anomalies(scored_portfolio)
    print(anomalies[["IDpol", "final_premium", "anomaly_flag"]].head())
