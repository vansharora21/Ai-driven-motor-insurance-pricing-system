from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_policy_df() -> pd.DataFrame:
    """Small but valid policy dataframe for fast unit tests."""
    rng = np.random.default_rng(42)
    n = 30
    return pd.DataFrame(
        {
            "IDpol": np.arange(1, n + 1),
            "ClaimNb": rng.integers(0, 3, size=n),
            "Exposure": rng.uniform(0.1, 1.0, size=n),
            "VehPower": rng.integers(4, 12, size=n),
            "VehAge": rng.integers(0, 15, size=n),
            "DrivAge": rng.integers(21, 70, size=n),
            "BonusMalus": rng.integers(50, 120, size=n),
            "VehBrand": rng.choice(["B1", "B2", "B12"], size=n),
            "VehGas": rng.choice(["Regular", "Diesel"], size=n),
            "Area": rng.choice(["A", "B", "C"], size=n),
            "Density": rng.integers(20, 2000, size=n),
            "Region": rng.choice(["Centre", "Rhone-Alpes", "Bretagne"], size=n),
        }
    )


@pytest.fixture
def sample_claim_df(sample_policy_df: pd.DataFrame) -> pd.DataFrame:
    """Positive-claim rows for severity-model tests."""
    claims = sample_policy_df.loc[sample_policy_df["ClaimNb"] > 0, ["IDpol"]].copy()
    if claims.empty:
        claims = sample_policy_df.head(5)[["IDpol"]].copy()

    claims["ClaimAmount"] = np.linspace(200.0, 2500.0, len(claims))

    feature_columns = [
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
    return claims.merge(sample_policy_df[["IDpol"] + feature_columns], on="IDpol", how="left")
