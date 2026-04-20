from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import get_random_seed

FEATURE_COLUMNS = [
    "IDpol",
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


def generate_synthetic_data(
    real_data_path: str | Path = "data/freMTPL2freq.csv",
    output_path: str | Path = "data/freMTPL2_bootstrap_sample.csv",
    num_rows: int = 1000,
) -> pd.DataFrame:
    """
    Optional bootstrap sampler for experimentation.

    This is not part of the production training path. It simply resamples the
    real frequency dataset to produce a feature-only batch file for demos.
    """
    real_data = pd.read_csv(real_data_path)
    sampled = real_data[FEATURE_COLUMNS].sample(
        n=num_rows,
        replace=True,
        random_state=get_random_seed(),
    ).reset_index(drop=True)
    sampled.to_csv(output_path, index=False)
    print(f"Saved {num_rows} bootstrap-sampled policies to {output_path}")
    return sampled


if __name__ == "__main__":
    generate_synthetic_data()
