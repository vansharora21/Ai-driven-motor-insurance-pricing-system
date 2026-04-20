from __future__ import annotations

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


def generate_mock_data(num_samples: int = 250) -> pd.DataFrame:
    """
    Generate a realistic batch-inference sample by drawing rows from freMTPL2.
    """
    source_df = pd.read_csv("data/freMTPL2freq.csv")
    sample_df = source_df[FEATURE_COLUMNS].sample(
        n=min(num_samples, len(source_df)),
        random_state=get_random_seed(),
    ).reset_index(drop=True)
    sample_df.to_csv("data/sample_batch.csv", index=False)
    print(f"Generated {len(sample_df)} sample policies in data/sample_batch.csv")
    return sample_df


if __name__ == "__main__":
    generate_mock_data()
