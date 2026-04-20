from __future__ import annotations

from pathlib import Path

import pandas as pd

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


def export_batch_template(
    input_file: str | Path = "data/freMTPL2freq.csv",
    output_file: str | Path = "data/freMTPL2_batch_template.csv",
    num_rows: int = 1000,
) -> pd.DataFrame:
    """
    Export a clean feature-only batch template from the real frequency dataset.
    """
    source_df = pd.read_csv(input_file)
    batch_df = source_df[FEATURE_COLUMNS].head(num_rows).copy()
    batch_df.to_csv(output_file, index=False)
    print(f"Exported {len(batch_df)} rows to {output_file}")
    return batch_df


if __name__ == "__main__":
    export_batch_template()
