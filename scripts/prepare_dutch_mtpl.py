"""Prepare the Dutch MTPL portfolio (30k policies) for B2B ingestion.

Source: insurancerating R package (CRAN), MTPL dataset — a real Dutch motor
third-party liability portfolio. Columns: age_policyholder, nclaims, exposure,
amount (EUR), power (kW), bm (bonus-malus level 1-23), zip (region 0-3).

Mapping to the canonical freMTPL2 feature schema:
    exposure            -> Exposure            (direct)
    power (kW)          -> VehPower            (min-max rescaled to freMTPL2
                                                fiscal-CV range [4, 20])
    age_policyholder    -> DrivAge             (direct)
    bm (1-23)           -> BonusMalus          (linear to freMTPL2 50-230)
    zip (0-3)           -> Area                (A/B/C/D)
    (not available)     -> VehAge=6, VehBrand=B12, VehGas=Regular,
                           Density=393, Region=Centre  (freMTPL2 medians/modes)

Outcomes:
    claim_occurred = nclaims > 0
    claim_amount   = amount / nclaims   (average per-claim severity, EUR —
                                         model-space currency; pricing converts
                                         to INR at predict time)
    premium_paid   = 0                  (not present in the Dutch MTPL)

Output: data/dutch_mtpl_b2b.csv (ready for scripts/ingest_quotes.py --table b2b_portfolios)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "dutch_mtpl_raw.csv"
OUT_PATH = PROJECT_ROOT / "data" / "dutch_mtpl_b2b.csv"

# freMTPL2 fiscal-horsepower range (VehPower clip bounds in config).
VEHPOWER_MIN, VEHPOWER_MAX = 4.0, 20.0
# freMTPL2 BonusMalus percentage range.
BONUSMALUS_MIN, BONUSMALUS_MAX = 50.0, 230.0

ZIP_TO_AREA = {"0": "A", "1": "B", "2": "C", "3": "D"}


def prepare() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    df = df[df["exposure"] > 0].copy()

    out = pd.DataFrame()
    out["Exposure"] = df["exposure"]

    # Engine power (kW) -> freMTPL2 fiscal CV range via min-max rescale.
    power_min, power_max = float(df["power"].min()), float(df["power"].max())
    out["VehPower"] = VEHPOWER_MIN + (df["power"] - power_min) * (
        VEHPOWER_MAX - VEHPOWER_MIN
    ) / (power_max - power_min)

    out["VehAge"] = 6.0  # not in Dutch data; freMTPL2 median
    out["DrivAge"] = df["age_policyholder"]

    # Bonus-malus level (1-23) -> freMTPL2 percentage (50-230).
    bm_min, bm_max = float(df["bm"].min()), float(df["bm"].max())
    out["BonusMalus"] = BONUSMALUS_MIN + (df["bm"] - bm_min) * (
        BONUSMALUS_MAX - BONUSMALUS_MIN
    ) / (bm_max - bm_min)

    out["VehBrand"] = "B12"
    out["VehGas"] = "Regular"
    out["Area"] = df["zip"].astype(str).map(ZIP_TO_AREA)
    out["Density"] = 393.0
    out["Region"] = "Centre"

    # Outcomes.
    out["claim_occurred"] = (df["nclaims"] > 0)
    out["claim_amount"] = df["amount"] / df["nclaims"].clip(lower=1)
    out["premium_paid"] = 0.0

    return out


def main() -> None:
    prepared = prepare()
    prepared.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(prepared)} rows to {OUT_PATH}")
    print(f"  policies with claims: {int(prepared['claim_occurred'].sum())}")
    print(f"  severity rows (claim_amount>0): {int((prepared['claim_amount'] > 0).sum())}")
    print(f"  VehPower range: {prepared['VehPower'].min():.2f}-{prepared['VehPower'].max():.2f}")
    print(f"  BonusMalus range: {prepared['BonusMalus'].min():.2f}-{prepared['BonusMalus'].max():.2f}")
    print(f"  Area distribution: {prepared['Area'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()