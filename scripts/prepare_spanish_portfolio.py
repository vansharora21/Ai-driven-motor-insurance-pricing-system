"""Prepare the Spanish motor insurance portfolio (105,555 rows) for B2B ingestion.

Source: "Dataset of an actual motor vehicle insurance portfolio" (Mendeley Data,
DOI 10.17632/5cxyb5fp4f.2) — a real Spanish non-life motor portfolio spanning
Nov 2015 - Dec 2018. 30 columns including real premiums and claim costs.

Mapping to the canonical freMTPL2 feature schema:
    ID                  -> IDpol
    (policy-year row)   -> Exposure = 1.0
    Power (kW)          -> VehPower  (0s replaced with median, then min-max
                                      rescaled to freMTPL2 fiscal-CV range [4,20])
    Year_matriculation  -> VehAge    = 2018 - year
    Date_birth          -> DrivAge   = contract year - birth year
    N_claims_history    -> BonusMalus = 50 + 9*history, clipped to [50, 230]
    (not available)     -> VehBrand=B12, Density=393, Region=Centre
    Type_fuel           -> VehGas    (P -> Regular, D -> Diesel)
    Area (0/1)          -> Area      (A/B)

Outcomes:
    claim_occurred = N_claims_year > 0
    claim_amount   = Cost_claims_year / N_claims_year  (avg per-claim, EUR —
                       model-space currency; pricing converts to INR)
    premium_paid   = Premium (EUR)  (real earned premium — calibration gold)

Output: data/spanish_motor_b2b.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "spanish_motor_raw.csv"
OUT_PATH = PROJECT_ROOT / "data" / "spanish_motor_b2b.csv"

VEHPOWER_MIN, VEHPOWER_MAX = 4.0, 20.0
BONUSMALUS_MIN, BONUSMALUS_MAX = 50.0, 230.0
REFERENCE_YEAR = 2018  # last year of the portfolio window

FUEL_MAP = {"P": "Regular", "D": "Diesel"}
AREA_MAP = {0: "A", 1: "B"}


def _parse_year(value) -> int | None:
    """Parse DD/MM/YYYY into a year, tolerating NaN/malformed values."""
    if pd.isna(value):
        return None
    try:
        return int(str(value).split("/")[-1])
    except (ValueError, IndexError):
        return None


def prepare() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, sep=";")
    df = df[df["Premium"].notna()].copy()

    out = pd.DataFrame()
    out["IDpol"] = df["ID"]
    out["Exposure"] = 1.0

    # Engine power (kW) -> freMTPL2 fiscal CV range via min-max rescale.
    power = df["Power"].astype(float)
    power = power.replace(0.0, power[power > 0].median())
    power_min, power_max = float(power.min()), float(power.max())
    out["VehPower"] = VEHPOWER_MIN + (power - power_min) * (
        VEHPOWER_MAX - VEHPOWER_MIN
    ) / (power_max - power_min)

    out["VehAge"] = (REFERENCE_YEAR - df["Year_matriculation"]).clip(lower=0)

    contract_year = df["Date_start_contract"].map(_parse_year)
    birth_year = df["Date_birth"].map(_parse_year)
    driv_age = contract_year - birth_year
    out["DrivAge"] = driv_age.fillna(40.0).clip(lower=18, upper=100)

    # Bonus-malus approximated from claim history (higher history -> worse level).
    history = df["N_claims_history"].fillna(0).astype(float)
    out["BonusMalus"] = (BONUSMALUS_MIN + 9.0 * history).clip(
        lower=BONUSMALUS_MIN, upper=BONUSMALUS_MAX
    )

    out["VehBrand"] = "B12"
    out["VehGas"] = df["Type_fuel"].map(FUEL_MAP).fillna("Regular")
    out["Area"] = df["Area"].map(AREA_MAP).fillna("A")
    out["Density"] = 393.0
    out["Region"] = "Centre"

    # Outcomes.
    n_claims = df["N_claims_year"].fillna(0).astype(int)
    out["claim_occurred"] = n_claims > 0
    out["claim_amount"] = df["Cost_claims_year"].fillna(0.0) / n_claims.clip(lower=1)
    out["premium_paid"] = df["Premium"].astype(float)

    return out


def main() -> None:
    prepared = prepare()
    prepared.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(prepared)} rows to {OUT_PATH}")
    print(f"  policies with claims: {int(prepared['claim_occurred'].sum())}")
    print(f"  severity rows (claim_amount>0): {int((prepared['claim_amount'] > 0).sum())}")
    print(f"  avg per-claim severity (EUR): {prepared.loc[prepared['claim_amount']>0, 'claim_amount'].mean():.2f}")
    print(f"  avg premium_paid (EUR): {prepared['premium_paid'].mean():.2f}")
    print(f"  VehPower range: {prepared['VehPower'].min():.2f}-{prepared['VehPower'].max():.2f}")
    print(f"  DrivAge range: {prepared['DrivAge'].min():.0f}-{prepared['DrivAge'].max():.0f}")
    print(f"  VehGas distribution: {prepared['VehGas'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()