"""Drift monitoring: compare model predictions against actual outcomes.

The `quotes` table stores every scored policy (predicted frequency, severity,
premium). When actual outcomes arrive (a claim occurred / a premium was paid),
we can measure how far the model drifted from reality and decide *when* to
retrain.

This script currently reports on B2B portfolios (which carry real outcomes)
and flags quotes that have been matched to outcomes. It is intentionally
read-only: it never modifies data.

Usage:
    python scripts/drift_check.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.db import fetch_quotes_for_drift, fetch_training_data


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    try:
        data = fetch_training_data()
        quotes = fetch_quotes_for_drift()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    b2b = data["b2b_portfolios"]
    print(f"Quotes stored: {len(quotes)} | B2B portfolios with outcomes: {len(b2b)}")

    if not b2b:
        print("No outcome data yet. Drift analysis needs actual claims/premiums.")
        return

    # --- Severity drift: predicted severity vs actual claim amount ---------
    rows = []
    for record in b2b:
        claim_amount = _safe_float(record.get("claim_amount"))
        if claim_amount <= 0:
            continue
        rows.append(
            {
                "actual_claim_amount": claim_amount,
                "predicted_claim_severity": _safe_float(record.get("predicted_claim_severity")),
            }
        )

    if rows:
        df = pd.DataFrame(rows)
        df["ratio"] = df["actual_claim_amount"] / df["predicted_claim_severity"].replace(0, np.nan)
        print("\n--- Severity drift (B2B actual claims vs predicted severity) ---")
        print(f"Matched claims: {len(df)}")
        print(f"Median actual/predicted ratio: {df['ratio'].median():.2f}")
        print(f"Mean  actual/predicted ratio: {df['ratio'].mean():.2f}")
        print(f"P10/P90 ratio: {df['ratio'].quantile(0.10):.2f} / {df['ratio'].quantile(0.90):.2f}")
        if df["ratio"].median() > 1.2:
            print("⚠  Severity is UNDER-predicted by >20%. Consider retraining the severity model.")
        elif df["ratio"].median() < 0.8:
            print("⚠  Severity is OVER-predicted by >20%. Consider retraining the severity model.")
        else:
            print("✓ Severity predictions are within ±20% of actuals.")
    else:
        print("\nNo B2B rows with positive claim amounts for severity drift.")

    # --- Frequency drift: predicted frequency vs observed claim rate --------
    freq_rows = []
    for record in b2b:
        exposure = _safe_float(record.get("Exposure")) or 1.0
        claim_occurred = bool(record.get("claim_occurred")) or _safe_float(record.get("claim_amount")) > 0
        freq_rows.append(
            {
                "observed_frequency": float(claim_occurred) / exposure,
                "predicted_frequency": _safe_float(record.get("predicted_annual_frequency")),
            }
        )

    if freq_rows:
        freq_df = pd.DataFrame(freq_rows)
        observed_mean = freq_df["observed_frequency"].mean()
        predicted_mean = freq_df["predicted_frequency"].mean()
        print("\n--- Frequency drift (observed vs predicted claim rate) ---")
        print(f"Observed mean claim rate:   {observed_mean:.4f} claims/yr")
        print(f"Predicted mean claim rate:  {predicted_mean:.4f} claims/yr")
        if predicted_mean > 0:
            ratio = observed_mean / predicted_mean
            print(f"Observed/predicted ratio:   {ratio:.2f}")
            if ratio > 1.2 or ratio < 0.8:
                print("⚠  Frequency drift detected. Consider retraining the frequency model.")
            else:
                print("✓ Frequency predictions are within ±20% of observed rates.")
        else:
            print("Predicted frequency is zero; cannot compute ratio.")

    print("\nNext step: when drift is persistent, run `python scripts/retrain.py`.")


if __name__ == "__main__":
    main()