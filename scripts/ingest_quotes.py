"""Backfill quotes / B2B portfolios into Supabase from CSV files.

Use cases:
    1. Import a CSV of previously scored predictions into `quotes`.
    2. Import an insurer portfolio CSV (with real outcomes) into `b2b_portfolios`.

The API already auto-saves every /predict call; this script exists for
backfilling historical data or bulk-loading B2B portfolios.

Examples:
    python scripts/ingest_quotes.py --csv predictions.csv --table quotes
    python scripts/ingest_quotes.py --csv portfolio.csv --table b2b_portfolios --name "Acme Insurer Q3"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.db import POLICY_COLUMNS, RESULT_COLUMNS, save_b2b_portfolio, save_quote


def _row_to_policy(row: pd.Series) -> dict:
    return {column: row.get(column) for column in POLICY_COLUMNS}


def _row_to_result(row: pd.Series) -> dict:
    return {column: row.get(column) for column in RESULT_COLUMNS}


def ingest_quotes_csv(csv_path: Path) -> int:
    """Insert every row of a scored-predictions CSV into `quotes`."""
    df = pd.read_csv(csv_path)
    saved = 0
    for _, row in df.iterrows():
        if save_quote(_row_to_policy(row), _row_to_result(row), source="backfill"):
            saved += 1
    return saved


def ingest_b2b_csv(csv_path: Path, portfolio_name: str) -> int:
    """Insert an insurer portfolio CSV into `b2b_portfolios`."""
    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        record = {column: row.get(column) for column in POLICY_COLUMNS}
        record["claim_occurred"] = bool(row.get("claim_occurred", False))
        record["claim_amount"] = row.get("claim_amount")
        record["premium_paid"] = row.get("premium_paid")
        rows.append(record)
    return save_b2b_portfolio(rows, portfolio_name=portfolio_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill data into Supabase.")
    parser.add_argument("--csv", required=True, help="Path to the CSV file.")
    parser.add_argument(
        "--table",
        choices=["quotes", "b2b_portfolios"],
        default="quotes",
        help="Which table to insert into.",
    )
    parser.add_argument("--name", default="untitled", help="Portfolio name (b2b_portfolios only).")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    if args.table == "quotes":
        saved = ingest_quotes_csv(csv_path)
        print(f"Saved {saved} rows to quotes.")
    else:
        saved = ingest_b2b_csv(csv_path, args.name)
        print(f"Saved {saved} rows to b2b_portfolios (portfolio={args.name!r}).")

    if saved == 0:
        print("WARNING: 0 rows saved. Check SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY env vars.")
        sys.exit(1)


if __name__ == "__main__":
    main()