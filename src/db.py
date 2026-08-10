"""Supabase (Postgres) persistence layer for the data flywheel.

Three tables, each serving a distinct purpose:

    quotes           -> every scored policy, auto-saved, anonymized.
                        Enables drift monitoring and volume tracking.
    training_data    -> only quotes where the user consented to research use.
                        The clean, consented dataset used for retraining.
    b2b_portfolios   -> insurer portfolio uploads with real outcomes
                        (claims + premiums). The ground-truth goldmine.

Design rules:
- The API must never crash because Supabase is unreachable or unconfigured.
  All write helpers are best-effort and swallow/log errors.
- Only anonymized policy attributes are stored. No emails, no PII.
- Env vars: SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("motor-pricing-db")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
# New-style Supabase projects expose SUPABASE_SECRET_KEY (server-side, full
# access). Older projects used SUPABASE_SERVICE_ROLE_KEY. Support both.
SUPABASE_SECRET_KEY = os.environ.get("SUPABASE_SECRET_KEY", "").strip() or os.environ.get(
    "SUPABASE_SERVICE_ROLE_KEY", ""
).strip()

# Column names shared by the quotes / training_data tables.
POLICY_COLUMNS = [
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

RESULT_COLUMNS = [
    "predicted_annual_frequency",
    "predicted_claim_severity",
    "expected_loss",
    "pure_premium",
    "technical_premium",
    "final_premium",
    "risk_score",
    "risk_category",
]

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_client: Any | None = None


def is_configured() -> bool:
    """True when both Supabase env vars are present."""
    return bool(SUPABASE_URL and SUPABASE_SECRET_KEY)


def get_client() -> Any:
    """Lazy singleton Supabase client. Raises RuntimeError if unconfigured."""
    global _client
    if _client is None:
        if not is_configured():
            raise RuntimeError(
                "Supabase is not configured. Set SUPABASE_URL and SUPABASE_SECRET_KEY."
            )
        try:
            from supabase import create_client

            _client = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "supabase-py is not installed. Run `pip install -r requirements.txt`."
            ) from exc
    return _client


# ---------------------------------------------------------------------------
# Schema (run once in the Supabase SQL editor)
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
-- 1) Every scored policy (anonymized). Auto-saved on every /predict call.
create table if not exists public.quotes (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default now(),
    source text not null default 'single',          -- single | batch | bulk_upload
    consent boolean not null default false,
    Exposure float, VehPower float, VehAge float,
    DrivAge float, BonusMalus float,
    VehBrand text, VehGas text, Area text,
    Density float, Region text,
    predicted_annual_frequency float,
    predicted_claim_severity float,
    expected_loss float,
    pure_premium float,
    technical_premium float,
    final_premium float,
    risk_score float,
    risk_category text
);

-- 2) Consented quotes only (the retraining dataset).
create table if not exists public.training_data (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default now(),
    source text not null default 'single',
    Exposure float, VehPower float, VehAge float,
    DrivAge float, BonusMalus float,
    VehBrand text, VehGas text, Area text,
    Density float, Region text,
    predicted_annual_frequency float,
    predicted_claim_severity float,
    expected_loss float,
    pure_premium float,
    technical_premium float,
    final_premium float,
    risk_score float,
    risk_category text,
    -- Optional outcome fields (filled later via follow-up / B2B matching).
    claim_occurred boolean,
    claim_amount float,
    premium_paid float
);

-- 3) Insurer portfolio uploads with real outcomes (the goldmine).
create table if not exists public.b2b_portfolios (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default now(),
    portfolio_name text not null default 'untitled',
    Exposure float, VehPower float, VehAge float,
    DrivAge float, BonusMalus float,
    VehBrand text, VehGas text, Area text,
    Density float, Region text,
    claim_occurred boolean,
    claim_amount float,
    premium_paid float
);

create index if not exists idx_quotes_created_at on public.quotes (created_at);
create index if not exists idx_training_created_at on public.training_data (created_at);
create index if not exists idx_b2b_created_at on public.b2b_portfolios (created_at);
"""


# ---------------------------------------------------------------------------
# Write helpers (best-effort, never raise into the API)
# ---------------------------------------------------------------------------


def _policy_payload(policy: dict[str, Any]) -> dict[str, Any]:
    """Extract only the anonymized policy attributes we are allowed to store."""
    return {column: policy.get(column) for column in POLICY_COLUMNS}


def _result_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Extract the scored outputs we persist alongside the policy."""
    return {column: result.get(column) for column in RESULT_COLUMNS}


def save_quote(
    policy: dict[str, Any],
    result: dict[str, Any],
    *,
    source: str = "single",
    consent: bool = False,
) -> bool:
    """Persist one scored policy to `quotes` (always) and, if consented, to
    `training_data`. Returns True on success, False when unconfigured/failed."""
    if not is_configured():
        return False
    try:
        client = get_client()
        payload = {**_policy_payload(policy), **_result_payload(result), "source": source, "consent": consent}
        client.table("quotes").insert(payload).execute()
        if consent:
            client.table("training_data").insert(payload).execute()
        return True
    except Exception:
        logger.exception("Failed to save quote to Supabase")
        return False


def save_b2b_portfolio(rows: list[dict[str, Any]], portfolio_name: str = "untitled") -> int:
    """Bulk-insert an insurer portfolio. Returns number of rows inserted."""
    if not is_configured() or not rows:
        return 0
    try:
        client = get_client()
        payload = [
            {**_policy_payload(row), "portfolio_name": portfolio_name,
             "claim_occurred": row.get("claim_occurred"),
             "claim_amount": row.get("claim_amount"),
             "premium_paid": row.get("premium_paid")}
            for row in rows
        ]
        client.table("b2b_portfolios").insert(payload).execute()
        return len(payload)
    except Exception:
        logger.exception("Failed to save B2B portfolio to Supabase")
        return 0


# ---------------------------------------------------------------------------
# Read helpers (for retraining / drift)
# ---------------------------------------------------------------------------


def fetch_training_data() -> dict[str, list[dict[str, Any]]]:
    """Pull consented quotes + B2B portfolios for retraining.

    Returns {"training_data": [...], "b2b_portfolios": [...]}.
    Raises RuntimeError when Supabase is unconfigured.
    """
    if not is_configured():
        raise RuntimeError("Supabase is not configured.")
    client = get_client()
    training = client.table("training_data").select("*").order("created_at").execute()
    b2b = client.table("b2b_portfolios").select("*").order("created_at").execute()
    return {
        "training_data": training.data or [],
        "b2b_portfolios": b2b.data or [],
    }


def fetch_quotes_for_drift(limit: int = 5000) -> list[dict[str, Any]]:
    """Pull recent quotes for predicted-vs-actual drift analysis."""
    if not is_configured():
        raise RuntimeError("Supabase is not configured.")
    client = get_client()
    response = client.table("quotes").select("*").order("created_at", desc=True).limit(limit).execute()
    return response.data or []


def table_counts() -> dict[str, int]:
    """Row counts per table for the /model-info or admin endpoints."""
    if not is_configured():
        return {"configured": False}
    try:
        client = get_client()
        counts: dict[str, int] = {"configured": True}
        for table in ("quotes", "training_data", "b2b_portfolios"):
            response = client.table(table).select("id", count="exact").limit(1).execute()
            counts[table] = int(response.count or 0)
        return counts
    except Exception:
        logger.exception("Failed to read Supabase table counts")
        return {"configured": True, "error": "count query failed"}