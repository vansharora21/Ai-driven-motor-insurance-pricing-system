-- Supabase schema for the Motor Insurance Pricing data flywheel.
--
-- How to run:
--   1. Open your Supabase project dashboard
--   2. Go to SQL Editor (left sidebar)
--   3. Paste this entire file into the editor
--   4. Click "Run"
--
-- Creates three tables:
--   quotes           -> every scored policy (anonymized), auto-saved on /predict
--   training_data    -> consented quotes only (the retraining dataset)
--   b2b_portfolios   -> insurer portfolio uploads with real outcomes

-- 1) Every scored policy (anonymized). Auto-saved on every /predict call.
create table if not exists public.quotes (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default now(),
    source text not null default 'single',          -- single | batch | bulk_upload | backfill
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

-- Row-level security: the API uses the secret key (bypasses RLS).
-- Keep tables locked down by default; only the service role can read/write.
alter table public.quotes enable row level security;
alter table public.training_data enable row level security;
alter table public.b2b_portfolios enable row level security;