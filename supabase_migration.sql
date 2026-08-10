-- Migration: add missing columns to existing Supabase tables.
--
-- Run this in the Supabase SQL Editor (New query -> paste -> Run) AFTER the
-- tables already exist. It only ADDS columns; it never drops or alters
-- existing data, so it is safe to run on tables that already have rows.
--
-- If you prefer a clean slate instead: drop the three tables and re-run
-- supabase_schema.sql (tables are empty, so nothing is lost).

-- quotes: missing driver age + optional outcome columns
alter table public.quotes
    add column if not exists drirage float,
    add column if not exists claim_occurred boolean,
    add column if not exists claim_amount float,
    add column if not exists premium_paid float,
    add column if not exists portfolio_name text;

-- training_data: missing consent flag + driver age
alter table public.training_data
    add column if not exists consent boolean not null default false,
    add column if not exists drirage float,
    add column if not exists portfolio_name text;

-- b2b_portfolios: missing source/consent + driver age + scored outputs
alter table public.b2b_portfolios
    add column if not exists source text not null default 'single',
    add column if not exists consent boolean not null default false,
    add column if not exists drirage float,
    add column if not exists predicted_annual_frequency float,
    add column if not exists predicted_claim_severity float,
    add column if not exists expected_loss float,
    add column if not exists pure_premium float,
    add column if not exists technical_premium float,
    add column if not exists final_premium float,
    add column if not exists risk_score float,
    add column if not exists risk_category text;