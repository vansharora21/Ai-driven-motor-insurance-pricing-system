# B2B Frontend — Portfolio Pricing & Trend Analysis (Insurer-Facing)

## Purpose
An insurance company uploads last year's portfolio (CSV) and receives:
- Portfolio risk & premium analytics
- Trend analysis (claims, frequency, severity, risk mix)
- A re-priced book with recommended premium adjustments for the new policy year
- What-if scenario guidance for underwriting decisions

## Target User
Underwriting / actuarial / pricing teams at insurance companies.

## Input
CSV of last year's portfolio. Two accepted formats:
1. **freMTPL2 schema** — `Exposure, VehPower, VehAge, DrivAge, BonusMalus, VehBrand, VehGas, Area, Density, Region, ClaimNb, ClaimAmount`
2. **Survey-style responses** — same 9 questions as the B2C form (mapped internally via the shared mapping logic)

## Outputs
- **Portfolio summary** — policy count, claim rate, avg frequency/severity, total expected loss
- **Risk distribution** — Low/Medium/High counts + chart
- **Premium distribution** — histogram (₹)
- **Trend analysis** — year-over-year changes in risk mix, claim frequency, severity
- **Re-priced book** — per-policy recommended premium for the new year (CSV download)
- **Premium adjustment guidance** — recommended % change by risk segment
- **What-if scenarios** — e.g., +20% vehicle power, −10 yrs driver age, bonus-malus shifts (reuses `scenario_simulation.py` logic)
- **Fraud / anomaly flags** — Isolation Forest scan of the priced portfolio (optional)

## Architecture
```
Next.js (Vercel)
  └─ /portfolio page (authenticated)
       └─ /api/portfolio (serverless, batch processing)
            ├─ CSV upload → validation (reuse data_loader schema checks)
            ├─ model.json inference in TS (same GLM as B2C)
            ├─ analytics + trend computation in TS
            └─ CSV / JSON export
```

## Re-Pricing Logic (new-year guidance)
1. Score every policy with the current models (frequency × severity → expected loss)
2. Compare against last year's charged premium
3. Recommend risk-based premium = technical premium (₹) with expense loading
4. Segment-level guidance: which segments to raise / lower, and by how much
5. Flag adverse selection risks — low-risk drivers overcharged, high-risk undercharged

## Security
- Authenticated access (login / API key) — B2B only
- Uploads processed server-side; files not persisted (or encrypted at rest)
- No customer PII in outputs (pseudonymized policy IDs only)

## Roadmap
- Multi-year trend comparison
- Drift detection → retraining trigger
- Exportable actuarial report (PDF)
- Integration with policy administration systems (API)