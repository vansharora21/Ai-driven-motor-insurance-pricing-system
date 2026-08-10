# B2C Frontend — "Get My Quote" (Customer-Facing)

## Purpose
A normal person (car owner) answers 9 simple driving-behavior questions and instantly receives:
- A personalized annual premium estimate (₹)
- Risk category (Low / Medium / High)
- A transparent breakdown of how the price was calculated (waterfall chart)
- A personalized insurance report with coverage recommendations

## Target User
Individual car owners shopping for or renewing motor insurance.

## The 9 Questions (from the Google Form)
1. Age (18–100)
2. Vehicle Type (Sedan / SUV / Hatchback / Motorcycle / Other)
3. Vehicle age (in years)
4. Average daily kilometers driven
5. Night driving frequency (Low / Medium / High)
6. Harsh braking frequency (Low / Medium / High)
7. Traffic accidents in the last 2 years
8. Motor insurance claim filed in the last 2 years (Yes / No)
9. Years of driving experience

## Survey → Model Mapping
Reuses `_google_form_to_pricing_frame` logic (currently in `app.py`, to be moved into `src/`):

| Survey answer | Mapped to | Rule |
|---|---|---|
| Age | `DrivAge` | direct |
| Vehicle age | `VehAge` | direct |
| Daily km | `Exposure` | km/20, clipped 0.1–2.5 |
| Vehicle Type | `VehPower` / `VehBrand` / `VehGas` | Sedan→6/B12, SUV→8/B11, Hatchback→5/B10, Motorcycle→4/B2+Diesel, Other→6/B12 |
| Night driving | `Area` (risk band) | Low/Med/High → +0/1/2 points |
| Harsh braking | `Area` | same scoring |
| Accidents (2yr) | `Area` + `Density` + `BonusMalus` | +points, +180 density, +14 bonus-malus |
| Claim filed (2yr) | `Area` + `Density` + `BonusMalus` | +points, +140 density, +18 bonus-malus |
| Experience | `BonusMalus` | −0.6/yr (max 30 yrs) |
| — | `Region` | fixed "Centre" |

## Outputs
- `predicted_annual_frequency` (claims/yr)
- `predicted_claim_severity` (₹ per claim)
- `expected_loss` (₹)
- `technical_premium` / `final_premium` (₹)
- `risk_score` (vs. portfolio baseline)
- `risk_category` (Low / Medium / High)
- Waterfall chart — pricing proof (expected loss → expense loading → fixed expense → final premium)
- Insurance report — plain-language risk explanation + coverage recommendations

## Currency
INR (₹). Config: FX 93 ₹/€, fixed expense ₹500, minimum premium ₹2,500, risk bands Low ≤₹15k / Medium ≤₹40k / High >₹40k.

## Architecture
```
Next.js (Vercel, free — no sleep, no cron)
  └─ /quote page (React form, 9 questions)
       └─ /api/quote (serverless function)
            ├─ survey → freMTPL2 mapping (TS)
            ├─ model.json (GLM coefficients, hosted on HF Hub)
            ├─ inference in TS: freq = exp(β·x), severity = exp(β·x)
            └─ pricing in TS (₹) + report generation
```

## Privacy
- No email required. No PII stored.
- Optional: save anonymous response to a research dataset (with consent).

## Insurance Report — Coverage Recommendations
Based on risk category + vehicle type:

| Risk | Recommended coverage |
|---|---|
| Low | Third-party + basic own-damage; higher voluntary deductible to lower premium |
| Medium | Comprehensive + zero-depreciation |
| High | Comprehensive + zero-depreciation + engine protection + roadside assistance; consider telematics/UBI discount |
| Motorcycle | Two-wheeler-specific policy |
| Vehicle >10 yrs | Consider third-party only or declining own-damage value |

## Success Metrics
- Quote completion rate
- Time-to-quote (< 30 seconds)
- Report usefulness (user feedback)