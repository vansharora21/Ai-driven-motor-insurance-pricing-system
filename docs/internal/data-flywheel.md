# Data Flywheel — How User Inputs Add Value (Retraining on Real Data)

## The Loop
```
User/Insurer inputs → stored (anonymized) → retraining dataset
        ↑                                        ↓
   better premiums ← improved models ← retrain on real data
```

## B2C Inputs (9-question quotes)
- Every quote = a labeled risk profile (behavior → predicted premium).
- Over time: real behavioral distributions (night driving, braking, mileage) that freMTPL2 never captured → retrain the frequency model on Indian driver behavior, not French data.
- Optional "did you buy / what did you pay" follow-up → price elasticity data (what people actually pay vs. quoted).

## B2B Inputs (portfolio CSVs) — the goldmine
Insurers hand over real claims + premiums + outcomes (the ground truth the model currently lacks):
- **Retrain severity** on Indian claim costs (replace the EUR→₹ FX hack with real Indian claim data)
- **Calibrate pricing config** — fixed expense, minimum premium, risk bands from real market data
- **Validate risk categories** — did High-risk drivers actually claim more?
- **Detect drift** — compare model predictions vs. actual claims year-over-year

## What to Build for This
1. **Data store** (Postgres / SQLite) — persist anonymized quotes + portfolio uploads
2. **Retraining pipeline** — existing `train.py` already does this; just point it at the new data
3. **Drift monitoring** — compare predicted vs. actual to know *when* to retrain

## Honest Caveat
One 250-person survey won't retrain a model meaningfully — but it is the first real slice of Indian behavioral data. The B2B portfolio uploads are what make the model genuinely ours instead of a freMTPL2 clone.

## Privacy Rules
- Store anonymized data only (no emails, no PII)
- Portfolio uploads: pseudonymized policy IDs, no customer PII in outputs
- Consent required for storing B2C responses