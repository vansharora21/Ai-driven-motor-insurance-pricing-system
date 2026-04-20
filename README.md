# AI-Driven Motor Insurance Pricing System

Research-grade motor insurance pricing platform based on the French Motor Third-Party Liability (freMTPL2) dataset. The system separates model training from inference, then exposes reproducible predictions through a CLI pipeline and a Streamlit application.

## Overview

Insurance pricing requires accurate estimation of both claim frequency and claim severity. This project solves that by training two specialized models and combining their outputs into actuarially interpretable premium components.

The project is designed for:

- reproducible academic experiments
- professional ML engineering workflows
- transparent pricing output interpretation

## Key Features

- Dual-model actuarial pipeline:
	- Poisson regression for claim frequency
	- Gamma regression for claim severity
- End-to-end data validation for training and inference inputs
- Shared feature engineering pipeline across training and prediction
- Structured experiment tracking with dataset version metadata
- Inference-only Streamlit app with single-policy and batch scoring
- Saved artifacts for fast deployment (no retraining in app runtime)

## System Architecture

High-level flow:

1. Data ingestion and validation
2. Feature engineering and schema normalization
3. Model training and holdout evaluation
4. Pricing composition (frequency x severity x exposure)
5. Artifact and experiment persistence
6. Inference via CLI or Streamlit app

Core components:

- Data layer: loading, schema checks, cleaning, and dataset alignment
- Modeling layer: frequency and severity estimators
- Pricing layer: loss, premium, risk score, and risk category computation
- Delivery layer: prediction CLI and Streamlit UI

## Dataset Description

This project uses the freMTPL2 French motor insurance dataset split into:

- data/freMTPL2freq.csv:
	- policy-level exposure and claim counts
	- features such as VehPower, VehAge, DrivAge, BonusMalus, Area, Region
- data/freMTPL2sev.csv:
	- claim-level paid amounts for severity modeling

Primary targets:

- ClaimNb for frequency modeling
- ClaimAmount for severity modeling

## Project Structure

```text
.
|- app.py                          # Streamlit inference application
|- train.py                        # Backward-compatible training wrapper
|- predict.py                      # Backward-compatible prediction wrapper
|- run_pipeline.py                 # Backward-compatible alias to training
|- pyproject.toml                  # Packaging + pytest config
|- requirements.txt                # Runtime dependencies
|- configs/
|  |- modeling_config.json         # Model, feature, evaluation, and pricing config
|- scripts/
|  |- train.py                     # Canonical training entry point
|  |- predict.py                   # Canonical prediction entry point
|- src/
|  |- config.py                    # Config loading and determinism helpers
|  |- data_loader.py               # Data loading, validation, cleaning, merge logic
|  |- feature_engineering.py       # Schema normalization + preprocessing pipeline
|  |- frequency_model.py           # Poisson frequency model
|  |- severity_model.py            # Gamma severity model
|  |- pricing_engine.py            # Premium and risk segmentation logic
|  |- model_artifacts.py           # Save/load models and metadata
|  |- experiment_tracking.py       # Structured run summaries
|  |- visualization.py             # Evaluation and portfolio plots
|- data/
|  |- freMTPL2freq.csv
|  |- freMTPL2sev.csv
|- models/
|  |- frequency_model.joblib
|  |- severity_model.joblib
|  |- model_metadata.json
|- results/
|  |- evaluation/metrics.json
|  |- experiments/<run_id>/run_summary.json
|  |- plots/
|  |- premium_reports/
|- tests/
|  |- test_data_preprocessing.py
|  |- test_feature_engineering.py
|  |- test_pricing_logic.py
|  |- test_model_predictions.py
```

## Workflow

1. Load raw frequency and severity datasets.
2. Validate schema:
	 - required columns
	 - numeric compatibility
	 - missing-value checks
3. Clean and engineer model-ready features.
4. Split data into train/test sets.
5. Train holdout models and evaluate performance.
6. Retrain final models on full datasets.
7. Compute pricing outputs for the portfolio.
8. Save:
	 - trained model binaries
	 - metadata and defaults for inference
	 - evaluation metrics and plots
	 - experiment run summary with dataset version hashes

## Models Used

- Frequency Model:
	- sklearn PoissonRegressor
	- target: ClaimNb / Exposure
	- exposure used as sample weights
- Severity Model:
	- sklearn GammaRegressor
	- trained on strictly positive ClaimAmount rows
- Pricing Engine:
	- expected_loss = predicted_claim_count x predicted_claim_severity
	- final premium derived from configurable expense loading and minimum premium
	- risk category assigned using annualized expected loss thresholds

## How to Run

### 1) Installation

```bash
pip install -r requirements.txt
```

Optional dev dependencies:

```bash
pip install -e .[dev]
```

### 2) Training

Preferred:

```bash
python -m scripts.train
```

Backward-compatible:

```bash
python train.py
```

### 3) Batch Prediction

Preferred:

```bash
python -m scripts.predict --input data/freMTPL2freq.csv --output results/premium_reports/predictions.csv
```

Backward-compatible:

```bash
python predict.py --input data/freMTPL2freq.csv --output results/premium_reports/predictions.csv
```

### 4) Run Streamlit App

```bash
streamlit run app.py
```

The app loads saved models from models/ and does not retrain.

## Example Usage

Single policy input (conceptual):

- Exposure: 1.0
- VehPower: 6
- VehAge: 5
- DrivAge: 40
- BonusMalus: 60
- VehBrand: B12
- VehGas: Regular
- Area: C
- Density: 500
- Region: Centre

Typical outputs:

- predicted_annual_frequency
- predicted_claim_severity
- predicted_claim_count
- expected_loss
- final_premium
- risk_category

## Results and Outputs

After training, outputs are generated in:

- models/
	- frequency_model.joblib
	- severity_model.joblib
	- model_metadata.json
- results/evaluation/
	- metrics.json
- results/plots/
	- calibration and distribution charts
- results/premium_reports/
	- premium ranking and prediction files
- results/experiments/
	- per-run summaries with model parameters, metrics, and dataset hashes

## Validation and Testing

Run tests:

```bash
pytest -q
```

Test coverage focuses on:

- preprocessing validation
- feature engineering behavior
- pricing logic correctness
- model prediction interfaces

## Future Improvements

- Add uncertainty intervals for frequency and severity predictions
- Add model monitoring and drift detection for production scoring
- Add richer calibration diagnostics and fairness analysis by segment
- Add API packaging (FastAPI) for service deployment
- Extend to alternative GLM/GBM benchmark models with explainability reports
