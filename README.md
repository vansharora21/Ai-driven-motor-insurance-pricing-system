# Motor Insurance Pricing Engine

Production-style motor insurance pricing project built on the real `freMTPL2` French motor insurance datasets.

## What Changed

This repository now uses:

- `data/freMTPL2freq.csv` for claim frequency and policy features
- `data/freMTPL2sev.csv` for real observed claim severities
- saved sklearn pipelines for training and inference
- a Streamlit app that only performs inference

The previous demo-only pieces such as synthetic severity targets and in-app training have been removed from the main workflow.

## Project Structure

```text
.
|- app.py                     # Streamlit inference UI
|- train.py                   # Full training, evaluation, and artifact generation
|- predict.py                 # CLI and reusable inference entry point
|- run_pipeline.py            # Backward-compatible alias to train.py
|- data/
|  |- freMTPL2freq.csv        # Policy-level exposure and claim counts
|  |- freMTPL2sev.csv         # Claim-level paid amounts
|  |- sample_batch.csv        # Optional batch sample generated from real data
|- artifacts/
|  |- frequency_model.joblib
|  |- severity_model.joblib
|  |- model_metadata.json
|- results/
|  |- evaluation/metrics.json
|  |- plots/
|  |- premium_reports/top_premiums.csv
|- src/
|  |- data_loader.py          # Dataset loading, cleaning, and merge logic
|  |- feature_engineering.py  # Shared feature cleaning and preprocessing
|  |- frequency_model.py      # Poisson regression pipeline
|  |- severity_model.py       # Gamma regression pipeline
|  |- pricing_engine.py       # Premium calculation and risk-band logic
|  |- model_artifacts.py      # Save/load helpers for models and metadata
|  |- visualization.py        # Evaluation and portfolio plots
|  |- fraud_detection.py      # Optional post-pricing anomaly detector
|  |- simulation.py           # Optional portfolio stress testing helper
```

## Training

Install dependencies:

```bash
pip install -r requirements.txt
```

Train models and generate artifacts:

```bash
python train.py
```

This will:

- load and clean `freMTPL2freq` and `freMTPL2sev`
- merge them by `IDpol`
- train a Poisson frequency model and Gamma severity model
- evaluate both on holdout sets
- save joblib artifacts and evaluation metrics
- generate plots and a `top_premiums.csv` report

## Inference

Run batch inference from the command line:

```bash
python predict.py --input data/sample_batch.csv --output results/premium_reports/predictions.csv
```

Expected input columns:

```text
IDpol, Exposure, VehPower, VehAge, DrivAge, BonusMalus, VehBrand, VehGas, Area, Density, Region
```

## Streamlit App

Launch the app:

```bash
streamlit run app.py
```

The app keeps the same three main workflows:

- single-policy premium calculation
- batch CSV scoring
- portfolio analytics based on saved training outputs

## Optional Utilities

- `generate_mock_data.py`: creates a realistic sample batch file from freMTPL2
- `process_responses.py`: exports a clean batch template from the frequency dataset
- `create_notebook.py`: generates a Jupyter notebook for the new real-data workflow
- `src/fraud_detection.py`: optional anomaly detection on scored portfolios
- `src/simulation.py`: optional stress testing on already prepared policy data
