# Motor Insurance Pricing Engine

Research-grade motor insurance pricing system built on the real freMTPL2 frequency and severity datasets.

## Project Structure

```text
.
|- app.py                        # Streamlit inference UI
|- train.py                      # Backward-compatible wrapper for scripts/train.py
|- predict.py                    # Backward-compatible wrapper for scripts/predict.py
|- run_pipeline.py               # Backward-compatible training alias
|- pyproject.toml                # Packaging and test configuration
|- requirements.txt
|- scripts/
|  |- train.py                   # Canonical training entry point
|  |- predict.py                 # Canonical batch inference entry point
|- src/
|  |- __init__.py                # Python package marker
|  |- config.py
|  |- data_loader.py
|  |- feature_engineering.py
|  |- frequency_model.py
|  |- severity_model.py
|  |- pricing_engine.py
|  |- model_artifacts.py
|  |- experiment_tracking.py
|  |- visualization.py
|- configs/
|  |- modeling_config.json       # Canonical modeling config
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
|  |- conftest.py
|  |- test_data_preprocessing.py
|  |- test_feature_engineering.py
|  |- test_pricing_logic.py
|  |- test_model_predictions.py
```

## Input Validation

Validation is enforced for both training datasets and user-uploaded inference CSV files:

- required columns are checked and missing column names are reported
- numeric schema is validated for required numeric fields
- missing values in required fields trigger clear validation errors
- alias columns are supported for inference input normalization

Validation errors raise DataValidationError with actionable messages.

## Training

Install dependencies:

```bash
pip install -r requirements.txt
```

Train and evaluate models:

```bash
python -m scripts.train
```

Backward-compatible command (still supported):

```bash
python train.py
```

Training outputs:

- model binaries and metadata under models/
- evaluation metrics under results/evaluation/
- portfolio plots and premium reports under results/
- structured experiment run records under results/experiments/

## Experiment Tracking

Each training run saves:

- model parameters from configuration
- evaluation metrics for frequency and severity models
- dataset version metadata (file path, size, timestamp, sha256)
- dataset summary and data quality statistics

These records are written to results/experiments/<timestamp>/run_summary.json.

## Inference

Run batch inference:

```bash
python -m scripts.predict --input data/freMTPL2freq.csv --output results/premium_reports/predictions.csv
```

Backward-compatible command:

```bash
python predict.py --input data/freMTPL2freq.csv --output results/premium_reports/predictions.csv
```

## Streamlit App

```bash
streamlit run app.py
```

The app validates uploaded CSV files and returns clear, user-friendly errors for malformed files or schema violations.

## Testing

Run unit tests:

```bash
pytest -q
```

Coverage includes preprocessing validation, feature engineering, pricing logic, and model prediction behavior.
