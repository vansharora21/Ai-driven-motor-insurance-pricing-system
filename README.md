# AI-Driven Motor Insurance Pricing System

> **In One Line:** A production-grade actuarial machine learning system that predicts motor insurance premiums by separately modeling claim frequency and severity — then combining them into interpretable, risk-segmented pricing decisions.

---

## Table of Contents

- [Overview](#-overview)
- [How the System Works](#-how-the-system-works)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [System Workflow](#-system-workflow)
- [Models Used](#-models-used)
- [Experiments & Results](#-experiments--results)
- [How to Run](#-how-to-run)
- [Streamlit App](#-streamlit-app)
- [Example Usage](#-example-usage)
- [Validation & Testing](#-validation--testing)
- [Future Improvements](#-future-improvements)
- [Summary](#-summary)
- [Citation & References](#-citation--references)
- [License](#-license)

---

## 🎯 Overview

### The Problem

Insurance companies face a fundamental challenge: **how to price motor insurance premiums fairly and accurately.**

This is difficult because:

- **Not all policies result in claims** — claim frequency varies significantly across drivers and vehicles.
- **When claims do occur, their costs vary widely** — a minor fender-bender and a total write-off require very different payouts.
- **Risk depends on many interacting factors** — driver age, vehicle power, geographic region, bonus-malus history, and more.

Simply averaging historical claims across all policyholders leads to unfair pricing (low-risk drivers subsidize high-risk ones) and potential financial losses for the insurer.

### The Solution

This project implements the **standard actuarial two-part modeling approach** using machine learning:

1. **Frequency Model** → Predicts *how many claims* a policy is likely to generate per year.
2. **Severity Model** → Predicts *the average cost of each claim* when one occurs.

By combining these two predictions with the policy's exposure (duration), the system produces an **actuarially sound premium** that reflects the specific risk profile of each policyholder.

The system also includes a **multi-model experiment pipeline** that benchmarks Poisson/Gamma GLMs against Random Forest and XGBoost alternatives, enabling data-driven model selection.

### Who Is This For?

| Audience | Value |
|----------|-------|
| **Students & Researchers** | Learn professional actuarial pricing with production-grade code |
| **Data Scientists** | Understand the dual-model approach and experiment with ML alternatives |
| **Insurance Professionals** | Explore ML-driven pricing models on real-world data |
| **Developers** | Study modular ML architecture with testing, artifact management, and reproducibility |

---

## 📊 How the System Works

### The Concept

Insurance pricing answers three questions:

1. **"How often will claims happen?"** → The frequency model predicts this (e.g., 0.12 claims/year)
2. **"When claims happen, how much will they cost?"** → The severity model predicts this (e.g., €2,100/claim)
3. **"What is the expected cost to insure this person?"** → Expected Loss = Frequency × Severity

```
Expected Loss = 0.12 claims/year × €2,100/claim = €252/year
Final Premium = €252 × 1.30 (expense loading) + €50 (fixed cost) ≈ €378/year
```

### A Worked Example

Consider two policyholders:

| | Driver A (Young, Sports Car) | Driver B (Experienced, Sedan) |
|---|---|---|
| **Predicted Frequency** | 0.25 claims/year | 0.08 claims/year |
| **Predicted Severity** | €3,200/claim | €1,800/claim |
| **Expected Loss** | €800/year | €144/year |
| **Final Premium** | ~€1,090/year | ~€237/year |
| **Risk Category** | High | Low |

Both receive fair, personalized pricing based on their actual risk profile — not a one-size-fits-all average.

---

## ✨ Key Features

### Dual-Model Actuarial Pipeline
- **Poisson Regression** for claim frequency (standard for count data)
- **Gamma Regression** for claim severity (standard for positive-valued cost data)
- Exposure-weighted training that accounts for partial policy-years

### Multi-Model Experiment Pipeline
- Compare **Poisson, Random Forest, and XGBoost** for frequency
- Compare **Gamma, Random Forest, and XGBoost** for severity
- Automated **9-combination grid** (3 frequency × 3 severity models)
- JSON metrics output, RMSE comparison charts, and per-policy premium comparison CSV

### Visualization & Diagnostics
- Frequency calibration plots (predicted vs. actual by decile)
- Severity scatter plots (predicted vs. actual)
- Model comparison RMSE bar charts
- Feature importance plots for tree-based models
- Premium distribution and risk category charts
- Prediction error distribution histograms

### Scenario Simulation
- What-if analysis: vary vehicle power, driver age, or bonus-malus
- Generates premium sensitivity curves and delta reports
- CLI-driven with CSV and plot outputs

### Streamlit Inference App
- Interactive single-policy scoring with form inputs
- Batch CSV upload and scoring
- Model info, saved metrics, and training visualization display
- Inference-only (loads pre-trained models, no retraining)

### Production-Ready Design
- Comprehensive data validation and schema checking
- Shared feature engineering pipeline across training and inference
- Configurable hyperparameters, pricing rules, and risk thresholds via JSON
- Experiment tracking with dataset versioning (SHA-256 hashes)
- Global random seed for full reproducibility

---

## 🏗️ System Architecture

### High-Level Pipeline

```
Raw Data (freMTPL2freq.csv + freMTPL2sev.csv)
         │
    ┌────▼────────────────┐
    │  Data Validation     │ ← Schema checks, type coercion, missing value handling
    └────┬────────────────┘
         │
    ┌────▼────────────────┐
    │  Feature Engineering │ ← Normalize inputs, clip bounds, log-transform, encode
    └────┬────────────────┘
         │
    ┌────▼────────────────┐
    │  Model Training      │ ← Train frequency + severity models (with holdout eval)
    └────┬────────────────┘
         │
    ┌────▼────────────────┐
    │  Experiment Layer    │ ← Compare GLM vs RF vs XGBoost across all combinations
    └────┬────────────────┘
         │
    ┌────▼────────────────┐
    │  Pricing Engine      │ ← Combine predictions → premiums, risk scores, categories
    └────┬────────────────┘
         │
    ┌────▼────────────────┐
    │  Output Layer        │ ← Artifacts, metrics, plots, reports, experiment logs
    └────┬────────────────┘
         │
    ┌────▼────────────────┐
    │  Streamlit UI        │ ← Interactive scoring (single + batch) with saved models
    └─────────────────────┘
```

### Core Components

| Layer | Component | Purpose |
|-------|-----------|---------|
| **Data** | `data_loader.py` | Load, validate, clean, and merge frequency/severity datasets |
| **Features** | `feature_engineering.py` | Normalize schemas, impute defaults, scale numerics, encode categoricals |
| **Models** | `frequency_model.py` | Train and evaluate frequency models (Poisson/RF/XGBoost) |
| **Models** | `severity_model.py` | Train and evaluate severity models (Gamma/RF/XGBoost) |
| **Models** | `model_factory.py` | Create configurable regressors from `modeling_config.json` |
| **Pricing** | `pricing_engine.py` | Calculate premiums, risk scores, relativities, and risk categories |
| **Experiments** | `experiments.py` | Run multi-model comparison grid with metrics and premium outputs |
| **Tracking** | `experiment_tracking.py` | Persist per-run summaries with dataset hashes and timestamps |
| **Visualization** | `visualization.py` | Generate calibration, comparison, and distribution plots |
| **Simulation** | `scenario_simulation.py` | What-if scenario analysis for underwriting sensitivity |
| **Config** | `config.py` | Centralized configuration management with JSON override support |
| **Artifacts** | `model_artifacts.py` | Save/load trained models, metadata, and metrics |
| **UI** | `app.py` | Streamlit inference app with single-policy and batch scoring |

---

## 📁 Project Structure

```
.
├── app.py                          # Streamlit inference app
├── train.py                        # Backward-compatible training entry point
├── predict.py                      # Backward-compatible prediction entry point
├── scenario_simulation.py          # Backward-compatible scenario entry point
├── run_pipeline.py                 # Full pipeline runner
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project metadata and optional deps
│
├── configs/
│   └── modeling_config.json        # Hyperparameters, pricing rules, experiment settings
│
├── scripts/
│   ├── train.py                    # Main training + evaluation + experiment script
│   ├── predict.py                  # CLI batch prediction script
│   └── scenario_simulation.py      # CLI scenario analysis script
│
├── src/                            # Core library modules
│   ├── config.py                   # Configuration loading and global determinism
│   ├── data_loader.py              # Data loading, validation, and cleaning
│   ├── feature_engineering.py      # Feature preprocessing pipeline
│   ├── frequency_model.py          # Frequency model training and evaluation
│   ├── severity_model.py           # Severity model training and evaluation
│   ├── model_factory.py            # Configurable model creation (Poisson/Gamma/RF/XGB)
│   ├── pricing_engine.py           # Premium calculation and risk scoring
│   ├── experiments.py              # Multi-model comparison experiment runner
│   ├── experiment_tracking.py      # Experiment run persistence and dataset versioning
│   ├── scenario_simulation.py      # What-if scenario simulation engine
│   ├── visualization.py            # Plot generation (calibration, comparison, importance)
│   └── model_artifacts.py          # Model and metadata persistence
│
├── data/
│   ├── freMTPL2freq.csv            # Frequency data (~678K policies)
│   └── freMTPL2sev.csv             # Severity data (~26K claims)
│
├── models/                         # Trained model artifacts
│   ├── frequency_model.joblib      # Serialized frequency model pipeline
│   ├── severity_model.joblib       # Serialized severity model pipeline
│   └── model_metadata.json         # Feature defaults, ranges, pricing config
│
├── results/
│   ├── evaluation/
│   │   ├── metrics.json            # Holdout evaluation metrics
│   │   └── model_comparison.json   # Multi-model experiment results
│   ├── plots/                      # Generated visualization plots
│   ├── premium_reports/            # Prediction and comparison CSVs
│   └── experiments/                # Timestamped experiment run summaries
│
└── tests/                          # Automated test suite
    ├── conftest.py                 # Shared test fixtures
    ├── test_data_preprocessing.py
    ├── test_feature_engineering.py
    ├── test_pricing_logic.py
    ├── test_model_predictions.py
    ├── test_experiments.py
    ├── test_feature_importance.py
    ├── test_scenario_simulation.py
    └── test_visualization.py
```

---

## 📂 Dataset

### freMTPL2 French Motor Insurance Dataset

This project uses the **freMTPL2** dataset, a widely used benchmark in actuarial science research. It contains real-world French motor third-party liability insurance data.

#### Frequency Data (`freMTPL2freq.csv`) — ~678,000 policies

Each row represents one insurance policy with:

| Column | Description |
|--------|-------------|
| `IDpol` | Unique policy identifier |
| `ClaimNb` | Number of claims during the policy period (target for frequency model) |
| `Exposure` | Policy duration in years (e.g., 1.0 = full year, 0.5 = six months) |
| `VehPower` | Vehicle horsepower class (1–20) |
| `VehAge` | Vehicle age in years |
| `DrivAge` | Driver age in years |
| `BonusMalus` | Bonus-malus coefficient (50 = best, higher = worse history) |
| `VehBrand` | Vehicle brand category (B1–B14) |
| `VehGas` | Fuel type (Regular or Diesel) |
| `Area` | Area risk segment (A–F) |
| `Density` | Population density of policyholder's area |
| `Region` | French geographic region |

#### Severity Data (`freMTPL2sev.csv`) — ~26,000 claims

Each row represents one paid claim with:

| Column | Description |
|--------|-------------|
| `IDpol` | Policy identifier (links to frequency data) |
| `ClaimAmount` | Paid claim amount in euros (target for severity model) |

#### Preprocessing Steps

1. **Schema validation** — verify required columns exist and contain valid types
2. **Numeric coercion** — convert all numeric columns, clip to configured bounds
3. **Missing value handling** — impute with domain-appropriate defaults (median for numerics, mode for categoricals)
4. **Feature derivation** — compute `LogDensity = log(1 + Density)` for better model fit
5. **Severity filtering** — keep only claims with `ClaimAmount > 0` (Gamma requirement)
6. **Policy-claim merge** — join severity totals back to policy records for integrated pricing

---

## 🔄 System Workflow

The training pipeline (`scripts/train.py`) executes the following steps:

### Step 1: Data Loading & Validation
- Load `freMTPL2freq.csv` (policies) and `freMTPL2sev.csv` (claims)
- Validate required columns, numeric types, and data integrity
- Clean and merge datasets into model-ready frames

### Step 2: Feature Engineering
- Normalize column names via alias mapping
- Impute missing values using configured defaults
- Clip numeric features to domain-valid ranges
- Compute derived features (`LogDensity`)
- Encode categoricals with `OneHotEncoder`, scale numerics with `StandardScaler`

### Step 3: Train/Test Split
- Split frequency data (80/20) stratified by claim occurrence (`has_claim`)
- Split severity data (80/20) randomly on claim-level records

### Step 4: Holdout Evaluation
- Train evaluation models (Poisson + Gamma by default) on training splits
- Evaluate on holdout test sets: RMSE, Poisson deviance, MAE, Gamma deviance
- Generate calibration and diagnostic plots

### Step 5: Multi-Model Experiments (if enabled)
- Run 9-combination grid: {Poisson, RF, XGBoost} × {Gamma, RF, XGBoost}
- Record per-combination metrics, runtime, and status
- Generate feature importance plots for tree-based models
- Save comparison RMSE chart and per-policy premium comparison CSV

### Step 6: Final Model Training
- Retrain production models on **full dataset** (train + test combined)
- This maximizes available information for deployment predictions

### Step 7: Pricing & Risk Scoring
- Score entire portfolio with final models
- Compute portfolio baselines (mean frequency, severity, expected loss)
- Calculate for each policy:
  - **Annualized Expected Loss** = Frequency × Severity
  - **Expected Loss** = Frequency × Severity × Exposure
  - **Technical Premium** = Expected Loss × (1 + Expense Loading) + Fixed Expense
  - **Final Premium** = max(Technical Premium, Minimum Premium)
  - **Risk Score** = 100 × (Policy Expected Loss / Portfolio Mean Expected Loss)
  - **Risk Category** = Low / Medium / High based on configured thresholds

### Step 8: Artifact Persistence
- Save model binaries (`.joblib`), metadata (`.json`), and evaluation metrics
- Save training plots and premium reports
- Write timestamped experiment run summary with dataset SHA-256 hashes

---

## 🧮 Models Used

### Frequency Model — Poisson Regression (Default)

**Purpose:** Predict the expected number of claims per policy per year.

**Why Poisson?**
- Claim counts are **non-negative integers** (0, 1, 2, … rarely >5)
- Poisson regression is the **standard actuarial choice** for modeling count data
- Naturally handles **exposure as sample weight**, correctly adjusting for partial policy-years

**Implementation:**
- `sklearn.linear_model.PoissonRegressor` with L2 regularization (α = 1e-4)
- Target: `ClaimNb / Exposure` (annualized frequency)
- Sample weight: `Exposure` (policy duration)
- Wrapped in an `sklearn.pipeline.Pipeline` with shared preprocessing

### Severity Model — Gamma Regression (Default)

**Purpose:** Predict the average cost of a claim when one occurs.

**Why Gamma?**
- Claim amounts are **strictly positive continuous values** (e.g., €100–€50,000)
- The Gamma distribution captures the **right-skewed nature** of cost data
- Standard actuarial choice for modeling claim severity

**Implementation:**
- `sklearn.linear_model.GammaRegressor` with L2 regularization (α = 1e-4)
- Target: `ClaimAmount` (positive claims only)
- Filtered to `ClaimAmount > 0` before training (Gamma requirement)
- Wrapped in an `sklearn.pipeline.Pipeline` with shared preprocessing

### Alternative Models — Random Forest & XGBoost

Both are available as configurable alternatives for either frequency or severity:

| Model | Frequency Objective | Severity Objective | Key Hyperparameters |
|-------|--------------------|--------------------|---------------------|
| **Random Forest** | Regression | Regression | 300 estimators, unlimited depth |
| **XGBoost** | `count:poisson` | `reg:gamma` | 300/400 estimators, lr=0.05, depth=6 |

**Why include them?**
- Tree-based models can capture **non-linear interactions** that GLMs miss
- The experiment pipeline enables **data-driven model selection**
- XGBoost with `count:poisson` / `reg:gamma` objectives respects distributional assumptions
- Feature importance from tree models provides **interpretability insights**

> **Note:** XGBoost is an optional dependency. Install with `pip install xgboost` or `pip install -e .[xgboost]`. If not installed, XGBoost experiments are automatically skipped.

### Pricing Engine

Combines frequency and severity predictions into actionable business outputs:

| Output | Formula | Example |
|--------|---------|---------|
| **Annualized Expected Loss** | Frequency × Severity | 0.12 × €2,100 = €252/year |
| **Expected Loss** | Frequency × Severity × Exposure | €252 × 1.0 = €252 |
| **Technical Premium** | Expected Loss × (1 + 0.30) + €50 | €252 × 1.30 + €50 = €378 |
| **Final Premium** | max(Technical Premium, €50) | €378 |
| **Risk Score** | 100 × (Policy Loss / Portfolio Avg Loss) | 95 (slightly below average) |
| **Risk Category** | Based on annualized loss thresholds | Low (≤€150) / Medium (≤€400) / High (>€400) |

---

## 🧪 Experiments & Results

### Model Comparison Pipeline

The experiment system (`src/experiments.py`) runs a **full factorial grid** of model combinations:

- **Frequency models:** Poisson, Random Forest, XGBoost
- **Severity models:** Gamma, Random Forest, XGBoost
- **Total runs:** 9 combinations (3 × 3)

Each combination is trained, evaluated, and scored independently. To manage compute time, experiments use configurable sampling (default: 20% of data, capped at 30,000 rows).

### Evaluation Metrics

| Model Type | Metrics |
|------------|---------|
| **Frequency** | RMSE, Mean Poisson Deviance |
| **Severity** | RMSE, MAE, Mean Gamma Deviance |

Baseline holdout results (Poisson + Gamma on full data):

| Metric | Value |
|--------|-------|
| Frequency RMSE | 0.2396 |
| Frequency Poisson Deviance | 0.3202 |
| Severity RMSE | 9,346.76 |
| Severity MAE | 2,046.98 |
| Severity Gamma Deviance | 1.5795 |

### Generated Outputs

| Output | Location | Description |
|--------|----------|-------------|
| `model_comparison.json` | `results/evaluation/` | Per-combination metrics, runtime, and status |
| `model_comparison.csv` | `results/premium_reports/` | Side-by-side premium predictions (GLM vs RF vs XGB) |
| `model_comparison_rmse.png` | `results/plots/` | RMSE bar chart across all model combinations |
| `feature_importance_*.png` | `results/plots/` | Top-10 feature importance for each tree-based model |
| `predicted_vs_actual.png` | `results/plots/` | Severity prediction scatter plot |
| `error_distribution.png` | `results/plots/` | Residual distribution histogram |
| `frequency_calibration.png` | `results/plots/` | Frequency calibration by prediction decile |
| `premium_distribution.png` | `results/plots/` | Portfolio premium distribution (log scale) |
| `risk_distribution.png` | `results/plots/` | Risk category counts (Low/Medium/High) |

### Scenario Simulation

The scenario simulation module (`src/scenario_simulation.py`) performs what-if analysis:

- Takes a **single policy row** as input
- Generates **7 scenarios**: baseline + 6 variations (±20% vehicle power, ±10 years driver age, ±20 bonus-malus)
- Outputs a CSV with premium deltas and a multi-panel sensitivity plot

---

## 🚀 How to Run

### Prerequisites

- Python ≥ 3.10

### 1. Dataset Setup
This project uses the **freMTPL2** dataset (French Motor Third-Party Liability claims).
1. Download `freMTPL2freq.csv` and `freMTPL2sev.csv` from OpenML (dataset IDs: [41214](https://www.openml.org/d/41214) and [41215](https://www.openml.org/d/41215)) or the `CASdatasets` R package.
2. Create a `data/` directory at the project root if it does not exist, and place both files in it:
   - `data/freMTPL2freq.csv`
   - `data/freMTPL2sev.csv`

### 2. Installation

```bash
git clone https://github.com/vansharora21/Ai-driven-motor-insurance-pricing-system.git
cd Ai-driven-motor-insurance-pricing-system
pip install -r requirements.txt
```

For development (testing):
```bash
pip install -e .[dev]
```

For XGBoost support (optional):
```bash
pip install -e .[xgboost]
```

> **Note on Model Artifacts (Option B):** Pretrained models (`models/*.joblib`) are committed to the repository so that CLI predictions and the Streamlit app work immediately without requiring you to retrain them first. Run the training script to regenerate them from scratch.

### 3. Train the Models

> **CLI Entry Points Note:** The scripts `train.py`, `predict.py`, and `scenario_simulation.py` at the repo root are thin entry point shims; the core CLI script logic is implemented inside the `scripts/` directory.

To train the models, run:
```bash
python train.py
```

This single command will:
1. Load and validate the freMTPL2 datasets
2. Split data into training and test sets
3. Train and evaluate Poisson (frequency) and Gamma (severity) models
4. Run the multi-model experiment grid (if enabled in config)
5. Retrain final models on the full dataset
6. Generate all plots, metrics, and reports
7. Save model artifacts to `models/`

**Expected output:**
```
Loading and preparing freMTPL2 datasets...
Splitting frequency and severity training sets...
Training evaluation models...
Evaluating holdout performance...
Running model comparison experiments...
...
Training complete.
Saved model artifacts to .../models
Saved evaluation metrics to .../results/evaluation/metrics.json
```

### 4. Run Batch Predictions (CLI)

```bash
python predict.py --input data/freMTPL2freq.csv --output results/premium_reports/predictions.csv
```

Output CSV includes: `predicted_annual_frequency`, `predicted_claim_severity`, `expected_loss`, `final_premium`, `risk_category`, and more.

### 5. Run Scenario Simulation

```bash
python scenario_simulation.py --input data/sample_batch.csv
```

Generates `results/premium_reports/scenario_analysis.csv` and `results/plots/scenario_premium_curves.png`.

### 6. Launch the Streamlit App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. **Important:** Train models first (Step 3) — the app loads pre-trained artifacts and does not retrain.

---

## 💡 Streamlit App

### Overview

The Streamlit app provides an **inference-only** interface for scoring policies using pre-trained models. It has three tabs:

### Tab 1: Single Policy Prediction

**Purpose:** Score one policy at a time with detailed results.

**How to use:**
1. Fill in the input form with policy attributes:
   - Exposure (policy duration in years)
   - Vehicle Power, Vehicle Age
   - Driver Age, Bonus-Malus coefficient
   - Vehicle Brand, Fuel Type, Area, Region
   - Population Density
2. Click **"Predict Premium"**
3. Review results: predicted frequency, severity, expected loss, final premium, and risk category badge

### Tab 2: Batch Upload

**Purpose:** Score multiple policies simultaneously from a CSV file.

**How to use:**
1. Download the input template CSV (provided in the app)
2. Fill in your policy data following the template format
3. Upload the CSV and view scored results with risk distribution charts
4. Download the predictions as CSV

**Required CSV columns:**
```
Exposure,VehPower,VehAge,DrivAge,BonusMalus,VehBrand,VehGas,Area,Density,Region
```

### Tab 3: Instructions & Model Info

Displays model documentation, saved evaluation metrics, and training visualization plots (premium distribution, risk distribution, frequency calibration, severity predictions).

### Understanding the Outputs

| Output | What It Means |
|--------|--------------|
| **Predicted Frequency** | Expected claims per year (e.g., 0.12 = ~12% annual claim probability) |
| **Predicted Severity** | Average claim cost when a claim occurs (e.g., €2,100) |
| **Expected Loss** | Annual expected payout = Frequency × Severity × Exposure |
| **Final Premium** | Recommended price after loading expenses and minimum floor |
| **Risk Category** | Low (≤€150 loss) / Medium (≤€400) / High (>€400) |

---

## 📊 Example Usage

### Single Policy

**Input:**
```
Exposure: 1.0 year       VehPower: 6        VehAge: 5 years
DrivAge: 40 years         BonusMalus: 60     VehBrand: B12
VehGas: Regular           Area: C            Density: 500/km²
Region: Centre
```

**Output:**
```
Predicted Annual Frequency:  0.12 claims/year
Predicted Claim Severity:    €2,100 per claim
Expected Loss:               €252/year
Final Premium:               €378/year
Risk Category:               Medium
Risk Score:                  95 (slightly below portfolio average)
```

### Batch Prediction

Upload a CSV with 100+ policies → receive instant premium predictions for every row, with risk distribution charts and downloadable results.

---

## ✅ Validation & Testing

Run the full test suite:

```bash
pytest -q
```

### Test Coverage

| Test Module | Coverage Area |
|-------------|---------------|
| `test_data_preprocessing.py` | Schema validation, type coercion, missing value handling |
| `test_feature_engineering.py` | Normalization, scaling, encoding correctness |
| `test_pricing_logic.py` | Premium calculation, risk scoring, category assignment |
| `test_model_predictions.py` | Frequency and severity prediction interfaces |
| `test_experiments.py` | Multi-model comparison pipeline |
| `test_feature_importance.py` | Feature importance extraction for tree-based models |
| `test_scenario_simulation.py` | Scenario generation, scoring, and report output |
| `test_visualization.py` | Plot generation and file output |

All tests validate behavior using synthetic fixtures — no retraining or dataset downloads required.

---

## 🔮 Future Improvements

- **Uncertainty Quantification** — confidence intervals around frequency and severity predictions
- **Drift Detection** — monitor model performance on new data and alert when retraining is needed
- **Fairness Analysis** — audit pricing across demographics for non-discriminatory underwriting
- **API Service** — package as a FastAPI microservice for enterprise deployment
- **SHAP Explainability** — add SHAP value analysis alongside feature importance
- **Calibration Diagnostics** — enhanced residual analysis and population stability indices

---

## 📌 Summary

This project implements a **complete, end-to-end motor insurance pricing system** built on actuarial best practices. It separately models claim frequency (Poisson) and severity (Gamma), combines them into risk-adjusted premiums, and provides a multi-model experiment framework for benchmarking against Random Forest and XGBoost alternatives. The system includes a full visualization pipeline, scenario simulation for underwriting sensitivity analysis, comprehensive testing, and an interactive Streamlit app for real-time policy scoring — all powered by the industry-standard freMTPL2 French Motor Insurance dataset.

---

## 📖 Citation & References

This project implements actuarial pricing methodologies following industry standards:

- **Poisson Regression** for frequency modeling — standard in non-life insurance for count data
- **Gamma Regression** for severity modeling — standard for positive-valued claim cost data
- **freMTPL2 Dataset** — French Motor Third-Party Liability insurance data, publicly available for actuarial research ([OpenML](https://www.openml.org/d/41214))
- **Dual-model pricing** — the frequency × severity decomposition is the foundational approach in actuarial ratemaking

---

## 📝 License

This project is provided as-is for educational and research purposes.
