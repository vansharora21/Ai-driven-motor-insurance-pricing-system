# AI-Driven Motor Insurance Pricing System

> **In One Line:** An actuarial machine learning system that predicts insurance premiums by separately modeling claim frequency and severity, then combining them for interpretable pricing decisions.

---

## 🎯 Overview

### The Problem

Insurance companies face a critical challenge: **how to price premiums fairly and accurately?**

Pricing is difficult because:
- Not all policies will have claims (claim frequency varies significantly)
- When claims do occur, their amounts vary widely (claim severity varies significantly)
- These variations depend on many factors: driver age, vehicle power, location, driving history, and more

Simply averaging claims across all policies leads to unfair pricing and profit losses. You need to model **what causes claims** and **what claims cost** separately.

### The Solution

This project uses **two specialized machine learning models** to solve this:

1. **Frequency Model** → How many claims is this policy likely to have per year?
2. **Severity Model** → If a claim happens, how much will it cost on average?

By combining these predictions with exposure time (policy duration), you get an **actuarially sound premium** that reflects the specific risk profile of each policy.

### Who Should Use This?

- **Students & Researchers:** Learn how professional insurance systems work with production-grade code
- **Data Scientists:** Understand the dual-model actuarial approach and how to implement it
- **Insurance Professionals:** Experiment with ML-driven pricing models
- **Developers:** Study modular ML architecture with proper testing and artifact management

---

## 📊 How the System Works (Simple Explanation)

### The Concept

Think of insurance pricing as answering three questions:

1. **"How often will claims happen?"** → Frequency model predicts this (e.g., 0.15 claims per year)
2. **"If claims happen, how much will they cost?"** → Severity model predicts this (e.g., €2,000 per claim)
3. **"So what's the expected cost to insure this person?"** → Expected Loss = Frequency × Severity

**Expected Loss = 0.15 claims/year × €2,000/claim = €300/year**

Then add operating costs and profit margins to get the final premium: **Final Premium ≈ €390/year**

### A Real Example

Consider two drivers:

**Driver A (Young, Sports Car):**
- High frequency risk (new drivers have more accidents)
- High severity risk (sports cars are expensive to fix)
- Result: High premium (e.g., €800/year)

**Driver B (Experienced, Regular Car):**
- Low frequency risk (experienced drivers have fewer accidents)
- Low severity risk (regular cars cost less to fix)
- Result: Low premium (e.g., €350/year)

Both get fair, personalized pricing based on their actual risk profile.

### The Data

The system learns from the **freMTPL2 French Motor Insurance Dataset**, which contains:
- 678,000+ real insurance policies
- Historical claim data
- 11 policy features (age, vehicle power, region, etc.)

This historical data trains the models to recognize patterns and predict future claims accurately.

---

## ✨ Key Features

- **Dual-Model Actuarial Pipeline**
  - Poisson Regression for claim frequency (count data)
  - Gamma Regression for claim severity (positive-valued continuous data)

- **Production-Ready Code**
  - Comprehensive data validation for safety
  - Shared feature engineering across training and inference
  - Structured experiment tracking with dataset versioning

- **Fast Inference**
  - Models trained once and saved as artifacts
  - Streamlit app loads pre-trained models (no retraining)
  - Instant single-policy or batch scoring

- **Transparent & Interpretable**
  - Clear premium components (frequency, severity, expected loss, risk category)
  - Pricing logic is auditable and explainable
  - Risk categories (Low/Medium/High) based on expected losses

- **Multiple Interfaces**
  - CLI for batch predictions on CSV files
  - Streamlit app for interactive single-policy and batch scoring
  - Python API for programmatic use

---

## 🏗️ System Architecture

### High-Level Workflow

```
Raw Data (Frequency & Severity)
         ↓
    ┌────────────────────┐
    │ Data Validation    │ ← Check schema, types, missing values
    └────────────────────┘
         ↓
    ┌────────────────────┐
    │ Feature            │ ← Normalize inputs, scale, encode
    │ Engineering        │
    └────────────────────┘
         ↓
    ┌────────────────────┐
    │ Model Training     │ ← Train frequency & severity models
    └────────────────────┘
         ↓
    ┌────────────────────┐
    │ Pricing Engine     │ ← Combine models into premiums
    └────────────────────┘
         ↓
    Output: Premiums & Risk Categories
```

### Core Components

| Component | Purpose |
|-----------|---------|
| **Data Layer** | Load, validate, clean, and align frequency/severity datasets |
| **Feature Engineering** | Normalize inputs, scale numerics, encode categories into a consistent format |
| **Frequency Model** | Poisson regression predicting claim count per year per unit exposure |
| **Severity Model** | Gamma regression predicting average claim amount when claims occur |
| **Pricing Engine** | Converts model outputs into premium components and risk scores |
| **Streamlit App** | Interactive UI for single-policy and batch scoring |
| **CLI Prediction** | Script for batch processing CSV files |

---

## 📁 Project Structure

```
.
├── app.py                          # Streamlit app (inference UI)
├── train.py                        # Backward-compatible training wrapper
├── predict.py                      # Backward-compatible prediction wrapper
├── requirements.txt                # Python dependencies
├── configs/
│   └── modeling_config.json        # Model hyperparameters, pricing rules
├── scripts/
│   ├── train.py                    # Main training script
│   └── predict.py                  # Main prediction script
├── src/                            # Core modules
│   ├── config.py                   # Configuration management
│   ├── data_loader.py              # Data loading & validation
│   ├── feature_engineering.py      # Feature preprocessing
│   ├── frequency_model.py          # Frequency modeling (Poisson)
│   ├── severity_model.py           # Severity modeling (Gamma)
│   ├── pricing_engine.py           # Premium calculation
│   ├── model_artifacts.py          # Model persistence
│   ├── experiment_tracking.py      # Experiment logging
│   └── visualization.py            # Plots & reports
├── data/
│   ├── freMTPL2freq.csv            # Frequency data (678k policies)
│   └── freMTPL2sev.csv             # Severity data (claims data)
├── models/                         # Trained artifacts
│   ├── frequency_model.joblib      # Frequency model binary
│   ├── severity_model.joblib       # Severity model binary
│   └── model_metadata.json         # Defaults & feature config
├── results/
│   ├── evaluation/metrics.json     # Model performance metrics
│   ├── plots/                      # Calibration & distribution plots
│   └── premium_reports/            # Prediction outputs
└── tests/
    ├── test_data_preprocessing.py
    ├── test_feature_engineering.py
    ├── test_pricing_logic.py
    └── test_model_predictions.py
```

---

## 🔄 System Workflow

### Step 1: Data Loading & Validation
- Load frequency dataset (policies + claim counts) and severity dataset (claims + amounts)
- Validate required columns, numeric types, and missing value thresholds
- Ensure data integrity before modeling

### Step 2: Feature Engineering
- Normalize all inputs to a consistent schema
- Handle missing values using domain-appropriate defaults
- Scale numeric features and encode categorical variables
- Create a shared pipeline used identically in training and inference

### Step 3: Train/Test Split
- Stratify by claim count (frequency) for balanced evaluation
- Reserve 20% of policies for holdout testing

### Step 4: Frequency Model Training
- Train Poisson regression on claim counts
- Use exposure (policy duration) as sample weights
- Evaluate on holdout test set using RMSE and Poisson deviance

### Step 5: Severity Model Training
- Train Gamma regression on positive claim amounts
- Filter to claims with amounts > €0 (requirement for Gamma distribution)
- Evaluate on holdout test set using RMSE and Gamma deviance

### Step 6: Final Model Training
- Retrain both models on full dataset (training + test data combined)
- This maximizes information for production predictions

### Step 7: Pricing & Risk Scoring
- For each policy, predict:
  - Annual claim frequency (e.g., 0.12 claims/year)
  - Average claim severity (e.g., €2,500)
- Calculate premium components:
  - **Expected Loss** = Frequency × Severity
  - **Technical Premium** = Expected Loss × Exposure
  - **Final Premium** = Technical Premium + Operating Expenses
- Assign risk category (Low/Medium/High) based on expected loss thresholds

### Step 8: Artifact Persistence
- Save trained model binaries (.joblib files)
- Save metadata (feature defaults, config, numeric ranges)
- Save evaluation metrics and visualizations
- Document dataset version hashes for reproducibility

---

## 🧮 Models Used

### Frequency Model (Poisson Regression)

**What it does:** Predicts how many claims will occur per year per unit of exposure (e.g., policies written for partial years have fractional exposure).

**Why Poisson?**
- Claim counts are **non-negative integers** (0, 1, 2, ..., rarely >5)
- Poisson regression is the standard actuarial choice for count data
- Naturally handles exposure as a sample weight

**Model Details:**
- Sklearn: `PoissonRegressor`
- Target: Number of claims per policy
- Exposure: Weighting by policy duration (1.0 = full year, 0.5 = half year)
- Regularization: L2 with alpha=1e-4

**Output Example:**
- Input: Driver age 35, vehicle power 6, bonus-malus 60, region Centre
- Output: 0.12 claims/year

### Severity Model (Gamma Regression)

**What it does:** Predicts the average amount of a claim when claims occur (e.g., average repair cost).

**Why Gamma?**
- Claim amounts are **positive-valued continuous data** (e.g., €100 to €50,000)
- Gamma distribution is flexible and fits cost data well
- Captures the right-skewed distribution of claim amounts

**Model Details:**
- Sklearn: `GammaRegressor`
- Target: Claim amount (for claims with amount > 0)
- Preprocessing: Filter to policies with actual claims before training
- Regularization: L2 with alpha=1e-4

**Output Example:**
- Input: Same policy as above
- Output: €2,100 average claim amount

### Pricing Engine

**Combines both models into actionable business outputs:**

| Output | Calculation | Meaning |
|--------|-----------|---------|
| Expected Loss | Frequency × Severity | €0.12 × €2,100 = €252/year |
| Technical Premium | Expected Loss × (1 + Loading) | Premium before profit margin |
| Final Premium | Technical Premium + Expense | What customer pays |
| Risk Score | (Expected Loss / Portfolio Avg) × 100 | Relative risk (100 = average) |
| Risk Category | Based on Expected Loss Thresholds | Low/Medium/High underwriting band |

---

## 🚀 How to Run

### 1) Installation

Clone the repository and install dependencies:

```bash
cd AI-Driven-Motor-Insurance-Pricing-System
pip install -r requirements.txt
```

Optional dev dependencies (for testing):

```bash
pip install -e .[dev]
```

### 2) Training the Models

Train both frequency and severity models on the full dataset:

**Preferred method:**
```bash
python -m scripts.train
```

**Alternative method:**
```bash
python train.py
```

**What happens:**
- Loads `data/freMTPL2freq.csv` and `data/freMTPL2sev.csv`
- Validates and cleans data
- Trains frequency and severity models
- Evaluates on holdout test set
- Saves trained models to `models/`
- Saves metrics and plots to `results/`

**Expected output:**
```
Loading datasets...
Validating schema...
Training frequency model...
Frequency Model RMSE: 0.45
Training severity model...
Severity Model RMSE: 1850.32
Saving artifacts...
✓ Training complete
```

### 3) Batch Prediction (CLI)

Score a CSV file with policy data and generate premium predictions:

**Preferred method:**
```bash
python -m scripts.predict \
  --input data/freMTPL2freq.csv \
  --output results/premium_reports/predictions.csv
```

**Alternative method:**
```bash
python predict.py \
  --input data/freMTPL2freq.csv \
  --output results/premium_reports/predictions.csv
```

**Output CSV includes:**
- All input columns
- `predicted_annual_frequency`: Predicted claims per year
- `predicted_claim_severity`: Predicted claim amount
- `predicted_claim_count`: Frequency × Exposure
- `expected_loss`: Frequency × Severity
- `final_premium`: Pricing engine output
- `risk_category`: Low/Medium/High

### 4) Interactive Scoring (Streamlit App)

Launch the interactive Streamlit application for single-policy and batch scoring:

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501/` and provides:
- **Single Policy Tab:** Form-based input for one policy at a time
- **Batch Upload Tab:** CSV upload for scoring multiple policies
- **Model Info Tab:** Documentation, metrics, and help

**Important:** The Streamlit app loads pre-trained models from `models/` and does **not** retrain them. Always run `python -m scripts.train` first.

---

## 💡 How to Use the Streamlit App

### Overview

The Streamlit app is designed for **inference only** (scoring existing or new policies). It has three tabs:

### Tab 1: Single Policy Prediction

**Purpose:** Score one policy at a time and see detailed results.

**How to use:**

1. **Enter policy details** using the form:
   - Exposure: Years insured (0-2.5, default 1.0)
   - Vehicle Power: Engine size (1-20, default 6)
   - Vehicle Age: Years old (0-100, default 6)
   - Driver Age: Years old (18-100, default 44)
   - Bonus-Malus: Discount/penalty score (50-350, default 50)
   - Vehicle Brand: Dropdown (B1, B2, ..., B14)
   - Fuel Type: Dropdown (Regular, Diesel)
   - Area: Dropdown (A, B, C, D, E, F)
   - Population Density: People per km² (0-27000, default 393)
   - Region: Dropdown (all French regions)

2. **Click "Score Policy"** to generate predictions

3. **Review the results:**
   - **Frequency:** How many claims per year
   - **Severity:** Average claim amount
   - **Expected Loss:** Annual expected insurance cost
   - **Final Premium:** What to charge the customer
   - **Risk Category Badge:** Visual risk indicator

### Tab 2: Batch Upload

**Purpose:** Score multiple policies at once from a CSV file.

**How to use:**

1. **Prepare a CSV file** with the same columns as Tab 1 (Exposure, VehPower, VehAge, etc.)
2. **Upload the file** using the file uploader
3. **Click "Score Batch"** to generate predictions
4. **View results:**
   - Table of all predictions
   - Risk distribution chart showing Low/Medium/High breakdown
   - Download predictions as CSV

**Expected CSV format:**
```
Exposure,VehPower,VehAge,DrivAge,BonusMalus,VehBrand,VehGas,Area,Density,Region
1.0,6,5,40,60,B12,Regular,C,500,Centre
0.5,8,2,25,100,B1,Diesel,A,1000,Île-de-France
```

### Tab 3: Model Info

**Purpose:** Documentation and reference information.

**Contents:**
- How the system works
- Feature explanations
- Model performance metrics
- Saved visualization plots

### What the Outputs Mean

| Output | Definition |
|--------|-----------|
| **Predicted Frequency** | Expected claims per policy per year (e.g., 0.15 = 15% chance of a claim) |
| **Predicted Severity** | Average claim amount when claims occur (e.g., €2,500) |
| **Expected Loss** | Annual expected insurance payout (Frequency × Severity) |
| **Final Premium** | Recommended price to charge customer (Expected Loss + Operating Cost + Profit) |
| **Risk Category** | Low (safe) / Medium (typical) / High (risky) based on expected loss |

---

## 📊 Example Usage

### Single Policy Example

**Input:**
```
Exposure: 1.0 year
Vehicle Power: 6
Vehicle Age: 5 years
Driver Age: 40 years
Bonus-Malus: 60 (average)
Vehicle Brand: B12
Fuel Type: Regular
Area: C
Population Density: 500/km²
Region: Centre
```

**Output:**
```
Predicted Annual Frequency:  0.12 claims/year
Predicted Claim Severity:    €2,100 per claim
Expected Loss:               €252/year (0.12 × €2,100)
Final Premium:               €390/year (€252 + €138 operating cost)
Risk Category:               Medium
Risk Score:                  95 (slightly below average)
```

### Batch Upload Example

Upload CSV with 100 policies → Get 100 premium predictions instantly.

Result table shows:
- Each policy's frequency, severity, expected loss, and premium
- Risk category color-coded (Green/Yellow/Red)
- Downloadable CSV with all results

Chart shows:
- Distribution of premiums across the portfolio
- How many policies fall into each risk category
- Premium ranges for planning and analysis

---

## 📈 Results and Outputs

After training (`python -m scripts.train`), the system generates:

### Model Artifacts (`models/`)
- `frequency_model.joblib` – Trained Poisson model (binary)
- `severity_model.joblib` – Trained Gamma model (binary)
- `model_metadata.json` – Feature defaults, numeric ranges, and config

### Evaluation Metrics (`results/evaluation/`)
- `metrics.json` – RMSE, deviance, and other performance statistics for both models

### Visualizations (`results/plots/`)
- Frequency calibration plots (predicted vs. actual)
- Severity prediction scatter plots
- Premium distribution across the portfolio
- Risk category distribution (Low/Medium/High pie chart)

### Predictions (`results/premium_reports/`)
- `predictions.csv` – Full scoring results from last batch prediction
- `top_premiums.csv` – Top 100 highest premiums (for underwriting review)

### Experiment Tracking (`results/experiments/`)
- Per-run JSON summaries with timestamps, dataset hashes, and hyperparameters

---

## ✅ Validation and Testing

Run the full test suite to ensure everything is working:

```bash
pytest -q
```

**Expected output:**
```
..........
10 passed in 15.99s
```

### Test Coverage

- **Data Preprocessing:** Validation, schema checking, missing value handling
- **Feature Engineering:** Normalization, scaling, encoding correctness
- **Pricing Logic:** Premium calculations, risk scoring, category assignment
- **Model Predictions:** Frequency and severity prediction interfaces and outputs

All tests are in the `tests/` directory and are designed to validate behavior without requiring retraining.

---

## 🔮 Future Improvements

- **Uncertainty Quantification:** Add confidence intervals around frequency and severity predictions
- **Drift Detection:** Monitor model performance on new data and alert when retraining is needed
- **Fairness Analysis:** Analyze pricing by demographics to ensure non-discriminatory underwriting
- **API Service:** Package as FastAPI microservice for enterprise deployment
- **Benchmark Models:** Compare Poisson/Gamma against GBM/XGBoost alternatives with SHAP explainability
- **Calibration Diagnostics:** Enhanced residual analysis and population stability indices

---

## 📖 Citation & References

This project implements actuarial pricing following industry best practices:
- Poisson regression for frequency modeling (standard in insurance)
- Gamma regression for severity modeling (standard in insurance)
- Data from: freMTPL2 French Motor Insurance Dataset (public, published research)

---

## 📝 License

This project is provided as-is for educational and research purposes.
