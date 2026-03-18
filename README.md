<h1 align="center">🚗 GenAI Motor Insurance Pricing Engine</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-Streamlit-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/Machine%20Learning-Actuarial%20Science-green.svg" alt="ML Models">
</p>

## 📖 Overview

The **GenAI Motor Insurance Pricing Engine** is a state-of-the-art actuarial pipeline designed to calculate dynamic, fair, and highly personalized motor insurance premiums. Moving away from static, demographic-driven pricing matrices, this engine ingests driver behavioral data (telematics) to directly compute statistical expected loss.

### ✨ Key Features
* 📊 **Actuarial Machine Learning**: Utilizes **Poisson regression** to predict accident frequency and a **Gamma distribution** model to forecast claim severity.
* 🤖 **Unsupervised Fraud Detection**: Employs an **Isolation Forest** to autonomously flag anomalies and mathematically highly unlikely risk vs. premium assignments.
* 🌐 **Interactive Web Dashboard**: Features a full **Streamlit UI** allowing you to input user behaviors on the fly, process batch CSV files, and generate interactive visual reports.
* 🏙️ **Dynamic Scenario Stress Testing**: Run programmatic simulations (e.g., "High Urban Congestion") to observe how environmental shifts dynamically recalculate total portfolio reserves.
* 🧬 **Synthetic Data Layer (SDV)**: Generate statistically identical, privacy-compliant synthetic populations based on real base-data.

---

## 🚀 Live Web Application (Streamlit)

You can launch the integrated frontend directly to interface with the pricing models.

### How to Run Locally

```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Launch the Streamlit server
streamlit run app.py
```
*The app will automatically open in your browser at `http://localhost:8501`. Here you can calculate premiums for single drivers, or upload a CSV in the "Batch Risk Analyzer" tab.*

---

## 🏗️ Project Architecture

```
genai-insurance-pricing/
├── app.py                           # 🟢 Streamlit Web Application Interface
├── data/
│   └── drivers.csv                  # 📂 Input telematics cohort data
├── notebooks/
│   └── analysis.ipynb               # 📓 Jupyter execution flow
├── results/
│   ├── plots/                       # 📈 Auto-rendered distribution & anomaly PNGs
│   └── premium_reports/             # 📑 Final output of priced drivers
├── src/
│   ├── feature_engineering.py       # ⚙️ Indexing models & behavioral weights
│   ├── frequency_model.py           # 🧠 Poisson count regression
│   ├── severity_model.py            # 🧠 Gamma continuous distribution
│   ├── pricing_engine.py            # 💰 Final load logic & premium computation
│   ├── simulation.py                # 🏙️ Environmental stress testing modules
│   ├── fraud_detection.py           # 🕵️ Isolation forest analyzer
│   └── visualization.py             # 🖼️ Matplotlib/Seaborn visualization layer
├── generate_mock_data.py            # 🛠️ Utility script for building base SDV distributions
├── run_pipeline.py                  # 🚀 Fast CLI tool to execute the whole stack headless
└── requirements.txt                 # 📦 Pinned ML environment variables
```

---

## 🛠️ Usage via CLI (Headless Mode)

If you wish to bypass the GUI and run the pricing models across the entire `drivers.csv` database, dumping the generated CSV reports and PNGs directly to the `results/` folder:

```bash
# Execute the entire script headless:
python run_pipeline.py
```

---

## 💳 Data Schema (Using Your Own Data)

To test the engine with custom cohorts, ensure your input CSV headers exactly match:
`driver_id, age, vehicle_type, vehicle_age, daily_mileage, night_driving_level, harsh_braking_level, accidents_last_2yr, claim_history`

*Note: The Streamlit **Batch Risk Analyzer** tab handles the injection and cleanup of custom CSV inputs natively.*
