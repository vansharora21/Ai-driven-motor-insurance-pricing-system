import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()

    text_intro = """# GenAI Motor Insurance Pricing Pipeline Analysis
This notebook runs the full pipeline for data ingestion, feature engineering, modeling, fraud detection, and scenario simulation for urban traffic conditions."""

    code_imports = """import sys
sys.path.append('../src')
from data_loader import load_data, preprocess_data
from feature_engineering import create_features
from frequency_model import train_frequency_model, predict_frequency
from severity_model import train_severity_model, predict_severity
from synthetic_data_generator import generate_synthetic_data
from fraud_detection import inject_fraud, detect_anomalies
from pricing_engine import calculate_premium
from simulation import simulate_urban_traffic
from visualization import plot_risk_distribution, plot_premium_distribution, plot_scenario_comparison, plot_fraud_anomalies, generate_reports"""

    code_pipeline = """# 1. Load and Preprocess Data
df_raw = load_data('../data/drivers.csv')
df = preprocess_data(df_raw)

# 2. Feature Engineering
df_features = create_features(df)
df_features.head()"""

    code_models = """# 3. Model Training & Prediction
freq_model = train_frequency_model(df_features)
sev_model = train_severity_model(df_features)

df_features['expected_frequency'] = predict_frequency(freq_model, df_features)
df_features['expected_severity'] = predict_severity(sev_model, df_features)"""

    code_pricing = """# 4. Calculate Premiums
df_premium = calculate_premium(df_features, df_features['expected_frequency'], df_features['expected_severity'])
df_premium[['driver_id', 'risk_category', 'final_premium']].head()"""

    code_visuals = """# 5. Visualizations
plot_risk_distribution(df_premium)
plot_premium_distribution(df_premium)
generate_reports(df_premium)
print("Plots generated in results/plots")"""

    code_simulation = """# 6. Scenario Simulation (Jaipur Traffic)
df_sim = simulate_urban_traffic(df_features, scenario_name="Jaipur High Congestion")

# Recalculate predictions and premiums for simulation
df_sim['expected_frequency'] = predict_frequency(freq_model, df_sim)
df_sim['expected_severity'] = predict_severity(sev_model, df_sim)
df_sim_premium = calculate_premium(df_sim, df_sim['expected_frequency'], df_sim['expected_severity'])

plot_scenario_comparison(df_premium, df_sim_premium, metric='final_premium')
print("Scenario simulation completed. Comparison plot generated.")"""

    code_fraud = """# 7. Fraud Detection
df_fraud = inject_fraud(df_premium)
df_detected, model = detect_anomalies(df_fraud)

frauds = df_detected[df_detected['anomaly_flag'] == 1]
print(f"Detected {len(frauds)} anomalies/frauds.")
if len(frauds) > 0:
    plot_fraud_anomalies(df_detected)"""

    nb['cells'] = [
        nbf.v4.new_markdown_cell(text_intro),
        nbf.v4.new_code_cell(code_imports),
        nbf.v4.new_code_cell(code_pipeline),
        nbf.v4.new_code_cell(code_models),
        nbf.v4.new_code_cell(code_pricing),
        nbf.v4.new_code_cell(code_visuals),
        nbf.v4.new_code_cell(code_simulation),
        nbf.v4.new_code_cell(code_fraud)
    ]

    with open('notebooks/analysis.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Notebook 'notebooks/analysis.ipynb' created successfully.")

if __name__ == '__main__':
    create_notebook()
