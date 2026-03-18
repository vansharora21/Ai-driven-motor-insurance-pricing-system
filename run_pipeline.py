import sys
import os

sys.path.append('src')

from data_loader import load_data, preprocess_data
from feature_engineering import create_features
from frequency_model import train_frequency_model, predict_frequency
from severity_model import train_severity_model, predict_severity
from fraud_detection import inject_fraud, detect_anomalies
from pricing_engine import calculate_premium
from simulation import simulate_urban_traffic
from visualization import plot_risk_distribution, plot_premium_distribution, plot_scenario_comparison, plot_fraud_anomalies, generate_reports

def run_pipeline():
    print("--- GenAI Motor Insurance Pricing Pipeline ---")
    print("1. Loading Data...")
    df_raw = load_data('data/drivers.csv')
    print(f"Loaded {len(df_raw)} drivers.")
    
    df = preprocess_data(df_raw)
    
    print("2. Feature Engineering...")
    df_features = create_features(df)
    
    print("3. Model Training...")
    freq_model = train_frequency_model(df_features)
    sev_model = train_severity_model(df_features)
    
    df_features['expected_frequency'] = predict_frequency(freq_model, df_features)
    df_features['expected_severity'] = predict_severity(sev_model, df_features)
    
    print("4. Calculating Premiums...")
    df_premium = calculate_premium(df_features, df_features['expected_frequency'], df_features['expected_severity'])
    
    print("5. Generating Visualizations...")
    os.makedirs('results/plots', exist_ok=True)
    plot_risk_distribution(df_premium)
    plot_premium_distribution(df_premium)
    generate_reports(df_premium)
    print("Visualizations and reports saved.")
    
    print("6. Scenario Simulation...")
    df_sim = simulate_urban_traffic(df_features, scenario_name="Jaipur High Congestion")
    df_sim['expected_frequency'] = predict_frequency(freq_model, df_sim)
    df_sim['expected_severity'] = predict_severity(sev_model, df_sim)
    df_sim_premium = calculate_premium(df_sim, df_sim['expected_frequency'], df_sim['expected_severity'])
    plot_scenario_comparison(df_premium, df_sim_premium, metric='final_premium')
    print("Simulation complete.")
    
    print("7. Fraud Detection...")
    df_fraud = inject_fraud(df_premium)
    df_detected, model = detect_anomalies(df_fraud)
    
    # Depending on how the flag is implemented, either 1 or -1 is anomaly
    if 'anomaly_flag' in df_detected.columns:
        frauds_count = len(df_detected[df_detected['anomaly_flag'].isin([1, -1])])
        print(f"Detected {frauds_count} anomalies.")
        if frauds_count > 0:
            plot_fraud_anomalies(df_detected)
            
    print("--- Pipeline Complete ---")

if __name__ == '__main__':
    run_pipeline()
