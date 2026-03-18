import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def inject_fraud(df, fraud_rate=0.05):
    """
    Injects synthetic fraud cases to test resilience.
    E.g., high repair costs (expected severity anomaly) combined with high frequency.
    """
    np.random.seed(42)
    df_fraud = df.copy()
    
    num_frauds = int(len(df_fraud) * fraud_rate)
    fraud_indices = np.random.choice(df_fraud.index, num_frauds, replace=False)
    
    # Simulate fraud scenarios (e.g. repeated claims + high severity on old vehicles)
    df_fraud.loc[fraud_indices, 'simulated_severity'] = np.random.uniform(200000, 500000, num_frauds)
    df_fraud.loc[fraud_indices, 'accidents_last_2yr'] = np.random.randint(3, 7, num_frauds)
    df_fraud.loc[fraud_indices, 'is_fraud'] = 1
    
    # Mark legitimate ones
    df_fraud['is_fraud'] = df_fraud['is_fraud'].fillna(0).astype(int)
    
    # Assign normal random severity to non-frauds that had claims
    non_frauds = df_fraud[df_fraud['is_fraud'] == 0]
    claimants = non_frauds[non_frauds['accidents_last_2yr'] > 0].index
    df_fraud.loc[claimants, 'simulated_severity'] = np.random.normal(30000, 10000, len(claimants))
    df_fraud['simulated_severity'] = df_fraud['simulated_severity'].fillna(0)
    
    return df_fraud

def detect_anomalies(df):
    """
    Builds an Isolation Forest to flag suspicious behavior.
    """
    print("Running anomaly detection (Isolation Forest)...")
    
    # Features indicative of fraud for modeling purposes
    # Since simulated severity and accidents are the injected fraud features
    features = ['accidents_last_2yr', 'simulated_severity', 'vehicle_age']
    
    # Train unsupervised contamination model
    clf = IsolationForest(contamination=0.05, random_state=42)
    
    # Only fit on those with non-zero severity (actual claims)
    claim_df = df[df['simulated_severity'] > 0].copy()
    
    if len(claim_df) < 5:
        print("Not enough claims to train fraud detector.")
        df['anomaly_flag'] = 0
        return df, None
        
    X = claim_df[features].fillna(0)
    clf.fit(X)
    
    # Predict (-1 is anomaly, 1 is normal -> remap to 1=anomaly, 0=normal)
    preds = clf.predict(X)
    claim_df['anomaly_flag'] = np.where(preds == -1, 1, 0)
    
    # Merge back
    df['anomaly_flag'] = 0
    df.update(claim_df[['anomaly_flag']])
    
    return df, clf

if __name__ == "__main__":
    from data_loader import load_data, preprocess_data
    
    # 1. Load data
    df = preprocess_data(load_data('../data/drivers.csv'))
    
    # 2. Inject some artificial claim severities to demonstrate fraud injection
    df_injected = inject_fraud(df)
    
    # 3. Detect
    df_detected, model = detect_anomalies(df_injected)
    
    frauds_found = df_detected[df_detected['anomaly_flag'] == 1]
    print(f"\nDetected {len(frauds_found)} anomalous claims.")
    print(frauds_found[['driver_id', 'accidents_last_2yr', 'simulated_severity', 'is_fraud', 'anomaly_flag']].head())
