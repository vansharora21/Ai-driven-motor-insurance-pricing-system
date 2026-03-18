import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

def train_severity_model(df):
    """
    Models the financial cost of accidents using a Gamma distribution.
    """
    print("Training Claim Severity Model (Gamma regression)...")
    
    # Generate mock claim costs for training based on requirements
    # minor: ~10,000, moderate: ~40,000, major: ~120,000
    df_copy = df.copy()
    
    # Ensure strictly positive target for Gamma distribution
    np.random.seed(42)
    base_cost = np.where(df_copy['accidents_last_2yr'] > 0, 
                         np.random.choice([10000, 40000, 120000], size=len(df), p=[0.6, 0.3, 0.1]),
                         1)  # using 1 instead of 0 for Gamma
                         
    df_copy['claim_severity_target'] = base_cost
    
    # We only train on drivers with actual claims/accidents
    train_df = df_copy[df_copy['claim_severity_target'] > 1].copy()
    
    if len(train_df) < 10:
        print("Warning: Insufficient claim data to train severity model robustly. Using dummy model.")
        return 'dummy_model'
        
    # Example formula for training based on vehicle characteristics
    # Adjust variables as needed based on actual one-hot encoded columns
    vehicle_cols = [c for c in train_df.columns if c.startswith('vehicle_type_')]
    if vehicle_cols:
        formula = f"claim_severity_target ~ vehicle_age + {' + '.join(vehicle_cols)}"
    else:
        formula = "claim_severity_target ~ vehicle_age"
        
    try:
        model = smf.glm(formula=formula, data=train_df, family=sm.families.Gamma(link=sm.families.links.log())).fit()
        print(model.summary())
        return model
    except Exception as e:
        print(f"Error training severity model: {e}")
        return 'dummy_model'

def predict_severity(model, df):
    """Predicts expected claim severity."""
    if model == 'dummy_model':
        # Default fallback mechanism
        return pd.Series(np.random.normal(30000, 5000, len(df)), index=df.index)
        
    if model is None:
        return None
    
    try:
        predictions = model.predict(df)
        return predictions
    except Exception as e:
        print(f"Prediction failed: {e}. Returning baseline estimates.")
        return pd.Series(np.random.normal(30000, 5000, len(df)), index=df.index)

if __name__ == "__main__":
    from data_loader import load_data, preprocess_data
    from feature_engineering import create_features
    
    df = preprocess_data(load_data('../data/drivers.csv'))
    if df is not None:
        df = create_features(df)
        model = train_severity_model(df)
        if model:
            sev = predict_severity(model, df)
            print("Expected Claim Severity in INR (first 5):")
            print(sev.head())
