import pandas as pd
import numpy as np

def calculate_premium(df, expected_frequency, expected_severity, base_rate=3000):
    """
    Calculates dynamic personalized insurance premiums.
    """
    df_premium = df.copy()
    
    # Core Components
    df_premium['base_premium'] = base_rate
    
    # Handle potentially missing predictions if models failed
    if expected_frequency is None:
        df_premium['expected_frequency'] = 0.15 # fallback
    else:
        df_premium['expected_frequency'] = expected_frequency
        
    if expected_severity is None:
        df_premium['expected_severity'] = 25000 # fallback
    else:
        df_premium['expected_severity'] = expected_severity
        
    # Calculate Expected Loss
    df_premium['expected_loss'] = df_premium['expected_frequency'] * df_premium['expected_severity']
    
    # Risk Adjustment
    # Assumes driver_risk_index (0.0 to 1.0) was calculated in feature_engineering
    risk_multiplier = 5000  # max additional penalty
    df_premium['risk_adjustment'] = df_premium['driver_risk_index'] * risk_multiplier
    
    # Safe Driver Discount
    # E.g., no accidents and low braking score gets a discount
    df_premium['safe_driver_discount'] = np.where(
        (df_premium['accidents_last_2yr'] == 0) & (df_premium['braking_score'] < 0.4),
        0.15 * df_premium['base_premium'], # 15% off base
        0
    )
    
    # Final Calculation
    df_premium['final_premium'] = (
        df_premium['base_premium'] + 
        df_premium['expected_loss'] + 
        df_premium['risk_adjustment'] - 
        df_premium['safe_driver_discount']
    )
    
    # Cap minimum premium
    df_premium['final_premium'] = df_premium['final_premium'].clip(lower=1500)
    
    return df_premium

if __name__ == "__main__":
    from data_loader import load_data, preprocess_data
    from feature_engineering import create_features
    
    df = preprocess_data(load_data('../data/drivers.csv'))
    
    if df is not None:
        df = create_features(df)
        
        # Mock predictions for testing the logic independently
        mock_freq = pd.Series(np.random.uniform(0.01, 0.4, len(df)), index=df.index)
        mock_sev = pd.Series(np.random.uniform(10000, 50000, len(df)), index=df.index)
        
        premium_df = calculate_premium(df, mock_freq, mock_sev)
        
        print(premium_df[['driver_id', 'risk_category', 'final_premium']].head())
