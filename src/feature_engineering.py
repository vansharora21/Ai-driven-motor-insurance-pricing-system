import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_features(df):
    """Engineers new features and calculates driver risk score."""
    if df is None:
        return None
        
    df = df.copy()
    
    # 1. Annual Mileage Estimate
    df['annual_mileage'] = df['daily_mileage'] * 365
    
    # 2. Extract scoring components
    scaler = MinMaxScaler()
    df['annual_mileage_normalized'] = scaler.fit_transform(df[['annual_mileage']])
    
    # Map encoded categorical levels to a 0-1 scale for the score
    df['braking_score'] = df['harsh_braking_encoded'] / 3.0
    df['night_driving_score'] = df['night_driving_encoded'] / 3.0
    
    # Accident history needs to be capped/scaled (e.g., maxing out penalization at 3 accidents)
    df['accident_history_score'] = np.clip(df['accidents_last_2yr'] / 3.0, 0, 1)
    
    # 3. Calculate Driver Risk Score (Formula from requirements)
    df['driver_risk_index'] = (
        0.35 * df['braking_score'] +
        0.25 * df['night_driving_score'] +
        0.25 * df['annual_mileage_normalized'] +
        0.15 * df['accident_history_score']
    )
    
    # Discretize risk score into categories for easier use
    df['risk_category'] = pd.qcut(df['driver_risk_index'], q=3, labels=['Low', 'Medium', 'High'])
    
    print("Feature engineering completed.")
    return df

if __name__ == "__main__":
    # Test block
    from data_loader import load_data, preprocess_data
    df = preprocess_data(load_data('../data/drivers.csv'))
    if df is not None:
        featured_df = create_features(df)
        print(featured_df[['driver_id', 'driver_risk_index', 'risk_category']].head())
