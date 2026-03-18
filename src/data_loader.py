import pandas as pd
import numpy as np

def load_data(filepath='data/drivers.csv'):
    """Loads the driver dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None

def preprocess_data(df):
    """Cleans and preprocesses the dataset."""
    if df is None:
        return None
        
    # Handle missing values (simple fill for demo purposes)
    df.fillna({
        'age': df['age'].median(),
        'vehicle_age': df['vehicle_age'].median(),
        'daily_mileage': df['daily_mileage'].median()
    }, inplace=True)
    
    # Categorical encoding mappings
    level_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
    
    df['night_driving_encoded'] = df['night_driving_level'].map(level_mapping).fillna(1)
    df['harsh_braking_encoded'] = df['harsh_braking_level'].map(level_mapping).fillna(1)
    
    # One-hot encode vehicle_type
    df = pd.get_dummies(df, columns=['vehicle_type'], drop_first=False)
    
    # Standardize column naming for boolean
    for col in df.columns:
        if col.startswith('vehicle_type_'):
            df[col] = df[col].astype(int)
            
    print("Data preprocessing completed.")
    return df

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        processed_df = preprocess_data(df)
        print(processed_df.head())
