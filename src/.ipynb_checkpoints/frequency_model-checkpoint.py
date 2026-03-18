import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

def train_frequency_model(df):
    """
    Trains a Poisson regression model to estimate the expected number
    of accidents per driver per year.
    """
    print("Training Accident Frequency Model (Poisson regression)...")
    
    # We use past accident history as proxy for training target
    # In reality, this would be an actual past claim frequency
    # For this demo, we'll try to predict 'accidents_last_2yr' / 2.0 based on risk factors
    df_copy = df.copy()
    df_copy['target_frequency'] = df_copy['accidents_last_2yr'] / 2.0
    
    # Formula predicting frequency from risk components
    formula = "target_frequency ~ braking_score + night_driving_score + annual_mileage_normalized + age + vehicle_age"
    
    try:
        model = smf.glm(formula=formula, data=df_copy, family=sm.families.Poisson()).fit()
        print(model.summary())
        return model
    except Exception as e:
        print(f"Error training frequency model: {e}")
        return None

def predict_frequency(model, df):
    """Predicts expected accidents per year for a dataset."""
    if model is None:
        return None
    
    predictions = model.predict(df)
    return predictions

if __name__ == "__main__":
    from data_loader import load_data, preprocess_data
    from feature_engineering import create_features
    
    df = preprocess_data(load_data('../data/drivers.csv'))
    if df is not None:
        df = create_features(df)
        model = train_frequency_model(df)
        if model:
            freq = predict_frequency(model, df)
            print("Expected Accidents per year (first 5):")
            print(freq.head())
