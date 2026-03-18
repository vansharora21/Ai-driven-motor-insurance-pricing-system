import pandas as pd
import numpy as np

def simulate_urban_traffic(df, scenario_name="Jaipur High Congestion"):
    """
    Modifies driver behavior metrics to simulate an intense urban 
    traffic environment with higher braking frequency and accidents.
    """
    print(f"Applying scenario constraints: {scenario_name}")
    df_sim = df.copy()
    
    # In High Congestion:
    # 1. Daily mileage slightly reduces (slower average speed)
    df_sim['daily_mileage'] = df_sim['daily_mileage'] * np.random.uniform(0.7, 0.9, len(df))
    
    # 2. Harsh braking events increase drastically due to stop-and-go traffic
    # Elevate everyone's encoded braking score by 1 level if not already max
    if 'harsh_braking_encoded' in df_sim.columns:
        df_sim['harsh_braking_encoded'] = df_sim['harsh_braking_encoded'].apply(lambda x: min(x + 1, 3))
    
    # 3. Accident probability increases
    # We'll artificially bump the historical accidents context for the simulation
    # to feed into the models as a stressed dataset.
    extra_accidents = np.random.poisson(0.5, len(df))
    df_sim['accidents_last_2yr'] += extra_accidents
    
    return df_sim

if __name__ == "__main__":
    from data_loader import load_data, preprocess_data
    
    df = preprocess_data(load_data('../data/drivers.csv'))
    if df is not None:
        baseline_braking = df['harsh_braking_encoded'].mean()
        baseline_accidents = df['accidents_last_2yr'].mean()
        
        df_sim = simulate_urban_traffic(df)
        
        print("\n=== Scenario Simulation Impact ===")
        print(f"Avg Harsh Braking Level: {baseline_braking:.2f} -> {df_sim['harsh_braking_encoded'].mean():.2f}")
        print(f"Avg Accidents (last 2yr): {baseline_accidents:.2f} -> {df_sim['accidents_last_2yr'].mean():.2f}")
