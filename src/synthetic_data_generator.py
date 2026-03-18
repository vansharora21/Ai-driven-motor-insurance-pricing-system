import pandas as pd
from sdv.lite import SingleTablePreset
from sdv.metadata import SingleTableMetadata
import warnings
warnings.filterwarnings('ignore')

def generate_synthetic_data(real_data_path='data/drivers.csv', output_path='data/synthetic_drivers.csv', num_rows=1000):
    """
    Uses SDV (Synthetic Data Vault) to generate realistic tabular variations
    matching the distribution of the original dataset.
    """
    print(f"Loading real data from {real_data_path} to synthesize {num_rows} new records...")
    try:
        real_data = pd.read_csv(real_data_path)
    except FileNotFoundError:
        print("Original data not found. Cannot generate synthetic data.")
        return None
        
    # Create metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)
    
    # Configure the primary key properly if detected
    metadata.update_column(
        column_name='driver_id',
        sdtype='id',
        regex_format='D[A-Z0-9]{5}'
    )

    print("Training synthetic model structure...")
    synthesizer = SingleTablePreset(metadata, name='FAST_ML')
    synthesizer.fit(real_data)
    
    print("Generating synthetic rows...")
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    
    # Post-process to ensure business logic constraints
    synthetic_data['age'] = synthetic_data['age'].clip(18, 90).astype(int)
    synthetic_data['vehicle_age'] = synthetic_data['vehicle_age'].clip(0, 30).astype(int)
    synthetic_data['daily_mileage'] = synthetic_data['daily_mileage'].clip(1, 500).astype(int)
    synthetic_data['accidents_last_2yr'] = synthetic_data['accidents_last_2yr'].clip(0, 10).astype(int)
    
    # Save to disk
    synthetic_data.to_csv(output_path, index=False)
    print(f"Successfully generated {num_rows} synthetic drivers. Saved to {output_path}")
    
    return synthetic_data

if __name__ == "__main__":
    generate_synthetic_data(num_rows=1000)
