import pandas as pd
import numpy as np
import random

def generate_mock_data(num_samples=100):
    np.random.seed(42)
    random.seed(42)
    
    data = []
    for i in range(1, num_samples + 1):
        driver_id = f"D{i:03d}"
        age = np.random.randint(18, 75)
        vehicle_type = random.choice(['Sedan', 'SUV', 'Hatchback', 'Truck'])
        vehicle_age = np.random.randint(0, 20)
        daily_mileage = np.random.randint(5, 150)
        night_driving_level = random.choice(['Low', 'Medium', 'High'])
        harsh_braking_level = random.choice(['Low', 'Medium', 'High'])
        accidents_last_2yr = np.random.poisson(0.3)
        claim_history = random.choice([0, 1]) if accidents_last_2yr > 0 else 0
        
        data.append({
            'driver_id': driver_id,
            'age': age,
            'vehicle_type': vehicle_type,
            'vehicle_age': vehicle_age,
            'daily_mileage': daily_mileage,
            'night_driving_level': night_driving_level,
            'harsh_braking_level': harsh_braking_level,
            'accidents_last_2yr': accidents_last_2yr,
            'claim_history': claim_history
        })
        
    df = pd.DataFrame(data)
    df.to_csv('data/drivers.csv', index=False)
    print(f"Generated {num_samples} mock driver records in data/drivers.csv")

if __name__ == "__main__":
    generate_mock_data()
