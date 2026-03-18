import pandas as pd
import numpy as np
import re
import os

input_file = r"c:\Users\BIT\Desktop\research 101\genai-insurance-pricing\data\Driver Behavior Survey for AI Motor Insurance Research (Responses) - Form responses 1.csv"
output_file = r"c:\Users\BIT\Desktop\research 101\genai-insurance-pricing\data\drivers.csv"

# Read the CSV
df = pd.read_csv(input_file)

out = pd.DataFrame()

# 1. Driver ID
out['driver_id'] = [f"D{i+1:03d}" for i in range(len(df))]

# 2. Driver Age
# Find the Age column
col_age = [c for c in df.columns if 'Age' in c and 'between 18' in c][0]
out['age'] = pd.to_numeric(df[col_age], errors='coerce').fillna(30).astype(int)

# 3. Vehicle Type
col_vtype = [c for c in df.columns if 'Vehicle Type' in c][0]
out['vehicle_type'] = df[col_vtype].fillna('Unknown')

# 4. Vehicle Age
col_vage = [c for c in df.columns if 'How old is your vehicle' in c][0]
def parse_num(val):
    if pd.isna(val) or str(val).lower().strip() == '':
        return 0.0
    val = str(val).lower()
    if 'month' in val:
        nums = re.findall(r'[\d\.]+', val)
        if nums: return float(nums[0]) / 12.0
    if '-' in val:
        parts = val.split('-')
        n1 = re.findall(r'[\d\.]+', parts[0])
        n2 = re.findall(r'[\d\.]+', parts[1])
        if n1 and n2:
            return (float(n1[0]) + float(n2[0])) / 2.0
    nums = re.findall(r'[\d\.]+', val)
    if nums:
        return float(nums[0])
    return 0.0

out['vehicle_age'] = df[col_vage].apply(parse_num)

# 5. Daily Mileage
col_mileage = [c for c in df.columns if 'kilometers do you drive per day' in c][0]
out['daily_mileage'] = df[col_mileage].apply(parse_num)

# 6. Night Driving Frequency
col_night = [c for c in df.columns if 'Night Driving' in c][0]
def parse_level(val):
    if pd.isna(val): return 'Low'
    val = str(val)
    if 'Low' in val: return 'Low'
    if 'Medium' in val: return 'Medium'
    if 'High' in val: return 'High'
    return 'Low'
out['night_driving_level'] = df[col_night].apply(parse_level)

# 7. Harsh Braking Frequency
col_brake = [c for c in df.columns if 'Harsh Braking' in c][0]
out['harsh_braking_level'] = df[col_brake].apply(parse_level)

# 8. Accident History
col_acc = [c for c in df.columns if 'traffic accidents' in c][0]
def parse_acc(val):
    if pd.isna(val): return 0
    val = str(val).lower()
    if val in ['none', 'nil', 'never']: return 0
    if '-' in val:
        parts = val.split('-')
        n1 = re.findall(r'\d+', parts[0])
        n2 = re.findall(r'\d+', parts[1])
        if n1 and n2:
            return int((int(n1[0]) + int(n2[0])) / 2)
    nums = re.findall(r'\d+', val)
    if nums:
        return int(nums[0])
    return 0

out['accidents_last_2yr'] = df[col_acc].apply(parse_acc)

# 9. Claim History
col_claim = [c for c in df.columns if 'insurance claim' in c][0]
out['claim_history'] = df[col_claim].apply(lambda x: 1 if str(x).lower().strip() == 'yes' else 0)

out.to_csv(output_file, index=False)
print(f"Successfully processed {len(out)} rows into {output_file}")
