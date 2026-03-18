# Google Form Design for Motor Insurance Pricing

If you are collecting real driver behavior data through a Google Form, you need to ask questions that perfectly map to the expected CSV input structure of our pipeline. 

Here are the fields you should create in your Google Form:

### 1. Driver ID (Auto-generated or Email)
* **Form Field:** "Email Address" or "Name (will be anonymized)"
* **Type:** Short Answer
* **Mapping:** `driver_id` (You can hash or replace this with 'D001', 'D002' when exporting)

### 2. Driver Age
* **Form Field:** "What is your age?"
* **Type:** Number (add validation: Must be between 18 and 100)
* **Mapping:** `age`

### 3. Vehicle Type
* **Form Field:** "What type of vehicle do you primarily drive?"
* **Type:** Multiple Choice
* **Options:** Sedan, SUV, Hatchback, Truck
* **Mapping:** `vehicle_type`

### 4. Vehicle Age
* **Form Field:** "How old is your vehicle (in years)?"
* **Type:** Number (add validation: Must be greater than or equal to 0)
* **Mapping:** `vehicle_age`

### 5. Daily Mileage
* **Form Field:** "On average, how many kilometers do you drive per day?"
* **Type:** Number
* **Mapping:** `daily_mileage`

### 6. Night Driving Frequency
* **Form Field:** "How often do you drive late at night (between 10 PM and 5 AM)?"
* **Type:** Multiple Choice
* **Options:** Low (Rarely), Medium (Occasionally), High (Frequently)
* **Mapping:** `night_driving_level` (Make sure to export just the 'Low', 'Medium', 'High' prefixes)

### 7. Harsh Braking Frequency
* **Form Field:** "How often do you find yourself braking abruptly or harshly in traffic?"
* **Type:** Multiple Choice
* **Options:** Low (Rarely), Medium (Occasionally), High (Frequently)
* **Mapping:** `harsh_braking_level` (Make sure to export just the 'Low', 'Medium', 'High' prefixes)

### 8. Accident History
* **Form Field:** "How many traffic accidents have you been involved in over the past 2 years?"
* **Type:** Number (add validation: Must be a whole number, e.g., 0, 1, 2)
* **Mapping:** `accidents_last_2yr`

### 9. Claim History
* **Form Field:** "Have you filed a motor insurance claim in the last 2 years?"
* **Type:** Multiple Choice
* **Options:** Yes, No
* **Mapping:** `claim_history` (When exporting, map "No" -> 0 and "Yes" -> 1)

---

### How to use the exported data:
1. Download the Google Form responses as a CSV.
2. Rename the columns exactly to: `driver_id`, `age`, `vehicle_type`, `vehicle_age`, `daily_mileage`, `night_driving_level`, `harsh_braking_level`, `accidents_last_2yr`, `claim_history`.
3. Clean the `claim_history` column to be `0` or `1`.
4. Place it in the `data/` folder as `drivers.csv` and run the pipeline!
