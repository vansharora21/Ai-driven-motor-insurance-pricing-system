# Google Form Design for Live Motor Insurance Pricing

The current Streamlit app can now read the newest CSV in the `data/` folder, map the Google Form responses into the pricing model's actuarial input schema, and show the premium proof chart directly in the UI.

Use the following Google Form fields so the export matches the live bridge logic:

### 1. Driver Identifier
* **Form Field:** `Driver Identifier (Email or Unique ID)`
* **Type:** Short answer
* **Purpose:** Traceability in the live pricing output

### 2. Email Address
* **Form Field:** `Email address`
* **Type:** Short answer
* **Purpose:** Optional contact field for the response export

### 3. Driver Age
* **Form Field:** `Age (Must be between 18 and 100)`
* **Type:** Number
* **Purpose:** Mapped to `DrivAge`

### 4. Vehicle Type
* **Form Field:** `Vehicle Type`
* **Type:** Multiple choice
* **Options:** Sedan, SUV, Hatchback, Truck, Motorcycle, Other
* **Purpose:** Mapped into proxy pricing inputs such as `VehPower`, `VehBrand`, and `VehGas`

### 5. Vehicle Age
* **Form Field:** `How old is your vehicle (in years)?`
* **Type:** Number
* **Purpose:** Mapped to `VehAge`

### 6. Daily Mileage
* **Form Field:** `On average, how many kilometers do you drive per day?`
* **Type:** Number
* **Purpose:** Used as a proxy signal for `Exposure` and `Density`

### 7. Night Driving Frequency
* **Form Field:** `Night Driving Frequency (10 PM – 5 AM)`
* **Type:** Multiple choice
* **Options:** Low (Rarely), Medium (Occasionally), High (Frequently)
* **Purpose:** Used in the proxy pricing bridge

### 8. Harsh Braking Frequency
* **Form Field:** `Harsh Braking Frequency (How often do you brake abruptly in traffic?)`
* **Type:** Multiple choice
* **Options:** Low (Rarely), Medium (Occasionally), High (Frequently)
* **Purpose:** Used in the proxy pricing bridge

### 9. Accident History
* **Form Field:** `How many traffic accidents have you been involved in during the last 2 years?`
* **Type:** Number
* **Purpose:** Mapped into `BonusMalus` and `Area` proxies

### 10. Claim History
* **Form Field:** `Have you filed a motor insurance claim in the last 2 years?`
* **Type:** Multiple choice
* **Options:** Yes, No
* **Purpose:** Mapped into `BonusMalus` and `Area` proxies

### 11. Driving Experience
* **Form Field:** `How many years of driving experience do you have?`
* **Type:** Number
* **Purpose:** Used to adjust the proxy pricing bridge

---

### How the live scoring works
1. Export the Google Form responses as a CSV.
2. Place the file in the `data/` folder.
3. Open the Streamlit app and use the `Live Google Form Data` tab.
4. The app reads the latest CSV, maps survey answers into pricing proxies, and displays the premium waterfall chart plus the final premium output.

### Important note
The live bridge is a proxy transformation so that the form responses can be scored by the current actuarial model. It is not the same as retraining the model directly on the survey data.
