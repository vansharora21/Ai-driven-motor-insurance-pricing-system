# Model Card: AI-Driven Motor Insurance Pricing System

This model card details the actuarial design, modeling decisions, evaluation metrics, and limitations of the machine learning pricing models trained on the `freMTPL2` dataset.

## 1. Actuarial Design Rationale

Rather than modeling the total claim cost directly, this system uses a **two-part frequency-severity decomposition**:

$$\text{Expected Loss} = \text{Claim Frequency} \times \text{Claim Severity}$$

### Why Decompose?
1. **Zero-Inflation:** Over $95\%$ of policyholders in a typical year do not file any claims. Directly modeling claim cost leads to a massive spike at zero. A single regression model (such as linear regression or basic tree models) struggles to fit this combination of a massive zero spike and a highly skewed continuous tail.
2. **Multiplicative Rating Structure:** Actuarial pricing utilizes multipliers (relativities) to determine risk. By using Generalized Linear Models (GLMs) with a **log-link function** (Poisson and Gamma distributions), the additive coefficients in the linear predictor map directly to multiplicative factors in the predictions:
   $$\log(E[Y]) = \beta_0 + \beta_1 X_1 + \dots \implies E[Y] = e^{\beta_0} \times e^{\beta_1 X_1} \times \dots$$
3. **Behavioral Differentiation:** The risk factors driving *how often* a claim occurs (frequency) are often different from the factors driving *how costly* a claim is (severity). For example, driver age heavily impacts frequency (younger drivers have more accidents), while vehicle power/weight impacts severity (high-speed collisions result in larger payouts). Decomposing the models allows underwriters to study and price these risk dynamics independently.

---

## 2. Model Specifications & Parameters

### Frequency Model
* **Algorithm:** Poisson Regression (GLM) / XGBoost Count-Poisson / Random Forest Regressor
* **Objective / Loss:** Poisson deviance
* **Link Function:** Log link
* **Target:** $\text{ClaimNb} / \text{Exposure}$ (annualized frequency)
* **Sample Weight:** $\text{Exposure}$ (policy duration in years)

### Severity Model
* **Algorithm:** Gamma Regression (GLM) / XGBoost Gamma / Random Forest Regressor
* **Objective / Loss:** Gamma deviance
* **Link Function:** Log link
* **Target:** $\text{ClaimAmount}$ (claims with positive amounts only)

---

## 3. Evaluation Metrics

To evaluate model performance, standard regression metrics like Root Mean Squared Error (RMSE) are supplemented with actuarial metrics:

### Actuarial Metrics
* **Poisson Deviance:** Measures count prediction quality. It is defined as:
  $$d(y, \hat{y}) = 2 \sum \left( y \log\left(\frac{y}{\hat{y}}\right) - (y - \hat{y}) \right)$$
  Lower values indicate the model is closer to the true claim count distribution.
* **Gamma Deviance:** Measures continuous cost prediction quality for right-skewed data. It is defined as:
  $$d(y, \hat{y}) = 2 \sum \left( \frac{y - \hat{y}}{\hat{y}} - \log\left(\frac{y}{\hat{y}}\right) \right)$$
* **Gini Coefficient & Lorenz Curve:** Measures the model's ability to **rank-order risk**. By sorting the portfolio from highest to lowest predicted risk, the Gini coefficient measures how much of the actual claim cost is captured by the highest-predicted-risk bands. A higher Gini indicates superior risk segmentation, allowing the insurer to charge competitive rates to low-risk drivers and appropriate rates to high-risk drivers.

### Why RMSE Alone is Misleading on Insurance Claims Data
Insurance claim sizes are highly right-skewed, characterized by a large number of small claims (fender-benders) and a very small number of massive, catastrophic claims (major bodily injury or total write-offs). 
* RMSE uses a **quadratic penalty** ($e^2$), making it extremely sensitive to outliers. 
* A model that predicts $99.9\%$ of the portfolio's claims accurately but misses the exact cost of a single rare, catastrophic claim will be penalized with a poor (high) RMSE.
* Actuarial pricing requires models that predict the **mean expected loss** correctly across risk cohorts rather than overfitting to individual large claims. Deviance metrics and Gini coefficients provide a more stable assessment of a pricing model's utility.

---

## 4. Dataset & Model Performance

### Dataset Summary
* **Source:** `freMTPL2` French Motor Third-Party Liability datasets (OpenML IDs 41214 and 41215).
* **Total Policy Records:** 678,013
* **Total Claim Records:** 26,444

### Baseline Holdout Performance (80/20 Train/Test Split)

* **Poisson Frequency Model:**
  * Test Sample Size: 135,603 policies
  * **RMSE:** `0.2396`
  * **Poisson Deviance:** `0.3202`
* **Gamma Severity Model:**
  * Test Sample Size: 5,289 positive claims
  * **RMSE:** `9,346.76`
  * **MAE:** `2,046.98`
  * **Gamma Deviance:** `1.5795`

---

## 5. Honest Limitations

* **Geographical & Temporal Context:** The dataset is based on French policyholders from **2013**. Driving patterns, repair costs, inflation, and vehicle safety features have changed significantly since then. These models cannot be deployed directly to modern markets without recalibration.
* **Lack of Live Fairness Auditing:** In insurance pricing, using variables like geographic region (`Area`/`Region`), age (`DrivAge`), or vehicle characteristics can act as proxies for protected or sensitive classes (e.g. socio-economic status, gender, race). While European regulations explicitly govern variables like gender in pricing, this model has not undergone a formal algorithmic fairness audit (e.g. checking for disparate impact or equalized odds).
* **Exposure Clamping:** The model clamps exposure values to a positive floor to prevent division-by-zero errors. This can slightly skew results for very short-term policies (e.g. policies active for only a few hours).
* **Extreme Catastrophic Claims:** Standard GLMs struggle to model extreme tail events (e.g. claims exceeding €100,000). A separate reinsurance threshold or extreme value theory (EVT) model is typically required in production to cap severity predictions.
