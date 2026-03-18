import streamlit as st
import pandas as pd
import sys
import os

sys.path.append('src')

from data_loader import load_data, preprocess_data
from feature_engineering import create_features
from frequency_model import train_frequency_model, predict_frequency
from severity_model import train_severity_model, predict_severity
from pricing_engine import calculate_premium

# --- Page Config ---
st.set_page_config(page_title="GenAI Motor Insurance Pricing", layout="wide", page_icon="🚗")

# --- Load Models & Data ---
@st.cache_resource
def load_and_train_models():
    # Load dataset
    df_raw = load_data('data/drivers.csv')
    df = preprocess_data(df_raw)
    df_features = create_features(df)
    
    # Train Models
    freq_model = train_frequency_model(df_features)
    sev_model = train_severity_model(df_features)
    
    return freq_model, sev_model

with st.spinner("Loading GenAI Pricing Engine and Models..."):
    freq_model, sev_model = load_and_train_models()

# --- UI Layout ---
st.title("🚗 GenAI Motor Insurance Pricing Engine")
st.markdown("Predict dynamic, personalized premiums using actuarial ML models based on driver behavior and historical claims.")

tab_calc, tab_upload, tab_portfolio = st.tabs(["💰 Premium Calculator", "📂 Batch Risk Analyzer", "📊 Portfolio Analytics"])

with tab_calc:
    st.header("Driver Profile Input")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Driver Age", min_value=18, max_value=100, value=30)
        vehicle_type = st.selectbox("Vehicle Type", ["Sedan", "SUV", "Hatchback", "Truck", "Motorcycle", "Other"])
        vehicle_age = st.number_input("Vehicle Age (years)", min_value=0.0, max_value=50.0, value=5.0)
        
    with col2:
        daily_mileage = st.number_input("Daily Mileage (km)", min_value=0.0, max_value=1000.0, value=25.0)
        night_driving = st.selectbox("Night Driving Frequency", ["Low", "Medium", "High"])
        harsh_braking = st.selectbox("Harsh Braking Frequency", ["Low", "Medium", "High"])
        
    with col3:
        accidents = st.number_input("Accidents in Last 2 Years", min_value=0, max_value=20, value=0)
        claims = st.selectbox("Filed Claim in Last 2 Years?", ["No", "Yes"])
        claim_val = 1 if claims == "Yes" else 0
        
    if st.button("Calculate Personalized Premium", type="primary", use_container_width=True):
        # Create a single row dataframe
        input_data = pd.DataFrame([{
            'driver_id': 'NEW_DRIVER',
            'age': age,
            'vehicle_type': vehicle_type,
            'vehicle_age': vehicle_age,
            'daily_mileage': daily_mileage,
            'night_driving_level': night_driving,
            'harsh_braking_level': harsh_braking,
            'accidents_last_2yr': accidents,
            'claim_history': claim_val
        }])
        
        try:
            # Preprocess exactly like the pipeline
            input_df = preprocess_data(input_data)
            input_features = create_features(input_df)
            
            # Predict
            exp_freq = predict_frequency(freq_model, input_features)
            exp_sev = predict_severity(sev_model, input_features)
            
            # Predict Premium
            res = calculate_premium(input_features, exp_freq, exp_sev)
            
            final_premium = res['final_premium'].iloc[0]
            risk_category = res['risk_category'].iloc[0]
            risk_index = res['driver_risk_index'].iloc[0]
            
            # Output Metrics
            st.divider()
            st.subheader("Pricing Results")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Calculated Premium", f"₹ {final_premium:,.2f}")
            m2.metric("Risk Category", risk_category)
            m3.metric("Driver Risk Index", f"{risk_index:.2f}")
            
            if risk_category == "High":
                st.error("⚠️ This driver is classified as HIGH RISK due to their behavior or claim history. Premium is scaled heavily.")
            elif risk_category == "Medium":
                st.warning("⚠️ This driver is classified as MEDIUM RISK. Premium is scaled moderately.")
            else:
                st.success("✅ This driver is classified as LOW RISK. Base premium applies.")
                
        except Exception as e:
            st.error(f"Error calculating premium: {e}")

with tab_upload:
    st.header("Batch Risk Analyzer")
    st.markdown("Upload a CSV file containing new driver profiles to instantly calculate their expected premiums and risk categories.")
    
    st.info("The CSV must contain standard columns: `driver_id, age, vehicle_type, vehicle_age, daily_mileage, night_driving_level, harsh_braking_level, accidents_last_2yr, claim_history`")
    
    uploaded_file = st.file_uploader("Upload Drivers CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read CSV
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(batch_df)} drivers.")
            
            with st.spinner("Processing batch data and calculating premiums..."):
                # Run the pipeline steps
                batch_clean = preprocess_data(batch_df)
                batch_features = create_features(batch_clean)
                
                b_exp_freq = predict_frequency(freq_model, batch_features)
                b_exp_sev = predict_severity(sev_model, batch_features)
                
                batch_premium = calculate_premium(batch_features, b_exp_freq, b_exp_sev)
                
            st.subheader("Batch Results")
            
            # Show summary metrics
            high_risk = len(batch_premium[batch_premium['risk_category'] == 'High'])
            med_risk = len(batch_premium[batch_premium['risk_category'] == 'Medium'])
            low_risk = len(batch_premium[batch_premium['risk_category'] == 'Low'])
            
            c1, c2, c3 = st.columns(3)
            c1.metric("🔴 High Risk Drivers", high_risk)
            c2.metric("🟡 Medium Risk Drivers", med_risk)
            c3.metric("🟢 Low Risk Drivers", low_risk)
            
            # Show Dataframe
            display_cols = ['driver_id', 'driver_risk_index', 'risk_category', 'final_premium']
            st.dataframe(batch_premium[display_cols], use_container_width=True)
            
            # Download button
            csv_out = batch_premium.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Calculated Premiums (CSV)",
                data=csv_out,
                file_name="batch_premiums_calculated.csv",
                mime="text/csv",
                type="primary"
            )
            
        except Exception as e:
            st.error(f"Error processing CSV: Ensure it matches the requested schema. Details: {e}")

with tab_portfolio:
    st.header("Portfolio Overview")
    st.markdown("Visualizations generated from the active historical dataset.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists("results/plots/risk_distribution.png"):
            st.image("results/plots/risk_distribution.png", caption="Risk Category Distribution", use_container_width=True)
        if os.path.exists("results/plots/premium_distribution.png"):
            st.image("results/plots/premium_distribution.png", caption="Overall Premium Spread", use_container_width=True)
        
    with col2:
        if os.path.exists("results/plots/scenario_comparison.png"):
            st.image("results/plots/scenario_comparison.png", caption="Simulation Stresses (Jaipur Model)", use_container_width=True)
            
    st.info("💡 To refresh these charts, run `run_pipeline.py` to regenerate the portfolio charts.")
