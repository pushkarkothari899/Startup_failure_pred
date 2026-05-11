import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Startup Failure Risk Predictor",
    page_icon="🚀",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("startup_failure_detection.pkl")

@st.cache_resource
def load_feature_columns():
    return joblib.load("feature_columns.pkl")
model = load_model()
feature_columns = load_feature_columns()

# Title
st.title("🚀 Startup Failure Risk Predictor")
st.markdown("*Predict the probability of a startup failing using ML trained on Crunchbase data*")
st.divider()

# Sidebar
st.sidebar.header("Enter Startup Details")
founded_year = st.sidebar.slider("Founded Year", 1990, 2015, 2010)
funding_total = st.sidebar.number_input("Total Funding (USD)", min_value=0, value=500000, step=10000)
funding_rounds = st.sidebar.slider("Number of Funding Rounds", 1, 10, 2)
seed = st.sidebar.number_input("Seed Funding (USD)", min_value=0, value=0, step=10000)
venture = st.sidebar.number_input("Venture Funding (USD)", min_value=0, value=0, step=10000)
angel = st.sidebar.number_input("Angel Funding (USD)", min_value=0, value=0, step=10000)

market = st.sidebar.selectbox("Market", [
    "Software", "Mobile", "E-Commerce", "Biotechnology",
    "Health Care", "Education", "Finance", "Gaming",
    "Social Media", "Security", "unknown"
])

country = st.sidebar.selectbox("Country", [
    "USA", "GBR", "CAN", "IND", "DEU", "FRA",
    "AUS", "ISR", "CHN", "SGP"
])

predict_btn = st.sidebar.button("Predict Risk", type="primary")

if predict_btn:
    # Build input row with all zeros
    input_data = pd.DataFrame(0, index=[0], columns=feature_columns)

    # Fill known values
    if 'founded_year' in input_data.columns:
        input_data['founded_year'] = founded_year
    if 'funding_total_usd' in input_data.columns:
        input_data['funding_total_usd'] = funding_total
    if 'funding_rounds' in input_data.columns:
        input_data['funding_rounds'] = funding_rounds
    if 'seed' in input_data.columns:
        input_data['seed'] = seed
    if 'venture' in input_data.columns:
        input_data['venture'] = venture
    if 'angel' in input_data.columns:
        input_data['angel'] = angel
    if 'funding_duration_days' in input_data.columns:
        input_data['funding_duration_days'] = 0  
    if 'funding_per_round' in input_data.columns:
        input_data['funding_per_round'] = funding_total / (funding_rounds + 1)
    if 'startup_age_years' in input_data.columns:
        input_data['startup_age_years'] = 2024 - founded_year    

    # Market encoding
    market_col = f'market_{market}'
    if market_col in input_data.columns:
        input_data[market_col] = 1

    # Country encoding
    country_col = f'country_code_{country}'
    if country_col in input_data.columns:
        input_data[country_col] = 1

    # Predict
    prob = model.predict_proba(input_data)[0][1]
    risk_pct = round(prob * 100, 1)

    # Display result
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Assessment")
        if risk_pct < 30:
            st.success(f"### ✅ Low Risk: {risk_pct}%")
            st.markdown("This startup shows **low failure risk** based on the provided data.")
        elif risk_pct < 60:
            st.warning(f"### ⚠️ Medium Risk: {risk_pct}%")
            st.markdown("This startup shows **moderate failure risk**. Proceed with caution.")
        else:
            st.error(f"### 🔴 High Risk: {risk_pct}%")
            st.markdown("This startup shows **high failure risk** based on historical patterns.")

        # Risk meter
        fig, ax = plt.subplots(figsize=(6, 1))
        ax.barh(['Risk'], [risk_pct], color='red' if risk_pct > 60 else 'orange' if risk_pct > 30 else 'green')
        ax.barh(['Risk'], [100 - risk_pct], left=[risk_pct], color='lightgrey')
        ax.set_xlim(0, 100)
        ax.set_xlabel("Failure Probability (%)")
        ax.set_title(f"Risk Score: {risk_pct}%")
        st.pyplot(fig)

    with col2:
        st.subheader("Feature Importance")
        img = plt.imread("feature_importance.png")
        st.image(img, caption="Top factors predicting startup failure", use_container_width=True)

    # SHAP explanation
    st.divider()
    st.subheader("Why this prediction?")
    with st.spinner("Generating explanation..."):
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(input_data)
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        shap.summary_plot(shap_vals, input_data, plot_type="bar", max_display=10, show=False)
        st.pyplot(fig2)

else:
    # Default view
    col1, col2 = st.columns(2)
    with col1:
        st.info("👈 Fill in the startup details on the left and click **Predict Risk**")
        st.markdown("""
        ### How it works
        - Enter your startup's details in the sidebar
        - Our XGBoost model analyzes 870+ features
        - Get a failure probability score instantly
        - See which factors drive the prediction via SHAP
        """)
    with col2:
        st.subheader("Top Failure Predictors")
        st.image("feature_importance_updated.png", use_container_width=True)

st.markdown("---")
with st.expander("📊 Model Insights & Data Limitations (Read Before Use)"):
    st.markdown("""
    ### The "Unknown Market" Survivorship Bias
    During the evaluation of this model, a significant data bias was discovered within the original Crunchbase dataset regarding the `market` feature.
    
    * **The Observation:** Startups with an "unknown" market category receive an artificially lower failure risk score compared to those with defined markets (e.g., "Software").
    * **The Cause:** This indicates a survivorship or reporting bias in the source data. Successful or rapidly acquired companies often had incomplete market tags left by data entry teams, whereas companies that failed were more meticulously categorized during post-mortems.
    * **The Takeaway:** While the XGBoost model correctly learned the mathematical patterns of the dataset, the dataset itself contains structural leakage. In a production environment, the `market` feature would be dropped, and the model retrained strictly on objective financial velocity metrics.
    """)