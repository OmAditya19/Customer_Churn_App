# pages/4_Recommendations.py

import streamlit as st
from modules.recommender import RecommendationEngine
import pandas as pd

st.title("🎯 Personalized Coupon Recommendations")

st.markdown("""
This module provides personalized retention strategies for predicted churners
using rule-based recommendations.
""")

# Check dependencies from previous step
if "at_risk_df" not in st.session_state:
    st.warning("⚠️ Please perform churn prediction first.")
    st.stop()

at_risk_df = st.session_state.at_risk_df

# Load and apply rules
try:
    engine = RecommendationEngine("rules.json")
    recommendations = engine.recommend(at_risk_df)
    st.session_state.recommendations = recommendations

    st.subheader("📋 Personalized Recommendations")
    st.dataframe(recommendations[[
        "Tenure", "Usage Frequency", "Support Calls", "Payment Delay",
        "Predicted Churn", "confidence_score", "recommendation"
    ]])

    # CSV Download
    st.subheader("📥 Download Recommendations")
    csv = recommendations.to_csv(index=False)
    st.download_button(
        label="📄 Download as CSV",
        data=csv,
        file_name='recommendations.csv',
        mime='text/csv'
    )

except Exception as e:
    st.error(f"❌ Failed to generate recommendations: {e}")

# Navigation Buttons
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    if st.button("⬅ Back to Churn Prediction"):
        st.switch_page("pages/3_Churn_Prediction.py")
