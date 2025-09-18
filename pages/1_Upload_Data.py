#pages/1_Upload_Data.py

import streamlit as st
from modules.data_handler import DataHandler

st.title("📤 Upload Your Customer Data")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

data_handler = DataHandler()

if uploaded_file is not None:
    df = data_handler.load_data(uploaded_file)
    st.session_state.df = df 
    data_handler.preview_data()
    data_handler.describe_data()
    data_handler.plot_kde_distributions()
    data_handler.correlation_heatmap()
else:
    st.info("Please upload a CSV file to continue.")

# Navigation Buttons
col1, col2, col3 = st.columns([1, 5, 1])

with col1:
    if st.button("⬅ Back to Home"):
        st.switch_page("app.py")

with col3:
    if st.button("Next: Segmentation ➡"):
        st.switch_page("pages/2_Segmentation.py")  # Update as needed
