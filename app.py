import streamlit as st

# Set up the app configuration
st.set_page_config(
    page_title="Customer Churn App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main content
st.title("📊 Customer Churn Prediction & Recommendation System")

st.markdown("""
Welcome to the **Customer Churn Prediction App**.  
This application helps you:
- Upload and process your customer data
- Segment customers into meaningful groups
- Predict churn for each group using deep learning
- Recommend retention strategies or coupons
- Visualize insights from your data

👉 Use the sidebar to navigate through the app.
""")

# Navigation Button
if st.button("➡ Next: Upload Data"):
    st.switch_page("pages/1_Upload_Data.py")

# Sidebar content
st.sidebar.title("Navigation")
st.sidebar.markdown("""
1. **Home**  
2. **Upload Data**  
3. **Customer Segmentation**  
4. **Churn Prediction**  
5. **Recommendations**  
6. **Visualizations**
""")
