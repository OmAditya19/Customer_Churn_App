# pages/2_Segmentation.py

import streamlit as st
from modules.clustering import Clustering

st.title("🧩 Customer Segmentation")

if 'df' not in st.session_state:
    st.warning("⚠️ Please upload your data in the Upload Data page first.")
    st.stop()

df = st.session_state.df
clusterer = Clustering(df)

df_std = clusterer.scale_features()
st.success("✅ Selected features have been scaled using RobustScaler.")

if st.checkbox("Show Scaled Data Preview"):
    st.dataframe(clusterer.get_scaled_preview())

st.subheader("🧮 Principal Component Analysis (PCA)")

if st.checkbox("Show Scree Plot (Explained Variance by Components)"):
    fig, n_comp = clusterer.scree_plot()
    st.pyplot(fig)
    st.markdown(f"✅ **{n_comp} principal components** explain at least **50%** of the variance.")

st.subheader("📌 Principal Component Loadings vs Original Features")
if st.checkbox("Show Component Loadings Heatmap"):
    df_pca_comp, fig = clusterer.component_loadings()
    st.dataframe(df_pca_comp.style.format("{:.2f}"))
    st.pyplot(fig)

st.subheader("🔀 KMeans Clustering")
if st.checkbox("Show Elbow Curve (WCSS)"):
    fig = clusterer.elbow_plot()
    st.pyplot(fig)
    st.markdown("✅ Elbow appears at **n=3**, suggesting 3 clusters is optimal.")

df_segmented = clusterer.cluster_customers()
st.session_state.df_segmented = df_segmented
st.success("✅ KMeans clustering applied and customer segments created.")

if st.checkbox("Show Segment-wise Mean Statistics"):
    summary = clusterer.segment_summary()
    st.dataframe(summary.style.format("{:.2f}"))

if st.checkbox("Show PCA Scatter Plot of Clusters"):
    fig = clusterer.scatter_plot()
    st.pyplot(fig)

# Navigation buttons
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    if st.button("⬅ Back to Upload Page"):
        st.switch_page("pages/1_Upload_Data.py")
with col3:
    if st.button("Next: Churn Prediction ➡"):
        st.switch_page("pages/3_Churn_Prediction.py")
