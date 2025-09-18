import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from modules.churn import ChurnPredictor

st.title("📉 Churn Prediction")

if 'df_segmented' not in st.session_state:
    st.warning("⚠️ Please perform segmentation first.")
    st.stop()

df_segmented = st.session_state.df_segmented

st.subheader("🧾 Segmented Data Preview")
st.dataframe(df_segmented.head())

st.subheader("🔍 Choose a Cluster for Churn Prediction")
cluster_option = st.selectbox("Select Customer Segment:", [
    "The Loyalists", "The High-Value Churners", "The Low-Maintenance Customers"
])

predictor = ChurnPredictor(df_segmented, cluster_option)

# Step 1: Filter Cluster
try:
    cluster_df = predictor.filter_cluster()
    st.success(f"✅ Showing customers from Cluster {cluster_option}")
    st.dataframe(cluster_df.drop(["Component 1","Component 2","Component 3","Component 4"],axis =1).head())
except Exception as e:
    st.error(f"❌ {e}")
    st.stop()

# Step 2: Load model
try:
    model = predictor.load_model()
    st.success(f"✅ Loaded Model: my_model_{cluster_option}.h5")
except Exception as e:
    st.error(f"❌ {e}")
    st.stop()

# Step 3: Prepare data
try:
    X_new, y_new = predictor.prepare_data(cluster_df)
    st.write("📌 Features used for prediction:")
    st.dataframe(X_new.head())
except Exception as e:
    st.error(f"❌ {e}")
    st.stop()

st.subheader("📊 Churn Prediction Evaluation")

if st.button("Run Churn Prediction"):
    try:
        y_pred_prob, y_pred = predictor.predict()
        acc, report_dict, conf_matrix = predictor.evaluate()

        st.success(f"✅ Test Accuracy: **{acc * 100:.2f}%**")

        # Report
        st.markdown("### 📄 Classification Report")
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))

        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        # Save for next
        st.session_state.y_pred_prob = y_pred_prob
        st.session_state.y_pred = y_pred

        # Advanced metrics
        st.subheader("📌 Advanced Evaluation")

        at_risk_df = predictor.get_at_risk()
        st.markdown("### 🚨 At-Risk Customers")
        st.dataframe(at_risk_df.head())
        st.session_state.at_risk_df = at_risk_df
 
        # ✅ CSV download option
        csv = at_risk_df.to_csv(index=False).encode('utf-8')
        st.download_button(
        label="📥 Download At-Risk Customers (CSV)",
        data=csv,
        file_name='at_risk_customers.csv',
        mime='text/csv'
        )

        # ROC
        st.markdown("### ROC Curve")
        fpr, tpr, roc_auc = predictor.plot_roc()
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='blue')
        ax1.plot([0, 1], [0, 1], color='red', linestyle='--')
        ax1.set_title("ROC Curve")
        ax1.legend()
        st.pyplot(fig1)

        # Precision-Recall
        st.markdown("### Precision Recall Curve")
        recall, precision, auc_pr = predictor.plot_precision_recall()
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.plot(recall, precision, marker='.', label=f'PR AUC = {auc_pr:.2f}')
        ax2.set_title("Precision-Recall Curve")
        ax2.legend()
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

# Navigation
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    if st.button("⬅ Back to Segmentation"):
        st.switch_page("pages/2_Segmentation.py")
with col3:
    if st.button("Next: Recommendations ➡"):
        st.switch_page("pages/4_Recommendations.py")
