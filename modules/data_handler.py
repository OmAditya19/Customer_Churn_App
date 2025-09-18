# modules/data_handler.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

class DataHandler:
    def __init__(self):
        self.df = None

    def load_data(self, uploaded_file):
        self.df = pd.read_csv(uploaded_file)
        return self.df

    def preview_data(self):
        st.success("File uploaded successfully!")
        st.subheader("🔍 Preview of Uploaded Data")
        st.dataframe(self.df.head())

    def describe_data(self):
        st.subheader("📊 Data Summary")
        st.dataframe(self.df.describe())

    def plot_kde_distributions(self):
        st.subheader("📈 Distribution Plots for Each Feature")
        if st.checkbox("Show KDE Distribution Plots"):
            num_cols = len(self.df.columns)
            cols_per_row = 3
            num_rows = (num_cols + cols_per_row - 1) // cols_per_row

            fig = plt.figure(figsize=(15, 5 * num_rows))
            for i, col in enumerate(self.df.columns):
                plt.subplot(num_rows, cols_per_row, i + 1)
                try:
                    sns.kdeplot(self.df[col], fill=True)
                    plt.title(f'Distribution of {col}')
                except Exception:
                    plt.text(0.5, 0.5, f"Can't plot: {col}", ha='center')
                    plt.title(f"{col} (skipped)")

            plt.tight_layout()
            st.pyplot(fig)

    def correlation_heatmap(self):
        st.subheader("🔗 Correlation Heatmap")
        if st.checkbox("Show Correlation Heatmap"):
            try:
                df_corr = self.df.drop('CustomerID', axis=1) if 'CustomerID' in self.df.columns else self.df
                corr = df_corr.corr(numeric_only=True)

                fig, ax = plt.subplots(figsize=(12, 9))
                sns.heatmap(corr, annot=True, cmap='bwr', ax=ax)
                ax.set_title("Correlation Heatmap", fontsize=14)
                ax.tick_params(axis='y', rotation=0)

                st.pyplot(fig)
                st.markdown(self.describe_correlations(corr))

            except Exception as e:
                st.error(f"Could not generate heatmap: {e}")

    def describe_correlations(self, corr_matrix):
        corr_pairs = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        corr_pairs = corr_pairs.stack().sort_values(key=lambda x: abs(x), ascending=False)

        top_positive = corr_pairs.head(3)
        bottom = corr_pairs.tail(3)

        explanation = "### 🔍 Correlation Insights\n"
        explanation += "**Top Correlated Features:**\n"
        for (f1, f2), val in top_positive.items():
            explanation += f"- `{f1}` and `{f2}` show a strong correlation of **{val:.4f}**.\n"

        explanation += "\n**Weakest Correlations:**\n"
        for (f1, f2), val in bottom.items():
            explanation += f"- `{f1}` and `{f2}` have a minimal correlation of **{val:.4f}**.\n"

        return explanation
