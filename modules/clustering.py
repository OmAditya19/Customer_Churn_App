# modules/clustering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


class Clustering:
    def __init__(self, df):
        self.df = df.copy()
        self.df_std = None
        self.pca = None
        self.scores_pca = None
        self.df_segmented = None
        self.features_to_scale = [
            'Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay',
            'Subscription Type', 'Contract Length', 'Total Spend', 'Last Interaction'
        ]

    def scale_features(self):
        self.df_std = self.df.copy()
        scaler = RobustScaler()
        self.df_std[self.features_to_scale] = scaler.fit_transform(self.df_std[self.features_to_scale])
        return self.df_std

    def get_scaled_preview(self):
        return self.df_std[self.features_to_scale].head()

    def scree_plot(self):
        self.pca = PCA()
        self.pca.fit(self.df_std.drop('CustomerID', axis=1))
        explained = self.pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)
        n_components_50 = np.argmax(cumulative >= 0.4) + 1

        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
        plt.style.use("seaborn-v0_8-darkgrid")
        ax.bar(range(1, len(explained) + 1), explained * 100, color='skyblue', edgecolor='black', label='Individual')
        ax.plot(range(1, len(cumulative) + 1), cumulative * 100, color='orange', linestyle='--', marker='o', label='Cumulative')
        ax.axhline(y=50, color='red', linestyle='dotted', label='50% Threshold')
        ax.axvline(x=n_components_50, color='green', linestyle='dotted', label=f'{n_components_50} Components')
        ax.set_title('Scree Plot', fontsize=14)
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Explained Variance (%)')
        ax.set_xticks(range(1, len(explained) + 1))
        ax.legend()
        fig.tight_layout()

        return fig, n_components_50

    def component_loadings(self):
        self.pca = PCA(n_components=4)
        self.pca.fit(self.df_std.drop('CustomerID', axis=1))
        df_pca_comp = pd.DataFrame(
            self.pca.components_,
            columns=self.df_std.drop('CustomerID', axis=1).columns,
            index=['Component 1', 'Component 2', 'Component 3', 'Component 4']
        )

        fig, ax = plt.subplots(figsize=(16, 4))
        sns.heatmap(df_pca_comp, vmin=-1, vmax=1, cmap='RdBu', annot=True, ax=ax)
        ax.set_title("PCA Components vs Original Features", fontsize=14)

        return df_pca_comp, fig
     
    def compute_scores_pca(self):
        if self.pca is None:
           self.pca = PCA(n_components=4)
           self.pca.fit(self.df_std.drop('CustomerID', axis=1))
        self.scores_pca = self.pca.transform(self.df_std.drop('CustomerID', axis=1))

    def elbow_plot(self):
        self.compute_scores_pca()
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(self.scores_pca)
            wcss.append(kmeans.inertia_)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, 11), wcss, marker='o', linestyle='-.', color='red')
        ax.set_title('KMeans Elbow Curve')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('WCSS')
        return fig

    def cluster_customers(self):
        self.compute_scores_pca()
        kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
        kmeans.fit(self.scores_pca)

        df_combined = pd.concat([self.df.reset_index(drop=True), pd.DataFrame(self.scores_pca)], axis=1)
        df_combined.columns.values[-4:] = ['Component 1', 'Component 2', 'Component 3', 'Component 4']
        df_combined['Segment K-means PCA'] = kmeans.labels_
        df_combined['Segments'] = df_combined['Segment K-means PCA'].map({
            0: 'The Loyalists',
            1: 'The High-Value Churners',
            2: 'The Low-Maintenance Customers'
          })

        self.df_segmented = df_combined
        return df_combined

    def segment_summary(self):
        return self.df_segmented.drop('CustomerID', axis=1).groupby('Segments').mean(numeric_only=True)

    def scatter_plot(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(
            x=self.df_segmented['Component 1'],
            y=self.df_segmented['Component 2'],
            hue=self.df_segmented['Segments'],
            palette='colorblind',
            s=30,
            edgecolor='black',
            alpha=0.8,
            ax=ax
        )
        ax.set_title("Customer Segments by PCA Components")
        ax.legend(title="Segments", loc='best')
        return fig
