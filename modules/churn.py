# modules/churn.py

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)

class ChurnPredictor:
    def __init__(self, df_segmented, cluster_option):
        self.df_segmented = df_segmented
        self.cluster_option = cluster_option
        self.model = None
        self.X = None
        self.y = None
        self.y_pred_prob = None
        self.y_pred = None
        self.at_risk_df = None

    def filter_cluster(self):
        cluster_df = self.df_segmented[self.df_segmented['Segments'] == self.cluster_option]
        if cluster_df.empty:
            raise ValueError("No data found for the selected cluster.")
        return cluster_df

    def load_model(self):
        model_path = f"models/my_model_{self.cluster_option}.h5"
        self.model = load_model(model_path)
        return self.model

    def prepare_data(self, df):
        drop_cols = [
            'CustomerID','Age', 'Gender', 'Subscription Type', 'Total Spend',
            'Contract Length', 'Last Interaction',
            'Component 1', 'Component 2', 'Component 3', 'Component 4',
            'Segment K-means PCA', 'Segments'
        ]
        df = shuffle(df.drop(columns=drop_cols, errors='ignore'), random_state=42)
        if 'Churn' not in df.columns:
            raise ValueError("'Churn' column not found in data.")
        self.X = df.drop(columns=['Churn'])
        self.y = df['Churn']
        return self.X, self.y

    def predict(self):
        if self.model is None:
            raise ValueError("Model not loaded.")
        self.y_pred_prob = self.model.predict(self.X)
        self.y_pred = (self.y_pred_prob > 0.5).astype(int).flatten()
        return self.y_pred_prob, self.y_pred

    def evaluate(self):
        test_loss, test_accuracy = self.model.evaluate(self.X, self.y, verbose=0)
        report = classification_report(self.y, self.y_pred, output_dict=True, zero_division=1)
        conf_matrix = confusion_matrix(self.y, self.y_pred)
        return test_accuracy, report, conf_matrix

    def plot_roc(self):
        fpr, tpr, _ = roc_curve(self.y, self.y_pred_prob)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

    def plot_precision_recall(self):
        precision, recall, _ = precision_recall_curve(self.y, self.y_pred_prob)
        auc_pr = auc(recall, precision)
        return recall, precision, auc_pr

    def get_at_risk(self):
        df = self.X.copy()
        df['churn_probability'] = self.y_pred_prob
        df['Predicted Churn'] = self.y_pred
        self.at_risk_df = df[df['Predicted Churn'] == 1]
        return self.at_risk_df
