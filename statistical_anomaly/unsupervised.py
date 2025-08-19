import sys
import os
# ensure project root parent is on sys.path so absolute package imports work when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from realtime_anomaly_project.config.settings import TICKERS
from realtime_anomaly_project.data_ingestion.yahoo import data_storage

def run_isolation_forest(ticker, data):
    """ Use Isolation Forest for anomaly detection """
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(data)
    preds = model.predict(data)
    return preds

def run_one_class_svm(ticker, data):
    """ Use One-Class SVM for anomaly detection """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    model = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
    model.fit(data_scaled)
    preds = model.predict(data_scaled)
    return preds

def compute_unsupervised_anomalies():
    """ Compute unsupervised anomalies using Isolation Forest and One-Class SVM """
    unsupervised_anomalies = {}

    for ticker in TICKERS:
        df = data_storage.get(ticker)
        if df is None or df.empty:
            print(f"No data for {ticker}, skipping unsupervised anomaly detection.")
            continue

        returns = df["close"].pct_change().dropna().values.reshape(-1, 1)

        isolation_forest_preds = run_isolation_forest(ticker, returns)
        one_class_svm_preds = run_one_class_svm(ticker, returns)

        unsupervised_anomalies[ticker] = {
            "isolation_forest_anomalies": isolation_forest_preds,
            "one_class_svm_anomalies": one_class_svm_preds,
        }

        print(f"{ticker}: Anomalies detected using Isolation Forest and One-Class SVM.")

    for ticker, anomaly_data in unsupervised_anomalies.items():
        print(f"\n{ticker}:")
        print(f"  Isolation Forest Anomalies: {list(anomaly_data['isolation_forest_anomalies'])}")
        print(f"  One-Class SVM Anomalies: {list(anomaly_data['one_class_svm_anomalies'])}")

if __name__ == "__main__":
    compute_unsupervised_anomalies()
