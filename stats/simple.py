import sys
import os

# Ensure project root is on sys.path so "config" package can be imported when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from config.settings import TICKERS, Z_ROLL, Z_K, RSI_N
from data_ingestion.yahoo import data_storage  # In-memory storage from data_ingestion/yahoo

def _rolling_z(x: pd.Series, win: int) -> pd.Series:
    """ Compute rolling Z-score for returns """
    mu = x.rolling(win, min_periods=win // 2).mean()
    sd = x.rolling(win, min_periods=win // 2).std().replace(0, np.nan)
    return (x - mu) / sd

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """ Calculate RSI (Relative Strength Index) """
    diff = close.diff().fillna(0)
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    rs = up.rolling(n, min_periods=n // 2).mean() / (down.rolling(n, min_periods=n // 2).mean() + 1e-9)
    return 100 - 100 / (1 + rs)

def compute_anomalies():
    """ Compute anomalies based on z-score and RSI """
    anomalies = {}
    
    for ticker in TICKERS:
        # Get stock data from in-memory dictionary (data_storage from yahoo.py)
        df = data_storage.get(ticker)
        if df is None or df.empty:
            print(f"No data for {ticker}, skipping anomaly detection.")
            continue

        # Get closing prices
        close = df["close"]

        # Calculate daily returns and compute rolling Z-score
        z = _rolling_z(close.pct_change(), Z_ROLL).abs()  # Absolute Z-score for simplicity

        # Calculate RSI (Relative Strength Index)
        rsi = _rsi(close, RSI_N)

        # Simple anomaly scoring: combine z-score and RSI
        stat_score = (z.clip(0, 4) / 4.0 + (rsi - 50).abs() / 50).clip(0, 1)

        # Detect anomalies based on Z-score threshold (Z_K)
        anomalies[ticker] = {
            "z_score": z.iloc[-1],  # Latest z-score
            "rsi": rsi.iloc[-1],    # Latest RSI
            "anomaly_flag": int(z.iloc[-1] >= Z_K),  # Flag if z-score exceeds threshold
            "stat_score": stat_score.iloc[-1],  # Combine score for anomaly
        }

    # Print anomalies for each ticker
    for ticker, anomaly in anomalies.items():
        print(f"{ticker}: Z-Score={anomaly['z_score']:.2f}, RSI={anomaly['rsi']:.2f}, Anomaly: {'Yes' if anomaly['anomaly_flag'] else 'No'}")

if __name__ == "__main__":
    compute_anomalies()
