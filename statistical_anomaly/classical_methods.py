import sys
import os
# ensure project root parent is on sys.path so absolute imports work when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from realtime_anomaly_project.config.settings import TICKERS, Z_ROLL, Z_K, RSI_N
from realtime_anomaly_project.data_ingestion.yahoo import data_storage  # In-memory storage from data_ingestion/yahoo

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
    rsi = 100 - 100 / (1 + rs)
    return rsi

def plot_anomalies(ticker, z_score, rsi, z_threshold, rsi_threshold):
    """ Plot Z-score and RSI with anomalies marked """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Z-score on the primary y-axis
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Z-score', color='tab:blue')
    ax1.plot(z_score.index, z_score, color='tab:blue', label='Z-score')
    ax1.axhline(y=z_threshold, color='red', linestyle='--', label=f'Z-threshold ({z_threshold})')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Plot RSI on the secondary y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('RSI', color='tab:green')
    ax2.plot(rsi.index, rsi, color='tab:green', label='RSI')
    ax2.axhline(y=rsi_threshold, color='orange', linestyle='--', label=f'RSI-threshold ({rsi_threshold})')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # Highlight anomalies
    anomaly_dates = z_score[z_score >= z_threshold].index
    ax1.scatter(anomaly_dates, z_score.loc[anomaly_dates], color='red', label='Anomalies (Z-score)', zorder=5)

    plt.title(f"Z-score and RSI for {ticker} with Anomalies")
    fig.tight_layout()
    plt.legend()
    plt.show()

def compute_anomalies():
    """ Compute anomalies based on z-score and RSI and plot the results """
    anomalies = {}

    for ticker in TICKERS:
        # Get stock data from in-memory dictionary (data_storage from yahoo.py)
        df = data_storage.get(ticker)
        if df is None or df.empty:
            print(f"No data for {ticker}, skipping anomaly detection.")
            continue

        # Get closing prices
        close = df["close"]

        # Calculate returns and compute rolling Z-score
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

        # Plot anomalies for each ticker
        plot_anomalies(ticker, z, rsi, Z_K, 70)  # 70 as RSI threshold

    # Print anomalies for each ticker
    for ticker, anomaly in anomalies.items():
        print(f"{ticker}: Z-Score={anomaly['z_score']:.2f}, RSI={anomaly['rsi']:.2f}, Anomaly: {'Yes' if anomaly['anomaly_flag'] else 'No'}")

if __name__ == "__main__":
    compute_anomalies()

# package marker
