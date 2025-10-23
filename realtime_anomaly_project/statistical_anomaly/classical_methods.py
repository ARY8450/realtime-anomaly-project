import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import settings, use defaults if not available
try:
    from realtime_anomaly_project.config.settings import TICKERS, Z_ROLL, Z_K, RSI_N
except ImportError:
    TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    Z_ROLL = 20
    Z_K = 2.0
    RSI_N = 14

# Try to import data storage, use mock if not available
try:
    from realtime_anomaly_project.data_ingestion.yahoo import data_storage
except ImportError:
    # Mock data storage
    class MockDataStorage:
        def get(self, ticker):
            return None
    data_storage = MockDataStorage()

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

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Z-score', color='tab:blue')
    ax1.plot(z_score.index, z_score, color='tab:blue', label='Z-score')
    ax1.axhline(y=z_threshold, color='red', linestyle='--', label=f'Z-threshold ({z_threshold})')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('RSI', color='tab:green')
    ax2.plot(rsi.index, rsi, color='tab:green', label='RSI')
    ax2.axhline(y=rsi_threshold, color='orange', linestyle='--', label=f'RSI-threshold ({rsi_threshold})')
    ax2.tick_params(axis='y', labelcolor='tab:green')

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
        df = data_storage.get(ticker)
        if df is None or df.empty:
            print(f"No data for {ticker}, skipping anomaly detection.")
            continue

        close = df["close"]

        z = _rolling_z(close.pct_change(), Z_ROLL).abs()

        rsi = _rsi(close, RSI_N)

        stat_score = (z.clip(0, 4) / 4.0 + (rsi - 50).abs() / 50).clip(0, 1)

        anomalies[ticker] = {
            "z_score": z.iloc[-1],
            "rsi": rsi.iloc[-1],
            "anomaly_flag": int(z.iloc[-1] >= Z_K),
            "stat_score": stat_score.iloc[-1],
        }

        plot_anomalies(ticker, z, rsi, Z_K, 70)

    for ticker, anomaly in anomalies.items():
        print(f"{ticker}: Z-Score={anomaly['z_score']:.2f}, RSI={anomaly['rsi']:.2f}, Anomaly: {'Yes' if anomaly['anomaly_flag'] else 'No'}")

class ClassicalAnomalyDetector:
    """Classical anomaly detection using statistical methods"""
    
    def __init__(self):
        self.z_threshold = 2.0
        self.rsi_threshold = 70
        
    def detect_anomalies(self, data):
        """Detect anomalies in the given data"""
        if data is None or data.empty:
            return {
                'anomaly_scores': pd.Series([0] * len(data)),
                'anomaly_flags': [False] * len(data)
            }
        
        # Calculate returns
        if 'close' in data.columns:
            returns = data['close'].pct_change().fillna(0)
        else:
            returns = data.iloc[:, 0].pct_change().fillna(0)
        
        # Calculate rolling Z-score
        z_scores = self._calculate_z_score(returns)
        
        # Calculate RSI if we have close prices
        if 'close' in data.columns:
            rsi = self._calculate_rsi(data['close'])
        else:
            rsi = pd.Series([50] * len(data), index=data.index)
        
        # Combine scores
        anomaly_scores = self._combine_scores(z_scores, rsi)
        
        # Determine anomaly flags
        anomaly_flags = anomaly_scores > self.z_threshold
        
        return {
            'anomaly_scores': anomaly_scores,
            'anomaly_flags': anomaly_flags.tolist()
        }
    
    def _calculate_z_score(self, returns, window=20):
        """Calculate rolling Z-score"""
        rolling_mean = returns.rolling(window=window, min_periods=window//2).mean()
        rolling_std = returns.rolling(window=window, min_periods=window//2).std()
        rolling_std = rolling_std.replace(0, np.nan)
        z_scores = (returns - rolling_mean) / rolling_std
        return z_scores.fillna(0).abs()
    
    def _calculate_rsi(self, close_prices, period=14):
        """Calculate RSI"""
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=period//2).mean()
        avg_loss = loss.rolling(window=period, min_periods=period//2).mean()
        
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _combine_scores(self, z_scores, rsi):
        """Combine Z-score and RSI into anomaly score"""
        # Normalize RSI to 0-1 scale (50 is neutral)
        rsi_score = (rsi - 50).abs() / 50
        
        # Combine scores
        combined_score = (z_scores.clip(0, 4) / 4.0 + rsi_score.clip(0, 1)) / 2
        
        return combined_score.clip(0, 1)

if __name__ == "__main__":
    compute_anomalies()

# package marker
