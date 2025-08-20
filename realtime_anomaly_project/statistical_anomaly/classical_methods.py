import sys
import os
# ensure project root parent is on sys.path so absolute imports work when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from realtime_anomaly_project.config.settings import TICKERS, Z_ROLL, Z_K, RSI_N
from realtime_anomaly_project.data_ingestion.yahoo import data_storage  # In-memory storage from data_ingestion/yahoo
from realtime_anomaly_project.database.db_setup import setup_database, StockData
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

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

def _load_ticker_df(ticker: str) -> pd.DataFrame | None:
    """
    Return a pandas DataFrame for ticker using in-memory data_storage first,
    then falling back to the DB. DataFrame will have a tz-aware DatetimeIndex
    and a numeric 'close' column when returned, otherwise None.
    """
    # try in-memory first
    df = data_storage.get(ticker)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df_local = df.copy()
    else:
        # fallback to DB
        Session = setup_database()
        with Session() as s:
            rows = s.query(StockData).filter(StockData.ticker == ticker).order_by(StockData.timestamp).all()
            if not rows:
                return None
            recs = []
            for r in rows:
                recs.append({
                    "datetime": r.timestamp,
                    "open": r.open_price,
                    "high": r.high_price,
                    "low": r.low_price,
                    "close": r.close_price,
                    "volume": r.volume
                })
            df_local = pd.DataFrame(recs)

    # normalize index/close
    try:
        if 'datetime' in df_local.columns:
            df_local['datetime'] = pd.to_datetime(df_local['datetime'], errors='coerce', utc=True)
            df_local = df_local.dropna(subset=['datetime']).set_index('datetime').sort_index()
        else:
            df_local.index = pd.to_datetime(df_local.index, errors='coerce', utc=True)
            df_local = df_local[~df_local.index.isna()].sort_index()

        if df_local.index.tz is None:
            df_local.index = df_local.index.tz_localize('UTC')
    except Exception:
        return None

    # normalize close
    if 'close' not in df_local.columns and 'Close' in df_local.columns:
        df_local = df_local.rename(columns={'Close': 'close'})
    if 'close' not in df_local.columns:
        # try to find numeric column
        numeric_cols = [c for c in df_local.columns if pd.api.types.is_numeric_dtype(df_local[c])]
        if numeric_cols:
            df_local = df_local.rename(columns={numeric_cols[0]: 'close'})
        else:
            return None

    df_local['close'] = pd.to_numeric(df_local['close'], errors='coerce')
    df_local = df_local.dropna(subset=['close'])
    if df_local.empty:
        return None

    return df_local

def compute_anomalies(plot_each=False):
    """ Compute anomalies based on z-score and RSI and (optionally) plot the results """
    anomalies = {}

    for ticker in TICKERS:
        df = _load_ticker_df(ticker)
        if df is None or df.empty:
            print(f"No data for {ticker}, skipping anomaly detection.")
            anomalies[ticker] = None
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
            "z_score": float(z.iloc[-1]) if not z.empty else None,
            "rsi": float(rsi.iloc[-1]) if not rsi.empty else None,
            "anomaly_flag": int(bool(z.iloc[-1] >= Z_K)) if not z.empty else 0,
            "stat_score": float(stat_score.iloc[-1]) if not stat_score.empty else None,
        }

        # Isolation Forest
        try:
            iso_forest = IsolationForest(contamination=0.01, random_state=0)
            df['anomaly_iso'] = iso_forest.fit_predict(df[['close']])
            df['anomaly_iso'] = df['anomaly_iso'].map({1: 0, -1: 1})  # Convert to binary anomaly flag
        except Exception:
            df['anomaly_iso'] = 0

        # One-Class SVM
        try:
            oc_svm = OneClassSVM(gamma='auto', nu=0.01)
            df['anomaly_svm'] = oc_svm.fit_predict(df[['close']])
            df['anomaly_svm'] = df['anomaly_svm'].map({1: 0, -1: 1})
        except Exception:
            df['anomaly_svm'] = 0

        # PCA
        try:
            pca = PCA(n_components=min(2, max(1, df[['close']].shape[1] if df[['close']].ndim>1 else 1)))
            pca_result = pca.fit_transform(df[['close']])
            df['pca_one'] = pca_result[:, 0] if pca_result.shape[1] >= 1 else 0
            if pca_result.shape[1] > 1:
                df['pca_two'] = pca_result[:, 1]
            mse = mean_squared_error(df[['close']], pca.inverse_transform(pca_result))
            print(f"{ticker} PCA Reconstruction MSE: {mse:.2f}")
        except Exception:
            pass

        if plot_each:
            plot_anomalies(ticker, z, rsi, Z_K, 70)  # 70 as RSI threshold

    # Print anomalies for each ticker
    for ticker, anomaly in anomalies.items():
        if anomaly is None:
            print(f"{ticker}: No data")
            continue
        print(f"{ticker}: Z-Score={anomaly['z_score']:.2f}, RSI={anomaly['rsi']:.2f}, Anomaly: {'Yes' if anomaly['anomaly_flag'] else 'No'}")

    return anomalies

if __name__ == "__main__":
    compute_anomalies()

# package marker
