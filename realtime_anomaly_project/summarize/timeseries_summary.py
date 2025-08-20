import sys
import os
# ensure project root parent is on sys.path so absolute package imports work when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import ruptures as rpt
import pandas as pd
import numpy as np
from datetime import timedelta
from realtime_anomaly_project.config.settings import TICKERS
from realtime_anomaly_project.data_ingestion.yahoo import data_storage  # In-memory storage from yahoo.py

def detect_changepoints(series, n_bkps=3):
    """ Detect changepoints in time-series data using ruptures (expects 1D numeric array). """
    arr = np.asarray(series)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    algo = rpt.KernelCPD(kernel="rbf").fit(arr)
    bkps = algo.predict(n_bkps)
    return bkps

def _parse_lookback_days(lookback: str) -> int:
    """ Parse a lookback like '60d' or an integer/str days -> int days """
    if lookback is None:
        return 30
    if isinstance(lookback, str) and lookback.endswith('d'):
        try:
            return int(lookback[:-1])
        except Exception:
            return 30
    try:
        return int(lookback)
    except Exception:
        return 30

def _ensure_datetime_index(df: pd.DataFrame):
    """ Ensure DataFrame has a datetime index and a 'datetime' column. Return copy. """
    df = df.copy()
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        df = df.set_index('datetime')
    else:
        # try to coerce index to datetime
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            # if that fails, try to infer a datetime column from common names
            for col in ['date', 'timestamp', 'ts']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    df = df.dropna(subset=[col])
                    df = df.set_index(col)
                    break
    return df

def summarize_stock_data(ticker, lookback_period="30d", n_bkps=3):
    """ Summarize stock data with changepoint detection and trend/volatility segmentation """
    # Get stock data from in-memory storage (data_storage from yahoo.py)
    df = data_storage.get(ticker)
    if df is None:
        print(f"No data for {ticker}, skipping summary.")
        return

    # Convert to DataFrame if it's not already
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            print(f"Data for {ticker} is not a DataFrame, skipping.")
            return

    # Ensure datetime index
    df = _ensure_datetime_index(df)
    if df.empty:
        print(f"No valid datetime-indexed data for {ticker}, skipping summary.")
        # debug dump
        try:
            print("DEBUG: raw data preview:", data_storage.get(ticker))
        except Exception:
            pass
        return

    # Filter data for the lookback period
    days = _parse_lookback_days(lookback_period)

    # Align cutoff timezone with the index timezone (handle tz-naive vs tz-aware)
    idx_tz = df.index.tz
    if idx_tz is None:
        now = pd.Timestamp.now()
    else:
        now = pd.Timestamp.now(tz=idx_tz)
    cutoff = now - pd.Timedelta(days=days)

    # Select rows with index >= cutoff
    try:
        df_filtered = df.loc[df.index >= cutoff]
    except Exception:
        df_filtered = df

    if df_filtered.empty:
        print(f"No data for {ticker} in the last {days} days, skipping summary.")
        # debug information
        print("DEBUG: index tz:", idx_tz)
        print("DEBUG: index min/max:", df.index.min(), df.index.max())
        print("DEBUG: head:", df.head(3).to_dict())
        return

    # Ensure we have a 'close' column (case-insensitive)
    if 'close' not in df_filtered.columns:
        # try common variants
        for alt in ['Close', 'close_price', 'closePrice']:
            if alt in df_filtered.columns:
                df_filtered = df_filtered.rename(columns={alt: 'close'})
                break
    if 'close' not in df_filtered.columns:
        print(f"No 'close' column for {ticker}, skipping summary.")
        return

    # Need more samples than breakpoints
    if len(df_filtered) <= max(1, n_bkps):
        print(f"Not enough samples for {ticker} to detect {n_bkps} changepoints, skipping.")
        return

    # Detect changepoints in the closing prices
    try:
        changepoints = detect_changepoints(df_filtered['close'].values, n_bkps=n_bkps)
    except Exception as exc:
        print(f"Changepoint detection failed for {ticker}: {exc}")
        return

    # Segment the data by the detected changepoints (showing trend segments)
    segments = []
    prev_bkp = 0
    # Ensure changepoints is iterable of indices
    for bkp in changepoints:
        # ruptures returns breakpoints as 1-based end indices (may equal len), use slicing carefully
        seg = df_filtered.iloc[prev_bkp:bkp]
        if len(seg) > 0:
            start_date = seg.index[0]
            end_date = seg.index[-1]
            price_change = float(seg['close'].iloc[-1] - seg['close'].iloc[0])
            volatility = float(np.std(seg['close'].values))
            mean_price = float(seg['close'].mean())
            segments.append({
                "start_date": start_date,
                "end_date": end_date,
                "price_change": price_change,
                "volatility": volatility,
                "mean_price": mean_price
            })
        prev_bkp = bkp

    summary = {
        "ticker": ticker,
        "changepoints": changepoints,
        "segments": segments
    }

    return summary

def summarize_all_tickers(lookback_period="60d", n_bkps=4):
    """ Generate summaries for all tickers """
    all_summaries = []
    for ticker in TICKERS:
        summary = summarize_stock_data(ticker, lookback_period=lookback_period, n_bkps=n_bkps)
        if summary:
            all_summaries.append(summary)

    return all_summaries

def print_summaries(summaries):
    """ Print summaries in a human-readable format """
    for summary in summaries:
        print(f"Summary for {summary['ticker']}:")
        for segment in summary['segments']:
            print(f"  - From {segment['start_date']} to {segment['end_date']}:")
            print(f"    - Price Change: {segment['price_change']:.2f}")
            print(f"    - Volatility: {segment['volatility']:.2f}")
            print(f"    - Mean Price: {segment['mean_price']:.2f}")
        print("\n")

if __name__ == "__main__":
    summaries = summarize_all_tickers()
    print_summaries(summaries)

# package marker