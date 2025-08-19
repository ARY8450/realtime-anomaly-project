import sys
import os
# ensure project root parent is on sys.path so absolute package imports work when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import ruptures as rpt
import pandas as pd
import numpy as np
from realtime_anomaly_project.config.settings import TICKERS
from realtime_anomaly_project.data_ingestion.yahoo import data_storage  # In-memory storage from yahoo.py

def detect_changepoints(series, n_bkps=3):
    """ Detect changepoints in time-series data using ruptures """
    algo = rpt.KernelCPD(kernel="rbf").fit(series)
    bkps = algo.predict(n_bkps)
    return bkps

def summarize_stock_data(ticker, lookback_period="30d"):
    """ Summarize stock data with changepoint detection and trend/volatility segmentation """
    # Get stock data from in-memory storage (data_storage from yahoo.py)
    df = data_storage.get(ticker)
    if df is None or df.empty:
        print(f"No data for {ticker}, skipping summary.")
        return
    
    # Filter data for the lookback period (if needed, adjust this logic)
    df = df.tail(pd.to_datetime(lookback_period).days)

    # Detect changepoints in the closing prices (using kernel-based changepoint detection)
    changepoints = detect_changepoints(df["close"].values)

    # Segment the data by the detected changepoints (showing trend segments)
    segments = []
    prev_bkp = 0
    for bkp in changepoints:
        segment = df.iloc[prev_bkp:bkp]
        if len(segment) > 0:
            segments.append({
                "start_date": segment.index[0],
                "end_date": segment.index[-1],
                "price_change": (segment["close"].iloc[-1] - segment["close"].iloc[0]),
                "volatility": np.std(segment["close"]),
                "mean_price": segment["close"].mean()
            })
        prev_bkp = bkp

    # Combine trend/volatility segments into a summary
    summary = {
        "ticker": ticker,
        "changepoints": changepoints,
        "segments": segments
    }

    return summary

def summarize_all_tickers():
    """ Generate summaries for all tickers """
    all_summaries = []
    for ticker in TICKERS:
        summary = summarize_stock_data(ticker)
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