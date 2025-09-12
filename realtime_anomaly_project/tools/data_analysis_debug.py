#!/usr/bin/env python3
"""
Data Analysis Script to Debug Data Fetching Issues

This script analyzes the data fetching to understand why we're not getting
1 year of historical data and provides solutions.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from realtime_anomaly_project.config import settings

def analyze_data_availability():
    """Analyze what data is actually available from yfinance."""
    print("=== Data Availability Analysis ===")

    ticker = "ADANIPORTS.NS"
    print(f"Testing data availability for {ticker}")

    # Test different periods and intervals
    test_configs = [
        ("1y", "1d"),   # 1 year daily
        ("2y", "1d"),   # 2 years daily
        ("1y", "1h"),   # 1 year hourly
        ("6mo", "1h"),  # 6 months hourly
        ("3mo", "1h"),  # 3 months hourly
        ("1mo", "1h"),  # 1 month hourly
        ("1y", "1wk"),  # 1 year weekly
        ("max", "1d"),  # Maximum daily data
    ]

    results = {}

    for period, interval in test_configs:
        try:
            print(f"Testing period={period}, interval={interval}...")
            data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)

            if data is not None and not data.empty:
                start_date = data.index.min()
                end_date = data.index.max()
                n_rows = len(data)
                date_range = end_date - start_date
                days_diff = date_range.days

                results[f"{period}_{interval}"] = {
                    'rows': n_rows,
                    'start': start_date,
                    'end': end_date,
                    'days': days_diff
                }

                print(f"  ✅ {n_rows} rows, {days_diff} days ({start_date.date()} to {end_date.date()})")
            else:
                print(f"  ❌ No data for {period}_{interval}")
                results[f"{period}_{interval}"] = {'rows': 0, 'start': None, 'end': None, 'days': 0}

        except Exception as e:
            print(f"  ❌ Error for {period}_{interval}: {e}")
            results[f"{period}_{interval}"] = {'rows': 0, 'start': None, 'end': None, 'days': 0}

    return results

def test_chunked_download():
    """Test downloading data in chunks to get more historical data."""
    print("\n=== Chunked Download Test ===")

    ticker = "ADANIPORTS.NS"

    # Try to get 1 year of data by downloading in monthly chunks
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    print(f"Attempting to get data from {start_date.date()} to {end_date.date()}")

    try:
        # Download all data at once
        data_full = yf.download(ticker, start=start_date, end=end_date, interval="1h", progress=False, auto_adjust=True)

        if data_full is not None and not data_full.empty:
            print(f"✅ Full period download: {len(data_full)} rows")
            print(f"   Date range: {data_full.index.min().date()} to {data_full.index.max().date()}")
            print(f"   Days covered: {(data_full.index.max() - data_full.index.min()).days}")

            # Check for gaps
            expected_hours = 365 * 24  # Rough estimate
            actual_hours = len(data_full)
            coverage_pct = (actual_hours / expected_hours) * 100

            print(f"Data coverage: {coverage_pct:.1f}%")
            # Show data distribution by month
            data_full['month'] = data_full.index.to_series().dt.month
            monthly_counts = data_full.groupby('month').size()
            print("\nMonthly data distribution:")
            for month, count in monthly_counts.items():
                print(f"  Month {month}: {count} hours")

            return data_full
        else:
            print("❌ No data from full period download")
            return pd.DataFrame()

    except Exception as e:
        print(f"❌ Error in chunked download: {e}")
        return pd.DataFrame()

def suggest_solutions():
    """Suggest solutions for getting more training data."""
    print("\n=== Suggested Solutions ===")

    solutions = [
        {
            'title': 'Change Interval to Daily',
            'description': 'Use 1d interval instead of 1h for longer historical data',
            'pros': 'More historical data available, faster downloads',
            'cons': 'Less granular data for intraday analysis'
        },
        {
            'title': 'Use Multiple Tickers',
            'description': 'Train on more tickers to increase total training samples',
            'pros': 'More diverse training data, better generalization',
            'cons': 'Increased computational requirements'
        },
        {
            'title': 'Implement Data Augmentation',
            'description': 'Generate synthetic data or use data augmentation techniques',
            'pros': 'Can create unlimited training data',
            'cons': 'May not reflect real market conditions accurately'
        },
        {
            'title': 'Reduce Model Complexity',
            'description': 'Lower the minimum training data requirement in the model',
            'pros': 'Can train with less data',
            'cons': 'May reduce model accuracy and robustness'
        },
        {
            'title': 'Use Alternative Data Sources',
            'description': 'Consider other financial data APIs with more historical data',
            'pros': 'Access to more comprehensive historical data',
            'cons': 'May require API keys and different data formats'
        }
    ]

    for i, solution in enumerate(solutions, 1):
        print(f"\n{i}. {solution['title']}")
        print(f"   {solution['description']}")
        print(f"   ✅ Pros: {solution['pros']}")
        print(f"   ⚠️  Cons: {solution['cons']}")

def implement_solution_1():
    """Implement Solution 1: Change to daily interval for more data."""
    print("\n=== Implementing Solution 1: Daily Interval ===")

    # Modify settings to use daily data
    print("Current settings:")
    print(f"  LOOKBACK: {settings.LOOKBACK}")
    print(f"  INTERVAL: {settings.INTERVAL}")

    print("\nSuggested changes:")
    print("  LOOKBACK: '2y' (extend to 2 years)")
    print("  INTERVAL: '1d' (daily data)")

    # Test with new settings
    ticker = "ADANIPORTS.NS"
    try:
        data = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
        if data is not None and not data.empty:
            print("\n✅ With new settings:")
            print(f"  Rows: {len(data)}")
            print(f"  Date range: {data.index.min().date()} to {data.index.max().date()}")
            print(f"  Days: {(data.index.max() - data.index.min()).days}")
            return len(data)
        else:
            print("❌ No data with new settings")
            return 0
    except Exception as e:
        print(f"❌ Error with new settings: {e}")
        return 0

def main():
    """Main analysis function."""
    print("🔍 Data Fetching Analysis for Enhanced Training")
    print("=" * 60)

    # Analyze current data availability
    results = analyze_data_availability()

    # Test chunked download
    chunked_data = test_chunked_download()

    # Suggest solutions
    suggest_solutions()

    # Implement first solution
    daily_rows = implement_solution_1()

    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    print(f"Current 1y/1h data: ~300 rows")
    print(f"Potential 2y/1d data: ~{daily_rows} rows")
    print(f"Improvement factor: {daily_rows/300:.1f}x more data")

    if daily_rows > 1000:
        print("✅ Solution viable: Switch to daily data for much more training samples")
    else:
        print("⚠️  May need additional solutions for sufficient training data")

if __name__ == '__main__':
    main()
