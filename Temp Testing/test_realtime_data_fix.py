#!/usr/bin/env python3
"""
Test script to verify comprehensive real-time data management and consistency.
Tests the RealTimeDataManager, data consistency across fetches, and prediction model integration.
"""

import sys
import os
import time
# Add the realtime_anomaly_project directory to the path so we can import from app
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'realtime_anomaly_project'))

from realtime_anomaly_project.app.realtime_data import RealTimeDataManager
import pandas as pd

def test_data_consistency():
    """Test that data fetched multiple times for the same ticker is consistent."""
    print("=== Testing Data Consistency ===")

    manager = RealTimeDataManager()

    # Test the same ticker multiple times
    ticker = "ADANIPORTS.NS"
    results = []

    for i in range(3):
        data = manager.fetch_live_quote(ticker)
        results.append(data)
        print(f"Fetch {i+1}: close={data.get('close')}, marketCap={data.get('marketCap')}")
        time.sleep(0.5)  # Small delay

    # Check consistency
    closes = [r.get('close') for r in results if r.get('close') is not None]
    if len(set(closes)) == 1:
        print("✅ Data is consistent across multiple fetches")
    else:
        print("⚠️  Data varies across fetches (this may be normal for live data)")

    return results[0]

def test_comprehensive_data_fields():
    """Test that all required data fields are fetched."""
    print("\n=== Testing Comprehensive Data Fields ===")

    manager = RealTimeDataManager()
    ticker = "TCS.NS"

    data = manager.fetch_live_quote(ticker)

    required_fields = [
        'open', 'high', 'low', 'close', 'volume',
        'marketCap', 'trailingPE', 'dividendYield',
        'fiftyTwoWeekHigh', 'fiftyTwoWeekLow', 'qtr_div_amt'
    ]

    missing_fields = []
    for field in required_fields:
        if data.get(field) is None:
            missing_fields.append(field)
        else:
            print(f"✅ {field}: {data[field]}")

    if not missing_fields:
        print("✅ All required fields are populated")
    else:
        print(f"⚠️  Missing fields: {missing_fields}")

    return data

def test_bulk_data_fetching():
    """Test fetching data for multiple tickers at once."""
    print("\n=== Testing Bulk Data Fetching ===")

    manager = RealTimeDataManager()
    tickers = ["ADANIPORTS.NS", "TCS.NS", "INFY.NS"]

    start_time = time.time()
    results = manager.fetch_all_tickers_live_data(tickers)
    end_time = time.time()

    print(f"✅ Fetched data for {len(results)} tickers in {end_time - start_time:.2f} seconds")

    for ticker, data in results.items():
        close = data.get('close')
        print(f"  {ticker}: close={close}")

    return results

def test_cache_functionality():
    """Test that caching works correctly."""
    print("\n=== Testing Cache Functionality ===")

    manager = RealTimeDataManager()
    ticker = "ADANIPORTS.NS"

    # First fetch
    start_time = time.time()
    data1 = manager.fetch_live_quote(ticker)
    first_fetch_time = time.time() - start_time

    # Second fetch (should be from cache)
    start_time = time.time()
    data2 = manager.fetch_live_quote(ticker)
    second_fetch_time = time.time() - start_time

    print(".2f")
    print(".2f")

    # Data should be identical
    if data1 == data2:
        print("✅ Cached data matches original")
    else:
        print("⚠️  Cached data differs from original")

    return data1, data2

def test_prediction_model_integration():
    """Test that prediction models can work with the fetched data."""
    print("\n=== Testing Prediction Model Integration ===")

    try:
        from realtime_anomaly_project.fusion import recommender
        print("✅ Recommender module imported successfully")

        # Test feature creation
        manager = RealTimeDataManager()
        ticker = "ADANIPORTS.NS"

        # Get historical data for testing
        hist_data = manager.fetch_ticker_data(ticker, "1y", "1d")
        if hist_data is not None and not hist_data.empty:
            print(f"✅ Historical data fetched: {len(hist_data)} rows")

            # Test feature creation
            features = recommender._make_features(hist_data)
            if not features.empty:
                print(f"✅ Features created: {len(features)} rows, {len(features.columns)} columns")
                print(f"   Feature columns: {list(features.columns[:5])}...")
            else:
                print("⚠️  No features created")
        else:
            print("⚠️  No historical data available for prediction testing")

    except ImportError as e:
        print(f"⚠️  Recommender module not available: {e}")
    except Exception as e:
        print(f"⚠️  Prediction model test failed: {e}")

def main():
    """Run all tests."""
    print("🧪 Starting Comprehensive Real-Time Data Tests\n")

    try:
        # Test data consistency
        test_data_consistency()

        # Test comprehensive fields
        test_comprehensive_data_fields()

        # Test bulk fetching
        test_bulk_data_fetching()

        # Test caching
        test_cache_functionality()

        # Test prediction integration
        test_prediction_model_integration()

        print("\n🎉 All tests completed!")

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
