#!/usr/bin/env python3
"""
Enhanced Analysis Demonstration Script

This script demonstrates the improved analysis capabilities with:
1. 1-year historical data training
2. Advanced technical indicators
3. Ensemble ML models
4. Feature selection
5. Better accuracy metrics

Usage: python enhanced_analysis_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from realtime_anomaly_project.config import settings
from realtime_anomaly_project.fusion import recommender
from realtime_anomaly_project.statistical_anomaly.show_results import load_ticker_df

def demonstrate_enhanced_features():
    """Demonstrate the enhanced feature engineering capabilities."""
    print("=== Enhanced Feature Engineering Demo ===")

    # Load a sample ticker
    ticker = "ADANIPORTS.NS"
    df = load_ticker_df(ticker)

    if df is None or len(df) < 100:
        print(f"Insufficient data for {ticker}")
        return

    print(f"Loaded {len(df)} rows for {ticker}")

    # Create enhanced features
    features = recommender._make_features(df)
    print(f"Created {len(features.columns)} features:")
    print(f"Sample features: {list(features.columns[:10])}")

    # Show some statistics
    print("\nFeature Statistics:")
    print(features.describe().loc[['mean', 'std', 'min', 'max']].T.head(10))

    return features

def compare_model_performance():
    """Compare performance of different model types."""
    print("\n=== Model Performance Comparison ===")

    # Load multiple tickers for training
    tickers = getattr(settings, 'TICKERS', [])[:5]  # Use first 5 tickers
    ticker_map = {}

    for t in tickers:
        df = load_ticker_df(t)
        if df is not None and len(df) >= 200:  # Need more data for 1y training
            ticker_map[t] = df
            print(f"Loaded {t}: {len(df)} rows")

    if len(ticker_map) < 2:
        print("Need at least 2 tickers with sufficient data")
        return

    print(f"\nTraining on {len(ticker_map)} tickers with 1-year data...")

    # Test different models
    results = {}

    try:
        print("Training standard model...")
        model_std, report_std, acc_std = recommender.train_model(ticker_map)
        results['standard'] = {'accuracy': acc_std, 'report': report_std}
        print(f"Standard model accuracy: {acc_std:.4f}")
    except Exception as e:
        print(f"Standard model failed: {e}")

    try:
        print("Training ensemble model...")
        model_ens, report_ens, acc_ens = recommender.train_ensemble_model(ticker_map)
        results['ensemble'] = {'accuracy': acc_ens, 'report': report_ens}
        print(f"Ensemble model accuracy: {acc_ens:.4f}")
    except Exception as e:
        print(f"Ensemble model failed: {e}")

    try:
        print("Training feature-selected model...")
        model_fs, report_fs, acc_fs, features = recommender.train_model_with_feature_selection(ticker_map)
        results['feature_selected'] = {'accuracy': acc_fs, 'report': report_fs, 'features': len(features)}
        print(f"Feature-selected model accuracy: {acc_fs:.4f}")
    except Exception as e:
        print(f"Feature-selected model failed: {e}")

    # Compare results
    print("\n=== Performance Comparison ===")
    for model_type, result in results.items():
        acc = result['accuracy']
        print("15")

    return results

def demonstrate_prediction_accuracy():
    """Demonstrate prediction accuracy with real-time data."""
    print("\n=== Real-time Prediction Demo ===")

    ticker = "ADANIPORTS.NS"
    df = load_ticker_df(ticker)

    if df is None or len(df) < 50:
        print(f"Insufficient data for {ticker}")
        return

    # Make prediction on latest data
    prediction = recommender.predict_from_df(df)

    if prediction:
        print(f"Latest prediction for {ticker}: {prediction['label']}")
        if prediction.get('probabilities'):
            probs = prediction['probabilities']
            print(f"Confidence - Buy: {probs[2]:.3f}, Hold: {probs[1]:.3f}, Sell: {probs[0]:.3f}")
    else:
        print("Could not generate prediction")

def show_data_range_info():
    """Show information about the 1-year data range."""
    print("\n=== Data Range Information ===")
    print(f"Training period: 1 year ({settings.LOOKBACK})")
    print(f"Data interval: {settings.INTERVAL}")
    print(f"Number of tickers: {len(getattr(settings, 'TICKERS', []))}")

    # Calculate expected data points
    if settings.INTERVAL == '1h':
        expected_points = 365 * 24  # Rough estimate
    elif settings.INTERVAL == '1d':
        expected_points = 365
    else:
        expected_points = 365 * 6  # 6 4-hour periods per day

    print(f"Expected data points per ticker: ~{expected_points}")

def main():
    """Main demonstration function."""
    print("🚀 Enhanced Analysis Capabilities Demo")
    print("=" * 50)

    show_data_range_info()
    demonstrate_enhanced_features()
    compare_model_performance()
    demonstrate_prediction_accuracy()

    print("\n" + "=" * 50)
    print("✅ Enhanced analysis demo completed!")
    print("\nKey improvements:")
    print("• 1-year historical training data")
    print("• Advanced technical indicators (Bollinger Bands, Stochastic, CCI, etc.)")
    print("• Ensemble ML models for better accuracy")
    print("• Automatic feature selection")
    print("• Improved cross-validation and evaluation")

if __name__ == '__main__':
    main()
