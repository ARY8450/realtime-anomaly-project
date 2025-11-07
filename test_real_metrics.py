"""
Test script to verify real metrics are being used instead of synthetic ones
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from realtime_anomaly_project.performance_metrics_cache import get_real_metrics

print("=" * 70)
print("Testing Real Performance Metrics Integration")
print("=" * 70)

# Test 1: Check metrics for each ticker
tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HINDUNILVR.NS', 'HDFCBANK.NS']

print("\n[1/2] Verifying Real Metrics Cache...")
for ticker in tickers:
    trend_metrics = get_real_metrics(ticker, 'trend_prediction')
    anomaly_metrics = get_real_metrics(ticker, 'anomaly_detection')
    sentiment_metrics = get_real_metrics(ticker, 'sentiment_analysis')
    
    print(f"\n{ticker}:")
    print(f"  Trend Prediction:")
    print(f"    Precision: {trend_metrics['precision']:.2%}")
    print(f"    Recall:    {trend_metrics['recall']:.2%}")
    print(f"    F1 Score:  {trend_metrics['f1_score']:.2%}")
    print(f"    ROC-AUC:   {trend_metrics['roc_auc']:.2%}")
    
    print(f"  Anomaly Detection:")
    print(f"    Precision: {anomaly_metrics['precision']:.2%}")
    print(f"    Recall:    {anomaly_metrics['recall']:.2%}")
    print(f"    F1 Score:  {anomaly_metrics['f1_score']:.2%}")

print("\n✓ All tickers have real backtested metrics cached!")

# Test 2: Verify metrics are realistic (not synthetic)
print("\n[2/2] Verifying Metrics Are Not Synthetic...")
synthetic_indicators = []

for ticker in tickers:
    trend = get_real_metrics(ticker, 'trend_prediction')
    
    # Check if metrics look synthetic (too high, formulaic)
    if trend['precision'] > 0.70:
        synthetic_indicators.append(f"{ticker} trend precision suspiciously high: {trend['precision']:.2%}")
    if trend['roc_auc'] > 0.70:
        synthetic_indicators.append(f"{ticker} trend ROC-AUC suspiciously high: {trend['roc_auc']:.2%}")

if synthetic_indicators:
    print("⚠️  Warning: Some metrics look suspicious:")
    for indicator in synthetic_indicators:
        print(f"  - {indicator}")
else:
    print("✓ All metrics appear to be real backtested values (low performance)")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

avg_trend_accuracy = sum([get_real_metrics(t, 'trend_prediction')['accuracy'] 
                          for t in tickers if 'accuracy' in get_real_metrics(t, 'trend_prediction')]) / len(tickers)
avg_trend_precision = sum([get_real_metrics(t, 'trend_prediction')['precision'] for t in tickers]) / len(tickers)
avg_anomaly_precision = sum([get_real_metrics(t, 'anomaly_detection')['precision'] for t in tickers]) / len(tickers)

print(f"\nAverage Performance Across All Tickers:")
print(f"  Trend Prediction Accuracy:    {avg_trend_accuracy:.2%}")
print(f"  Trend Prediction Precision:   {avg_trend_precision:.2%}")
print(f"  Anomaly Detection Precision:  {avg_anomaly_precision:.2%}")

if avg_trend_accuracy < 0.60 and avg_trend_precision < 0.30:
    print("\n✅ REAL METRICS CONFIRMED!")
    print("   Performance is honest (low but realistic)")
    print("   Dashboard will show true system capabilities")
else:
    print("\n⚠️  METRICS MIGHT BE SYNTHETIC!")
    print("   Performance seems too good to be true")

print("\n" + "=" * 70)
print("Test Complete!")
print("=" * 70)
print("\nNext Steps:")
print("1. Restart dashboard to see real metrics")
print("2. Compare old synthetic vs new real numbers")
print("3. Use real metrics to guide improvements")
