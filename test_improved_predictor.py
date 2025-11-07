"""
Backtest Improved Trend Predictor
Tests the ML+Technical ensemble system
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from real_backtesting_system import RealBacktester
import yfinance as yf
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import improved predictor
try:
    from realtime_anomaly_project.improved_trend_predictor import ImprovedTrendPredictor
    IMPROVED_AVAILABLE = True
except ImportError:
    logger.error("Improved predictor not available!")
    IMPROVED_AVAILABLE = False
    sys.exit(1)

print("=" * 70)
print("Backtesting Improved Trend Predictor (Target: 75% Accuracy)")
print("=" * 70)

if not IMPROVED_AVAILABLE:
    print("✗ Improved predictor not available")
    sys.exit(1)

# Initialize
backtester = RealBacktester(lookback_days=730)
improved_predictor = ImprovedTrendPredictor()

# Test tickers
tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HINDUNILVR.NS', 'HDFCBANK.NS']

results = {}

for ticker in tickers:
    print(f"\n{'='*70}")
    print(f"Testing {ticker}")
    print(f"{'='*70}")
    
    try:
        # Fetch data
        df = yf.download(ticker, period='2y', progress=False)
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [col.lower() for col in df.columns]
        
        if df.empty or len(df) < 100:
            print(f"✗ Insufficient data for {ticker}")
            continue
        
        print(f"Data points: {len(df)}")
        print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
        
        # Test improved predictor
        print("\nTesting Improved Predictor (ML + Technical)...")
        
        predictions = []
        actuals = []
        confidences = []
        ml_enabled_count = 0
        
        for i in range(50, len(df) - 5):  # Need history, check 5 days ahead
            hist_data = df.iloc[:i+1].copy()
            
            # Make prediction using improved predictor
            result = improved_predictor.predict(hist_data, ticker)
            prediction = result['prediction']
            confidence = result['confidence']
            
            if result.get('ml_enabled'):
                ml_enabled_count += 1
            
            # Get actual outcome
            current_price = df.iloc[i]['close']
            future_price = df.iloc[i+5]['close']
            actual_return = (future_price - current_price) / current_price
            
            # Convert to classes
            pred_class = 1 if prediction == 'BUY' else -1 if prediction == 'SELL' else 0
            actual_class = 1 if actual_return > 0.02 else -1 if actual_return < -0.02 else 0
            
            predictions.append(pred_class)
            actuals.append(actual_class)
            confidences.append(confidence)
        
        # Calculate metrics
        import numpy as np
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        confidences = np.array(confidences)
        
        # Overall accuracy
        accuracy = accuracy_score(actuals, predictions)
        
        # Binary metrics (UP vs NOT-UP)
        binary_preds = (predictions == 1).astype(int)
        binary_actuals = (actuals == 1).astype(int)
        
        precision = precision_score(binary_actuals, binary_preds, zero_division=0)
        recall = recall_score(binary_actuals, binary_preds, zero_division=0)
        f1 = f1_score(binary_actuals, binary_preds, zero_division=0)
        
        # High confidence predictions only
        high_conf_mask = confidences > 0.7
        if high_conf_mask.sum() > 10:
            high_conf_accuracy = accuracy_score(actuals[high_conf_mask], predictions[high_conf_mask])
        else:
            high_conf_accuracy = 0
        
        results[ticker] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'high_conf_accuracy': high_conf_accuracy,
            'high_conf_count': high_conf_mask.sum(),
            'total_predictions': len(predictions),
            'ml_enabled_pct': (ml_enabled_count / len(predictions)) * 100 if predictions.size > 0 else 0
        }
        
        print(f"\n✓ Results for {ticker}:")
        print(f"  Overall Accuracy:       {accuracy:.2%}")
        print(f"  Precision (BUY):        {precision:.2%}")
        print(f"  Recall (BUY):           {recall:.2%}")
        print(f"  F1 Score:               {f1:.2%}")
        print(f"  High Confidence Acc:    {high_conf_accuracy:.2%} ({high_conf_mask.sum()} predictions)")
        print(f"  ML Model Usage:         {(ml_enabled_count / len(predictions)) * 100:.1f}% of predictions")
        
    except Exception as e:
        print(f"✗ Error testing {ticker}: {e}")
        import traceback
        traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("SUMMARY - Improved Predictor Performance")
print("=" * 70)

if results:
    avg_accuracy = sum(r['accuracy'] for r in results.values()) / len(results)
    avg_precision = sum(r['precision'] for r in results.values()) / len(results)
    avg_f1 = sum(r['f1_score'] for r in results.values()) / len(results)
    avg_high_conf = sum(r['high_conf_accuracy'] for r in results.values()) / len(results)
    avg_ml_usage = sum(r['ml_enabled_pct'] for r in results.values()) / len(results)
    
    print(f"\nAverage Performance:")
    print(f"  Overall Accuracy:       {avg_accuracy:.2%}")
    print(f"  Precision:              {avg_precision:.2%}")
    print(f"  F1 Score:               {avg_f1:.2%}")
    print(f"  High Confidence Acc:    {avg_high_conf:.2%}")
    print(f"  ML Model Usage:         {avg_ml_usage:.1f}%")
    
    print(f"\nComparison to Baseline:")
    baseline_accuracy = 0.4770  # From previous backtesting
    improvement = (avg_accuracy - baseline_accuracy) * 100
    print(f"  Baseline (Technical):   {baseline_accuracy:.2%}")
    print(f"  Improved (ML+Tech):     {avg_accuracy:.2%}")
    print(f"  Improvement:            {improvement:+.1f} percentage points")
    
    if avg_accuracy >= 0.75:
        print(f"\n✅ TARGET ACHIEVED! {avg_accuracy:.2%} >= 75%")
    elif avg_accuracy >= 0.70:
        print(f"\n🎯 CLOSE TO TARGET! {avg_accuracy:.2%} (need {(0.75 - avg_accuracy) * 100:.1f}% more)")
    else:
        print(f"\n⚠️  BELOW TARGET: {avg_accuracy:.2%} (need {(0.75 - avg_accuracy) * 100:.1f}% more)")
        print(f"\nSuggestions to improve:")
        print(f"  1. Increase ML model training data (currently 2 years)")
        print(f"  2. Add more features (fundamental data, sentiment)")
        print(f"  3. Tune ensemble weights (currently 70% ML, 30% Technical)")
        print(f"  4. Focus on high-confidence predictions only")

else:
    print("\n✗ No results to display")

print("\n" + "=" * 70)
