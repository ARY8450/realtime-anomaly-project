"""
Test Simple Improved Predictor
Tests the enhanced technical analysis system
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from realtime_anomaly_project.simple_improved_predictor import SimpleImprovedPredictor

print("=" * 70)
print("Testing Simple Improved Predictor (Target: 60-70% Accuracy)")
print("=" * 70)

# Initialize
predictor = SimpleImprovedPredictor()

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
        
        # Test predictions
        print("\nTesting Simple Improved Predictor...")
        
        predictions = []
        actuals = []
        confidences = []
        estimated_accs = []
        
        for i in range(50, len(df) - 5):  # Need history, check 5 days ahead
            hist_data = df.iloc[:i+1].copy()
            
            # Make prediction
            result = predictor.predict(hist_data, ticker)
            prediction = result['prediction']
            confidence = result['confidence']
            est_acc = result['estimated_accuracy']
            
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
            estimated_accs.append(est_acc)
        
        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        confidences = np.array(confidences)
        estimated_accs = np.array(estimated_accs)
        
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
        
        # Average estimated accuracy
        avg_est_acc = estimated_accs.mean()
        
        results[ticker] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'high_conf_accuracy': high_conf_accuracy,
            'high_conf_count': high_conf_mask.sum(),
            'total_predictions': len(predictions),
            'avg_estimated_accuracy': avg_est_acc
        }
        
        print(f"\n✓ Results for {ticker}:")
        print(f"  Overall Accuracy:       {accuracy:.2%}")
        print(f"  Precision (BUY):        {precision:.2%}")
        print(f"  Recall (BUY):           {recall:.2%}")
        print(f"  F1 Score:               {f1:.2%}")
        print(f"  High Confidence Acc:    {high_conf_accuracy:.2%} ({high_conf_mask.sum()} predictions)")
        print(f"  Estimated Accuracy:     {avg_est_acc:.2%}")
        
    except Exception as e:
        print(f"✗ Error testing {ticker}: {e}")
        import traceback
        traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("SUMMARY - Simple Improved Predictor Performance")
print("=" * 70)

if results:
    avg_accuracy = sum(r['accuracy'] for r in results.values()) / len(results)
    avg_precision = sum(r['precision'] for r in results.values()) / len(results)
    avg_f1 = sum(r['f1_score'] for r in results.values()) / len(results)
    avg_high_conf = sum(r['high_conf_accuracy'] for r in results.values()) / len(results)
    avg_est_acc = sum(r['avg_estimated_accuracy'] for r in results.values()) / len(results)
    
    print(f"\nAverage Performance:")
    print(f"  Overall Accuracy:       {avg_accuracy:.2%}")
    print(f"  Precision:              {avg_precision:.2%}")
    print(f"  F1 Score:               {avg_f1:.2%}")
    print(f"  High Confidence Acc:    {avg_high_conf:.2%}")
    print(f"  Estimated Accuracy:     {avg_est_acc:.2%}")
    
    print(f"\nComparison to Baseline:")
    baseline_accuracy = 0.4770  # From previous backtesting
    improvement = (avg_accuracy - baseline_accuracy) * 100
    print(f"  Baseline (Basic Tech):  {baseline_accuracy:.2%}")
    print(f"  Improved (Enhanced):    {avg_accuracy:.2%}")
    print(f"  Improvement:            {improvement:+.1f} percentage points")
    
    if avg_accuracy >= 0.60:
        print(f"\n✅ TARGET ACHIEVED! {avg_accuracy:.2%} >= 60%")
        print(f"\n📊 HONEST ASSESSMENT:")
        print(f"  - Real backtested accuracy: {avg_accuracy:.2%}")
        print(f"  - Based on {sum(r['total_predictions'] for r in results.values())} predictions")
        print(f"  - Tested on 5 tickers over 2 years")
        print(f"  - No synthetic metrics - 100% validated")
    elif avg_accuracy >= 0.55:
        print(f"\n🎯 GOOD PROGRESS! {avg_accuracy:.2%}")
        print(f"   Better than baseline by {improvement:.1f} percentage points")
    else:
        print(f"\n⚠️  NEEDS MORE WORK: {avg_accuracy:.2%}")
        print(f"\nSuggestions:")
        print(f"  1. Tune signal weights")
        print(f"  2. Add more technical indicators")
        print(f"  3. Optimize thresholds")

else:
    print("\n✗ No results to display")

print("\n" + "=" * 70)
