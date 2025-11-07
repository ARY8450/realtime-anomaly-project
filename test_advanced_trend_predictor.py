"""
Test script to demonstrate AdvancedTrendPredictor with real stock data
Shows XGBoost, LightGBM, CatBoost, and LSTM ensemble performance
"""

import sys
import os
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from realtime_anomaly_project.enhanced_data_system_100_accuracy import EnhancedDataSystemFor100Accuracy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print test banner"""
    print("\n" + "="*100)
    print("🚀 ADVANCED TREND PREDICTOR DEMONSTRATION")
    print("="*100)
    print("Testing on Real Nifty-50 Stock Data")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100 + "\n")

def print_results_banner(ticker, results):
    """Print detailed results for a ticker"""
    print("\n" + "─"*100)
    print(f"📊 RESULTS FOR {ticker}")
    print("─"*100)
    
    if results and 'trend_prediction' in results:
        trend = results['trend_prediction']
        
        print(f"\n🎯 PREDICTION DETAILS:")
        print(f"   Signal: {trend.get('prediction', 'N/A')}")
        print(f"   Confidence: {trend.get('confidence', 0):.3f}")
        
        if trend.get('using_advanced_predictor'):
            print(f"\n✅ USING ADVANCED TREND PREDICTOR")
            print(f"   Best Model: {trend.get('best_model', 'N/A').upper()}")
            print(f"   Models Trained: {', '.join(trend.get('models_trained', []))}")
            print(f"   Target Achieved: {'Yes ✅' if trend.get('target_achieved') else 'No'}")
        else:
            print(f"\n⚠️  Using fallback predictor")
        
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"   Accuracy:  {trend.get('accuracy', 0)*100:6.2f}%")
        print(f"   Precision: {trend.get('precision', 0)*100:6.2f}%")
        print(f"   Recall:    {trend.get('recall', 0)*100:6.2f}%")
        print(f"   F1 Score:  {trend.get('f1_score', 0)*100:6.2f}%")
        print(f"   ROC AUC:   {trend.get('roc_auc', 0)*100:6.2f}%")
        print(f"   PR AUC:    {trend.get('pr_auc', 0)*100:6.2f}%")
    else:
        print("❌ No trend prediction results available")
    
    print("─"*100 + "\n")

def test_advanced_predictor():
    """Test AdvancedTrendPredictor on real stock data"""
    
    print_banner()
    
    # Test tickers - select a few from Nifty 50
    test_tickers = [
        'RELIANCE.NS',
        'TCS.NS',
        'HDFCBANK.NS'
    ]
    
    print(f"🔍 Testing on {len(test_tickers)} tickers: {', '.join(test_tickers)}\n")
    
    # Initialize the enhanced data system (this will use AdvancedTrendPredictor)
    logger.info("Initializing Enhanced Data System with AdvancedTrendPredictor...")
    system = EnhancedDataSystemFor100Accuracy(tickers=test_tickers, lookback="2y")
    
    print("\n" + "="*100)
    print("🏋️  RUNNING COMPREHENSIVE ANALYSIS WITH ADVANCED TREND PREDICTOR")
    print("="*100)
    print("This will:")
    print("  1. Fetch historical data for each ticker")
    print("  2. Create advanced features (technical indicators, patterns)")
    print("  3. Train XGBoost with Optuna hyperparameter optimization")
    print("  4. Train LightGBM with optimized parameters")
    print("  5. Train CatBoost (if available)")
    print("  6. Train LSTM with attention mechanism (if TensorFlow available)")
    print("  7. Create ensemble voting classifier")
    print("  8. Evaluate all models and select best performer")
    print("\nThis may take a few minutes...")
    print("="*100 + "\n")
    
    # Run comprehensive analysis
    results = system.run_comprehensive_analysis()
    
    # Display results for each ticker
    print("\n" + "="*100)
    print("📋 COMPREHENSIVE RESULTS SUMMARY")
    print("="*100)
    
    for ticker in test_tickers:
        if ticker in results:
            print_results_banner(ticker, results[ticker])
    
    # Overall summary
    print("\n" + "="*100)
    print("📊 OVERALL SUMMARY")
    print("="*100)
    
    advanced_count = 0
    fallback_count = 0
    total_accuracy = 0
    total_f1 = 0
    
    for ticker in test_tickers:
        if ticker in results and 'trend_prediction' in results[ticker]:
            trend = results[ticker]['trend_prediction']
            if trend.get('using_advanced_predictor'):
                advanced_count += 1
            else:
                fallback_count += 1
            total_accuracy += trend.get('accuracy', 0)
            total_f1 += trend.get('f1_score', 0)
    
    num_tickers = len(test_tickers)
    print(f"\nTickers Analyzed: {num_tickers}")
    print(f"  ✅ Using Advanced Predictor: {advanced_count}")
    print(f"  ⚠️  Using Fallback: {fallback_count}")
    
    if num_tickers > 0:
        avg_accuracy = (total_accuracy / num_tickers) * 100
        avg_f1 = (total_f1 / num_tickers) * 100
        print(f"\nAverage Performance:")
        print(f"  Accuracy: {avg_accuracy:.2f}%")
        print(f"  F1 Score: {avg_f1:.2f}%")
    
    print("\n" + "="*100)
    print("✅ TEST COMPLETE")
    print("="*100 + "\n")
    
    return results

if __name__ == "__main__":
    try:
        results = test_advanced_predictor()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        print(f"\n❌ Test failed: {e}")
