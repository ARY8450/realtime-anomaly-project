"""
Direct test of AdvancedTrendPredictor on a single stock
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

from realtime_anomaly_project.advanced_trend_predictor import AdvancedTrendPredictor
from realtime_anomaly_project.performance_optimizer import PerformanceOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print test banner"""
    print("\n" + "="*100)
    print("ADVANCED TREND PREDICTOR - DIRECT TEST")
    print("="*100)
    print("Testing XGBoost, LightGBM, CatBoost Ensemble with Hyperparameter Optimization")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100 + "\n")

def test_predictor():
    """Direct test of AdvancedTrendPredictor"""
    
    print_banner()
    
    ticker = "RELIANCE.NS"
    print(f"Testing on: {ticker}")
    print(f"Fetching 2 years of historical data...\n")
    
    # Fetch data
    df = yf.download(ticker, period="2y", progress=False)
    
    # Fix column names (handle MultiIndex from yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    
    print(f"Downloaded {len(df)} days of data")
    print(f"   Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}\n")
    
    # Create features
    print("Creating advanced features...")
    optimizer = PerformanceOptimizer()
    features_df = optimizer.create_advanced_features(df)
    
    # Create target (next day's direction)
    target = (df['close'].pct_change().shift(-1) > 0.0).astype(int)
    
    # Remove NaN values
    valid_idx = ~(features_df.isna().any(axis=1) | target.isna())
    X = features_df[valid_idx].values
    y = target[valid_idx].values
    
    print(f"Created {X.shape[1]} features for {len(X)} samples\n")
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    print(f"Data split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing:  {len(X_test)} samples\n")
    
    print("="*100)
    print("TRAINING ADVANCED MODELS")
    print("="*100)
    print("This will train and optimize:")
    print("  1. XGBoost with Optuna hyperparameter optimization (30 trials)")
    print("  2. LightGBM with Optuna optimization (30 trials)")
    print("  3. CatBoost with optimized parameters")
    print("  4. Ensemble voting classifier combining all models")
    print("\nThis may take several minutes... Please wait.\n")
    print("="*100 + "\n")
    
    # Initialize and train predictor
    predictor = AdvancedTrendPredictor(target_accuracy=0.85, use_gpu=False)
    
    # Train and evaluate
    results = predictor.train_and_evaluate(X_train, y_train, X_test, y_test)
    
    print("\n" + "="*100)
    print("FINAL RESULTS")
    print("="*100)
    
    best_model_name = results.get('best_model_name', 'unknown')
    best_accuracy = results.get('best_accuracy', 0.0)
    all_results = results.get('all_results', {})
    target_achieved = results.get('target_achieved', False)
    
    print(f"\nBEST MODEL: {best_model_name.upper()}")
    print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"   Target (85%) Achieved: {'YES' if target_achieved else 'NO'}")
    
    print(f"\nALL MODELS PERFORMANCE:")
    print(f"{'Model':<15} {'Accuracy':<12} {'F1 Score':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 100)
    
    for model_name, model_data in all_results.items():
        metrics = model_data.get('metrics', {})
        acc = metrics.get('accuracy', 0.0)
        f1 = metrics.get('f1_weighted', 0.0)
        prec = metrics.get('precision_weighted', 0.0)
        rec = metrics.get('recall_weighted', 0.0)
        
        print(f"{model_name.upper():<15} {acc:.4f} ({acc*100:5.2f}%)  {f1:.4f} ({f1*100:5.2f}%)  {prec:.4f} ({prec*100:5.2f}%)  {rec:.4f} ({rec*100:5.2f}%)")
    
    print("\n" + "="*100)
    print("TEST COMPLETE")
    print("="*100 + "\n")
    
    return results

if __name__ == "__main__":
    try:
        results = test_predictor()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        print(f"\nTest failed: {e}")
